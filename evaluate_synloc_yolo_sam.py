from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import SAM, YOLO
from xtcocotools.coco import COCO

from sskit import image_to_ground, match
from sskit.coco import LocSimCOCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fine-tuned YOLO detector and SAM-based ground-point "
            "localization on all SynLoc images, save challenge-compatible "
            "results, and evaluate them with mAP-LocSim."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/data/SoccerNet/SpiideoSynLoc"),
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        required=True,
        help="Path to the trained YOLO best_original.pt checkpoint.",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="mobile_sam.pt",
        help="SAM checkpoint/name, e.g. mobile_sam.pt or /path/to/sam3.pt.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_eval"),
    )

    parser.add_argument(
        "--sam-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional root created by cache_synloc_sam.py, e.g. "
            "output/synloc_sam_cache. Existing per-image .npz files are "
            "used instead of rerunning YOLO/SAM. Missing cache files are "
            "computed and saved automatically."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--bottom-band-height", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--bbox-score-threshold",
        type=float,
        default=None,
        help=(
            "Fixed score threshold for evaluation. Normally leave unset on val; "
            "for test, pass the threshold obtained on val for the bbox method."
        ),
    )
    parser.add_argument(
        "--sam-score-threshold",
        type=float,
        default=None,
        help=(
            "Fixed score threshold for evaluation. Normally leave unset on val; "
            "for test, pass the threshold obtained on val for the SAM method."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing progress checkpoint and start from scratch.",
    )

    return parser.parse_args()


def select_device() -> str | int:
    if torch.cuda.is_available():
        return 0

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def resolve_image_directory(dataset_root: Path, split: str) -> Path:
    candidates = {
        "val": [
            dataset_root / "images" / "val",
            dataset_root / "val",
            dataset_root / "valid",
            dataset_root / "validation",
        ],
        "test": [
            dataset_root / "images" / "test",
            dataset_root / "test",
        ],
    }

    for candidate in candidates[split]:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find image directory for split={split}. "
        f"Tried: {candidates[split]}"
    )


def resolve_annotation_file(dataset_root: Path, split: str) -> Path:
    candidates = {
        "val": [
            dataset_root / "annotations" / "val.json",
            dataset_root / "annotations" / "valid.json",
            dataset_root / "annotations" / "validation.json",
        ],
        "test": [
            dataset_root / "annotations" / "test.json",
        ],
    }

    for candidate in candidates[split]:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find annotation file for split={split}. "
        f"Tried: {candidates[split]}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    temporary_path.replace(path)


def clip_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    boxes = boxes.copy()

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        image_width - 1,
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        image_height - 1,
    )

    return boxes


def resize_mask_to_image(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if mask.shape == (image_height, image_width):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


def find_mask_ground_point_robust(
    mask: np.ndarray,
    bbox: np.ndarray,
    bottom_percentile: float = 98.0,
    band_fraction: float = 0.05,
) -> tuple[int, int] | None:
    """
    Estimate the player's ground-contact point robustly.

    Improvements over the simple lowest-pixel method:
    1. Restrict the segmentation to the YOLO bbox.
    2. Keep only the largest connected mask component.
    3. Use a percentile instead of the absolute lowest pixel.
    4. Estimate x from a bottom band whose height scales with player size.
    """

    image_height, image_width = mask.shape

    x1, y1, x2, y2 = bbox

    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(image_width - 1, int(np.ceil(x2)))
    y2 = min(image_height - 1, int(np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    # ---------------------------------------------------------
    # 1. Restrict SAM mask to the YOLO bbox
    # ---------------------------------------------------------

    cropped_mask = np.zeros_like(
        mask,
        dtype=np.uint8,
    )

    cropped_mask[y1:y2 + 1, x1:x2 + 1] = (
        mask[y1:y2 + 1, x1:x2 + 1]
        .astype(np.uint8)
    )

    if cropped_mask.sum() == 0:
        return None

    # ---------------------------------------------------------
    # 2. Keep only the largest connected component
    # ---------------------------------------------------------

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cropped_mask,
            connectivity=8,
        )
    )

    if number_of_labels <= 1:
        return None

    # Ignore label 0 because it is the background.
    largest_label = (
        1
        + np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )
    )

    clean_mask = labels == largest_label

    ys, xs = np.nonzero(clean_mask)

    if len(xs) == 0:
        return None

    # ---------------------------------------------------------
    # 3. Robust bottom level
    # ---------------------------------------------------------

    robust_bottom_y = int(
        round(
            np.percentile(
                ys,
                bottom_percentile,
            )
        )
    )

    # ---------------------------------------------------------
    # 4. Bottom band proportional to player bbox height
    # ---------------------------------------------------------

    bbox_height = y2 - y1 + 1

    band_height = max(
        2,
        int(round(bbox_height * band_fraction)),
    )

    lower_y = max(
        y1,
        robust_bottom_y - band_height,
    )

    bottom_pixels = (
        (ys >= lower_y)
        & (ys <= robust_bottom_y)
    )

    bottom_xs = xs[bottom_pixels]

    if len(bottom_xs) == 0:
        return None

    ground_x = int(
        round(np.median(bottom_xs))
    )

    ground_y = robust_bottom_y

    return ground_x, ground_y


def find_mask_ground_point(
    mask: np.ndarray,
    bottom_band_height: int,
) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    lowest_y = int(ys.max())
    bottom_pixels = ys >= lowest_y - bottom_band_height

    bottom_xs = xs[bottom_pixels]
    bottom_ys = ys[bottom_pixels]

    return int(np.median(bottom_xs)), int(np.median(bottom_ys))


def bbox_ground_point(box: np.ndarray) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, y2


def scale_point_to_annotation(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> tuple[float, float]:
    scale_x = annotation_width / actual_width
    scale_y = annotation_height / actual_height

    return point[0] * scale_x, point[1] * scale_y


def scale_box_to_annotation(
    box: np.ndarray,
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    scale_x = annotation_width / actual_width
    scale_y = annotation_height / actual_height

    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    return [x1, y1, x2 - x1, y2 - y1]


def make_detection(
    detection_id: int,
    image_id: int,
    score: float,
    bbox_xywh: list[float],
    center_point: tuple[float, float],
    ground_point: tuple[float, float],
) -> dict[str, Any]:
    x, y, width, height = bbox_xywh
    center_x, center_y = center_point
    ground_x, ground_y = ground_point

    # Two keypoints are used because the SynLoc ground-truth annotations
    # contain keypoint 0 = pelvis and keypoint 1 = ground-projected point.
    # LocSimCOCOeval will be configured with position_from_keypoint_index=1.
    keypoints = [
        float(center_x),
        float(center_y),
        1.0,
        float(ground_x),
        float(ground_y),
        1.0,
    ]

    return {
        "id": detection_id,
        "image_id": image_id,
        "category_id": 1,
        "score": float(score),
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(max(width, 0.0) * max(height, 0.0)),
        "keypoints": keypoints,
    }



def get_sam_cache_path(
    cache_root: Path,
    split: str,
    image_info: dict[str, Any],
) -> Path:
    """
    Return the per-image cache path produced by cache_synloc_sam.py.

    Expected layout:
        <cache_root>/<split>/sam_raw/<image_id>_<image_stem>.npz
    """
    image_id = int(image_info["id"])
    image_stem = Path(str(image_info["file_name"])).stem

    return (
        cache_root.resolve()
        / split
        / "sam_raw"
        / f"{image_id:06d}_{image_stem}.npz"
    )


def load_sam_cache(
    cache_path: Path,
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load cached YOLO boxes, YOLO scores, and raw SAM masks.

    Coordinates and masks are in the actual image resolution.
    """
    with np.load(
        cache_path,
        allow_pickle=False,
    ) as data:
        actual_width = int(data["actual_width"])
        actual_height = int(data["actual_height"])
        boxes = np.asarray(
            data["bboxes"],
            dtype=np.float32,
        )
        confidences = np.asarray(
            data["detection_scores"],
            dtype=np.float32,
        )
        masks = np.asarray(
            data["masks"],
            dtype=np.uint8,
        )

    return (
        actual_width,
        actual_height,
        boxes,
        confidences,
        masks,
    )


def save_sam_cache(
    cache_path: Path,
    image_info: dict[str, Any],
    actual_width: int,
    actual_height: int,
    boxes: np.ndarray,
    confidences: np.ndarray,
    masks: np.ndarray,
) -> None:
    """
    Save a cache entry compatible with cache_synloc_sam.py.
    """
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        cache_path,
        image_id=np.asarray(
            int(image_info["id"]),
            dtype=np.int64,
        ),
        file_name=np.asarray(
            str(image_info["file_name"])
        ),
        actual_width=np.asarray(
            actual_width,
            dtype=np.int32,
        ),
        actual_height=np.asarray(
            actual_height,
            dtype=np.int32,
        ),
        annotation_width=np.asarray(
            int(image_info["width"]),
            dtype=np.int32,
        ),
        annotation_height=np.asarray(
            int(image_info["height"]),
            dtype=np.int32,
        ),
        bboxes=np.asarray(
            boxes,
            dtype=np.float32,
        ),
        detection_scores=np.asarray(
            confidences,
            dtype=np.float32,
        ),
        masks=np.asarray(
            masks,
            dtype=np.uint8,
        ),
    )


def prepare_cached_masks_for_selected_detections(
    masks: np.ndarray,
    selected_original_indices: np.ndarray,
) -> list[np.ndarray | None]:
    """
    Preserve the original detection-to-mask index relationship.

    The original evaluator assumes SAM mask i corresponds to YOLO bbox i.
    If the cache contains fewer masks than boxes, detections beyond the mask
    count receive None and therefore use the existing bbox fallback.
    """
    selected_masks: list[np.ndarray | None] = []

    for original_index in selected_original_indices:
        index = int(original_index)

        if index < len(masks):
            selected_masks.append(masks[index])
        else:
            selected_masks.append(None)

    return selected_masks

def save_progress(
    path: Path,
    completed_image_ids: set[int],
    bbox_results: list[dict[str, Any]],
    sam_results: list[dict[str, Any]],
    next_detection_id: int,
    sam_fallback_count: int,
) -> None:
    save_json_atomic(
        path,
        {
            "completed_image_ids": sorted(completed_image_ids),
            "bbox_results": bbox_results,
            "sam_results": sam_results,
            "next_detection_id": next_detection_id,
            "sam_fallback_count": sam_fallback_count,
        },
    )


def run_inference(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, int],
]:
    dataset_root = args.dataset_root.resolve()
    image_directory = resolve_image_directory(
        dataset_root,
        args.split,
    )
    annotation_path = resolve_annotation_file(
        dataset_root,
        args.split,
    )

    dataset = load_json(annotation_path)
    images = dataset.get("images", [])

    output_dir = (
        args.output_dir.resolve()
        / args.split
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bbox_results_path = (
        output_dir
        / "yolo_bbox_results.json"
    )
    sam_results_path = (
        output_dir
        / "yolo_sam_results.json"
    )
    progress_path = (
        output_dir
        / "progress.json"
    )
    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    completed_image_ids: set[int] = set()
    bbox_results: list[dict[str, Any]] = []
    sam_results: list[dict[str, Any]] = []
    next_detection_id = 1
    sam_fallback_count = 0

    cache_stats = {
        "hits": 0,
        "misses": 0,
        "new_files_written": 0,
        "mask_count_mismatches": 0,
    }

    if (
        progress_path.is_file()
        and not args.restart
    ):
        state = load_json(
            progress_path
        )

        completed_image_ids = set(
            map(
                int,
                state[
                    "completed_image_ids"
                ],
            )
        )
        bbox_results = state[
            "bbox_results"
        ]
        sam_results = state[
            "sam_results"
        ]
        next_detection_id = int(
            state[
                "next_detection_id"
            ]
        )
        sam_fallback_count = int(
            state.get(
                "sam_fallback_count",
                0,
            )
        )

        print(
            f"Resuming from {progress_path}: "
            f"{len(completed_image_ids)}/"
            f"{len(images)} images completed."
        )

    sam_cache_root = (
        args.sam_cache_root.resolve()
        if args.sam_cache_root is not None
        else None
    )

    if sam_cache_root is not None:
        cache_directory = (
            sam_cache_root
            / args.split
            / "sam_raw"
        )
        cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"SAM cache:      "
            f"{cache_directory}"
        )

        cache_summary_path = (
            sam_cache_root
            / args.split
            / "sam_cache_summary.json"
        )

        if cache_summary_path.is_file():
            cache_summary = load_json(
                cache_summary_path
            )

            cached_conf = (
                cache_summary.get("conf")
            )

            if (
                cached_conf is not None
                and args.conf
                < float(cached_conf)
            ):
                raise ValueError(
                    "The SAM cache was generated with "
                    f"YOLO conf={cached_conf}, but this "
                    f"evaluation requests conf={args.conf}. "
                    "Detections below the cache threshold "
                    "were never stored, so they cannot be "
                    "recovered. Regenerate the cache using "
                    "a confidence <= the requested value."
                )

            for key, requested in (
                ("imgsz", args.imgsz),
                ("iou", args.iou),
                ("max_det", args.max_det),
            ):
                cached = (
                    cache_summary.get(key)
                )

                if (
                    cached is not None
                    and cached != requested
                ):
                    print(
                        "WARNING: cached YOLO setting "
                        f"{key}={cached}, while this run "
                        f"requests {key}={requested}. "
                        "Cached frames will use the "
                        "predictions already stored."
                    )

            cached_sam_model = (
                cache_summary.get(
                    "sam_model"
                )
            )
            if (
                cached_sam_model is not None
                and str(cached_sam_model)
                != str(args.sam_model)
            ):
                print(
                    "WARNING: cache summary says "
                    f"SAM model={cached_sam_model}, "
                    f"while this run specifies "
                    f"{args.sam_model}. Cached frames "
                    "will still use the masks that are "
                    "already stored."
                )

    device = select_device()

    print(f"Device:        {device}")
    print(f"Split:         {args.split}")
    print(f"Images:        {image_directory}")
    print(f"Annotations:   {annotation_path}")
    print(f"YOLO model:    {args.yolo_model}")
    print(f"SAM model:     {args.sam_model}")
    print(f"Output:        {output_dir}")
    print(f"YOLO conf:     {args.conf}")

    # Lazy initialization. If all pending images are cached, YOLO and SAM are
    # never loaded at all.
    yolo_model: YOLO | None = None
    sam_model: SAM | None = None

    def ensure_models_loaded() -> tuple[
        YOLO,
        SAM,
    ]:
        nonlocal yolo_model, sam_model

        if yolo_model is None:
            print(
                "\nLoading YOLO because a "
                "cache miss was encountered..."
            )
            yolo_model = YOLO(
                args.yolo_model
            )

        if sam_model is None:
            print(
                "Loading SAM because a "
                "cache miss was encountered..."
            )
            sam_model = SAM(
                args.sam_model
            )

        return (
            yolo_model,
            sam_model,
        )

    pending_images = [
        image_info
        for image_info in images
        if int(image_info["id"])
        not in completed_image_ids
    ]

    for processed_count, image_info in enumerate(
        tqdm(
            pending_images,
            desc=f"{args.split} inference",
        ),
        start=1,
    ):
        image_id = int(
            image_info["id"]
        )
        file_name = str(
            image_info["file_name"]
        )
        image_path = (
            image_directory
            / file_name
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        annotation_width = int(
            image_info["width"]
        )
        annotation_height = int(
            image_info["height"]
        )

        cache_path: Path | None = None

        if sam_cache_root is not None:
            cache_path = get_sam_cache_path(
                sam_cache_root,
                args.split,
                image_info,
            )

        # -------------------------------------------------------------
        # Fast path: cached YOLO detections + raw SAM masks.
        # -------------------------------------------------------------
        if (
            cache_path is not None
            and cache_path.is_file()
        ):
            (
                actual_width,
                actual_height,
                all_boxes,
                all_confidences,
                all_masks,
            ) = load_sam_cache(
                cache_path
            )

            cache_stats["hits"] += 1

            selected_original_indices = (
                np.flatnonzero(
                    all_confidences
                    >= float(args.conf)
                )
            )

            boxes = (
                all_boxes[
                    selected_original_indices
                ]
            )
            confidences = (
                all_confidences[
                    selected_original_indices
                ]
            )

            boxes = clip_boxes(
                boxes,
                actual_width,
                actual_height,
            )

            selected_masks = (
                prepare_cached_masks_for_selected_detections(
                    all_masks,
                    selected_original_indices,
                )
            )

            if (
                len(all_masks)
                != len(all_boxes)
            ):
                cache_stats[
                    "mask_count_mismatches"
                ] += 1

        # -------------------------------------------------------------
        # Slow path: run YOLO + SAM for this missing image and cache it.
        # -------------------------------------------------------------
        else:
            cache_stats["misses"] += 1

            image = cv2.imread(
                str(image_path)
            )
            if image is None:
                raise RuntimeError(
                    f"Could not read image: "
                    f"{image_path}"
                )

            (
                actual_height,
                actual_width,
            ) = image.shape[:2]

            (
                yolo_model_loaded,
                sam_model_loaded,
            ) = ensure_models_loaded()

            yolo_result = (
                yolo_model_loaded.predict(
                    source=image,
                    classes=[0],
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    device=device,
                    verbose=False,
                )[0]
            )

            if (
                yolo_result.boxes is None
                or len(
                    yolo_result.boxes
                ) == 0
            ):
                boxes = np.empty(
                    (0, 4),
                    dtype=np.float32,
                )
                confidences = np.empty(
                    (0,),
                    dtype=np.float32,
                )
                raw_masks_for_cache = np.empty(
                    (
                        0,
                        actual_height,
                        actual_width,
                    ),
                    dtype=np.uint8,
                )
                selected_masks = []

            else:
                boxes = (
                    yolo_result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                confidences = (
                    yolo_result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                boxes = clip_boxes(
                    boxes,
                    actual_width,
                    actual_height,
                )

                raw_masks_for_cache = np.empty(
                    (
                        0,
                        actual_height,
                        actual_width,
                    ),
                    dtype=np.uint8,
                )

                sam_prediction = (
                    sam_model_loaded.predict(
                        source=image,
                        bboxes=boxes.tolist(),
                        device=device,
                        verbose=False,
                        save=False,
                    )
                )

                if (
                    sam_prediction
                    and sam_prediction[0].masks
                    is not None
                ):
                    raw_masks = (
                        sam_prediction[0]
                        .masks.data
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    resized_masks = [
                        resize_mask_to_image(
                            mask,
                            actual_width,
                            actual_height,
                        ).astype(np.uint8)
                        for mask in raw_masks
                    ]

                    if resized_masks:
                        raw_masks_for_cache = (
                            np.stack(
                                resized_masks,
                                axis=0,
                            )
                        )

                selected_masks = [
                    (
                        raw_masks_for_cache[index]
                        if index
                        < len(
                            raw_masks_for_cache
                        )
                        else None
                    )
                    for index
                    in range(len(boxes))
                ]

                if (
                    len(raw_masks_for_cache)
                    != len(boxes)
                ):
                    cache_stats[
                        "mask_count_mismatches"
                    ] += 1

            if cache_path is not None:
                save_sam_cache(
                    cache_path=cache_path,
                    image_info=image_info,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    boxes=boxes,
                    confidences=confidences,
                    masks=raw_masks_for_cache,
                )
                cache_stats[
                    "new_files_written"
                ] += 1

        if len(boxes) == 0:
            completed_image_ids.add(
                image_id
            )

            if (
                processed_count
                % args.checkpoint_every
                == 0
            ):
                save_progress(
                    progress_path,
                    completed_image_ids,
                    bbox_results,
                    sam_results,
                    next_detection_id,
                    sam_fallback_count,
                )

            continue

        for index, (
            box,
            score,
        ) in enumerate(
            zip(
                boxes,
                confidences,
            )
        ):
            x1, y1, x2, y2 = map(
                float,
                box,
            )

            center_actual = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )
            bbox_ground_actual = (
                bbox_ground_point(
                    box
                )
            )

            bbox_xywh_ann = (
                scale_box_to_annotation(
                    box,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )
            center_ann = (
                scale_point_to_annotation(
                    center_actual,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )
            bbox_ground_ann = (
                scale_point_to_annotation(
                    bbox_ground_actual,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            bbox_results.append(
                make_detection(
                    detection_id=(
                        next_detection_id
                    ),
                    image_id=image_id,
                    score=float(score),
                    bbox_xywh=(
                        bbox_xywh_ann
                    ),
                    center_point=(
                        center_ann
                    ),
                    ground_point=(
                        bbox_ground_ann
                    ),
                )
            )

            sam_ground_actual: (
                tuple[float, float]
                | None
            ) = None

            mask_for_detection = (
                selected_masks[index]
                if index
                < len(selected_masks)
                else None
            )

            if mask_for_detection is not None:
                # Cached masks are already actual-image-sized, while a cache
                # created externally might not be. Keep the same safeguard.
                mask = resize_mask_to_image(
                    mask_for_detection,
                    actual_width,
                    actual_height,
                )

                # Keep the current robust ground-point estimator unchanged.
                sam_ground_actual = (
                    find_mask_ground_point_robust(
                        mask,
                        box,
                    )
                )

            if sam_ground_actual is None:
                sam_ground_actual = (
                    bbox_ground_actual
                )
                sam_fallback_count += 1

            sam_ground_ann = (
                scale_point_to_annotation(
                    sam_ground_actual,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            sam_results.append(
                make_detection(
                    detection_id=(
                        next_detection_id
                    ),
                    image_id=image_id,
                    score=float(score),
                    bbox_xywh=(
                        bbox_xywh_ann
                    ),
                    center_point=(
                        center_ann
                    ),
                    ground_point=(
                        sam_ground_ann
                    ),
                )
            )

            next_detection_id += 1

        completed_image_ids.add(
            image_id
        )

        if (
            processed_count
            % args.checkpoint_every
            == 0
        ):
            save_progress(
                progress_path,
                completed_image_ids,
                bbox_results,
                sam_results,
                next_detection_id,
                sam_fallback_count,
            )

    save_json_atomic(
        bbox_results_path,
        bbox_results,
    )
    save_json_atomic(
        sam_results_path,
        sam_results,
    )

    save_progress(
        progress_path,
        completed_image_ids,
        bbox_results,
        sam_results,
        next_detection_id,
        sam_fallback_count,
    )

    print(
        f"\nYOLO bbox detections: "
        f"{len(bbox_results)}"
    )
    print(
        f"YOLO+SAM detections:  "
        f"{len(sam_results)}"
    )
    print(
        f"SAM bbox fallbacks:   "
        f"{sam_fallback_count}"
    )

    if sam_cache_root is not None:
        print(
            "SAM cache:"
            f"\n  hits:                 "
            f"{cache_stats['hits']}"
            f"\n  misses:               "
            f"{cache_stats['misses']}"
            f"\n  new files written:    "
            f"{cache_stats['new_files_written']}"
            f"\n  mask-count mismatches:"
            f" {cache_stats['mask_count_mismatches']}"
        )

    return (
        annotation_path,
        bbox_results_path,
        sam_results_path,
        summary_path,
        cache_stats,
    )



def evaluate_locsim(
    annotation_path: Path,
    results_path: Path,
    fixed_score_threshold: float | None,
) -> dict[str, float]:
    coco_gt = COCO(str(annotation_path))
    coco_det = coco_gt.loadRes(str(results_path))

    evaluator = LocSimCOCOeval(
        coco_gt,
        coco_det,
        "bbox",
        [0.089, 0.089],
        True,
    )
    evaluator.params.useSegm = None
    evaluator.params.position_from_keypoint_index = 1

    if fixed_score_threshold is not None:
        evaluator.params.score_threshold = fixed_score_threshold

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    return {
        "mAP_LocSim_50_95": float(stats[0]),
        "precision_LocSim_50": float(stats[12]),
        "recall_LocSim_50": float(stats[13]),
        "f1_LocSim_50": float(stats[14]),
        "score_threshold": float(stats[15]),
        "frame_accuracy_LocSim_50": float(stats[16]),
    }


def project_result_to_pitch(
    result: dict[str, Any],
    image_info: dict[str, Any],
) -> np.ndarray:
    keypoints = np.asarray(result["keypoints"], dtype=np.float32).reshape(-1, 3)
    x, y = keypoints[1, :2]

    width = float(image_info["width"])
    height = float(image_info["height"])

    normalized = np.asarray(
        [[
            (x - (width - 1.0) / 2.0) / width,
            (y - (height - 1.0) / 2.0) / width,
        ]],
        dtype=np.float32,
    )

    pitch = image_to_ground(
        image_info["camera_matrix"],
        image_info["undist_poly"],
        normalized,
    )

    return pitch[:, :2].detach().cpu().numpy()[0]


def compute_baseline_style_stats(
    annotation_path: Path,
    results_path: Path,
    score_threshold: float,
    distance_threshold_m: float = 0.25,
) -> dict[str, float | int | None]:
    dataset = load_json(annotation_path)

    images_by_id = {
        int(image["id"]): image
        for image in dataset.get("images", [])
    }

    gt_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        if int(annotation.get("category_id", 1)) != 1:
            continue

        position = np.asarray(
            annotation["position_on_pitch"][:2],
            dtype=np.float32,
        )
        gt_by_image[int(annotation["image_id"])].append(position)

    predictions_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for result in load_json(results_path):
        if float(result["score"]) < score_threshold:
            continue

        image_id = int(result["image_id"])
        pitch_position = project_result_to_pitch(
            result,
            images_by_id[image_id],
        )
        predictions_by_image[image_id].append(pitch_position)

    total_detected = 0
    total_missed = 0
    total_extra = 0
    all_distances: list[float] = []

    for image_id in images_by_id:
        gt = np.asarray(
            gt_by_image.get(image_id, []),
            dtype=np.float32,
        ).reshape(-1, 2)
        est = np.asarray(
            predictions_by_image.get(image_id, []),
            dtype=np.float32,
        ).reshape(-1, 2)

        if len(gt) == 0:
            total_extra += len(est)
            continue

        if len(est) == 0:
            total_missed += len(gt)
            continue

        detected, missed, extra, distances, _ = match(
            gt,
            est,
            th=distance_threshold_m,
        )

        total_detected += int(detected)
        total_missed += int(missed)
        total_extra += int(extra)
        all_distances.extend(map(float, distances))

    denominator_precision = total_detected + total_extra
    denominator_recall = total_detected + total_missed

    precision = (
        total_detected / denominator_precision
        if denominator_precision > 0
        else 0.0
    )
    recall = (
        total_detected / denominator_recall
        if denominator_recall > 0
        else 0.0
    )

    mean_error = (
        float(np.mean(all_distances))
        if all_distances
        else None
    )

    return {
        "distance_threshold_m": distance_threshold_m,
        "detected": total_detected,
        "missed": total_missed,
        "extra": total_extra,
        "precision": precision,
        "recall": recall,
        "mean_matched_localization_error_m": mean_error,
    }


def print_method_summary(name: str, summary: dict[str, Any]) -> None:
    locsim = summary["locsim"]
    baseline = summary["baseline_style"]

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"mAP-LocSim @ 0.50:0.95:  {locsim['mAP_LocSim_50_95']:.6f}")
    print(f"Precision @ LocSim 0.50: {locsim['precision_LocSim_50']:.6f}")
    print(f"Recall @ LocSim 0.50:    {locsim['recall_LocSim_50']:.6f}")
    print(f"F1 @ LocSim 0.50:        {locsim['f1_LocSim_50']:.6f}")
    print(f"Frame accuracy:           {locsim['frame_accuracy_LocSim_50']:.6f}")
    print(f"Selected score threshold: {locsim['score_threshold']:.6f}")
    print()
    print("Baseline-style 0.25 m matching:")
    print(f"Detected:                  {baseline['detected']}")
    print(f"Missed:                    {baseline['missed']}")
    print(f"Extra:                     {baseline['extra']}")
    print(f"Precision:                 {baseline['precision']:.6f}")
    print(f"Recall:                    {baseline['recall']:.6f}")

    mean_error = baseline["mean_matched_localization_error_m"]
    if mean_error is None:
        print("Mean localization error:   n/a")
    else:
        print(f"Mean localization error:   {mean_error:.6f} m")


def main() -> None:
    args = parse_args()

    (
        annotation_path,
        bbox_results_path,
        sam_results_path,
        summary_path,
        cache_stats,
    ) = run_inference(args)

    print("\nEvaluating YOLO bbox bottom-center...")
    bbox_locsim = evaluate_locsim(
        annotation_path,
        bbox_results_path,
        args.bbox_score_threshold,
    )
    bbox_baseline = compute_baseline_style_stats(
        annotation_path,
        bbox_results_path,
        bbox_locsim["score_threshold"],
    )

    print("\nEvaluating YOLO + SAM mask ground-point...")
    sam_locsim = evaluate_locsim(
        annotation_path,
        sam_results_path,
        args.sam_score_threshold,
    )
    sam_baseline = compute_baseline_style_stats(
        annotation_path,
        sam_results_path,
        sam_locsim["score_threshold"],
    )

    summary = {
        "split": args.split,
        "yolo_model": args.yolo_model,
        "sam_model": args.sam_model,
        "imgsz": args.imgsz,
        "inference_conf": args.conf,
        "nms_iou": args.iou,
        "sam_cache_root": (
            str(args.sam_cache_root.resolve())
            if args.sam_cache_root is not None
            else None
        ),
        "sam_cache_stats_this_run": dict(
            cache_stats
        ),
        "bbox": {
            "results_file": str(bbox_results_path),
            "locsim": bbox_locsim,
            "baseline_style": bbox_baseline,
        },
        "sam": {
            "results_file": str(sam_results_path),
            "locsim": sam_locsim,
            "baseline_style": sam_baseline,
        },
    }

    save_json_atomic(summary_path, summary)

    print_method_summary("YOLO — bbox bottom-center", summary["bbox"])
    print_method_summary("YOLO + SAM — mask bottom point", summary["sam"])

    print("\nSaved:")
    print(f"  {bbox_results_path}")
    print(f"  {sam_results_path}")
    print(f"  {summary_path}")

    if args.split == "val":
        print("\nFor an unbiased test-set evaluation, reuse these validation thresholds:")
        print(
            "  --bbox-score-threshold "
            f"{bbox_locsim['score_threshold']:.10f}"
        )
        print(
            "  --sam-score-threshold "
            f"{sam_locsim['score_threshold']:.10f}"
        )


if __name__ == "__main__":
    main()
