from __future__ import annotations

# Enable CPU fallback for torchvision operators not implemented on MPS.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from sskit import image_to_ground, match
from sskit.coco import LocSimCOCOeval


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO exactly once on a SynLoc split and save a canonical "
            "per-image detection cache. Also evaluate the bbox bottom-center "
            "baseline against SynLoc ground truth when GT is available."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        required=True,
        help="Path to the fine-tuned YOLO best_original.pt checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_yolo_cache"),
    )

    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help=(
            "Low inference threshold used to preserve the full prediction "
            "ranking for all downstream experiments."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument(
        "--bbox-score-threshold",
        type=float,
        default=None,
        help=(
            "Optional fixed score threshold for LocSim/0.25m operating-point "
            "metrics. Leave unset on validation so LocSimCOCOeval selects the "
            "F1-optimal threshold."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute per-image cache files that already exist.",
    )

    return parser.parse_args()


def select_device() -> str | int:
    if torch.cuda.is_available():
        return 0

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def resolve_image_directory(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }

    tried: list[Path] = []

    for name in aliases[split]:
        for candidate in (
            root / "images" / name,
            root / name,
        ):
            tried.append(candidate)
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        f"Could not find image directory for split={split}. Tried: {tried}"
    )


def resolve_annotation_file(
    root: Path,
    split: str,
) -> Path | None:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }

    for name in aliases[split]:
        candidate = root / "annotations" / f"{name}.json"
        if candidate.is_file():
            return candidate

    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_image_metadata(
    annotation_path: Path | None,
) -> dict[str, dict[str, Any]]:
    if annotation_path is None:
        return {}

    dataset = load_json(annotation_path)

    metadata: dict[str, dict[str, Any]] = {}

    for image in dataset.get("images", []):
        file_name = str(image["file_name"])
        metadata[file_name] = image
        metadata[Path(file_name).name] = image

    return metadata


def clip_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    boxes = boxes.copy()

    if len(boxes) == 0:
        return boxes.reshape(0, 4)

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


def bbox_ground_point(
    bbox: np.ndarray,
) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, bbox)
    return (x1 + x2) / 2.0, y2


def scale_point_to_annotation(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> tuple[float, float]:
    return (
        point[0] * annotation_width / actual_width,
        point[1] * annotation_height / actual_height,
    )


def scale_box_to_annotation(
    bbox: np.ndarray,
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)

    sx = annotation_width / actual_width
    sy = annotation_height / actual_height

    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy

    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    ]


def make_detection(
    detection_id: int,
    image_id: int,
    score: float,
    bbox_xywh: list[float],
    ground_point: tuple[float, float],
) -> dict[str, Any]:
    x, y, width, height = bbox_xywh
    ground_x, ground_y = ground_point

    center_x = x + width / 2.0
    center_y = y + height / 2.0

    return {
        "id": int(detection_id),
        "image_id": int(image_id),
        "category_id": 1,
        "score": float(score),
        "bbox": [
            float(x),
            float(y),
            float(width),
            float(height),
        ],
        "area": float(
            max(width, 0.0)
            * max(height, 0.0)
        ),
        "keypoints": [
            float(center_x),
            float(center_y),
            1.0,
            float(ground_x),
            float(ground_y),
            1.0,
        ],
    }


def cache_path_for_image(
    cache_dir: Path,
    image_id: int,
    image_path: Path,
) -> Path:
    if image_id >= 0:
        stem = f"{image_id:06d}_{image_path.stem}"
    else:
        stem = image_path.stem

    return cache_dir / f"{stem}.npz"


def save_detection_cache(
    path: Path,
    *,
    image_id: int,
    file_name: str,
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        image_id=np.asarray(
            image_id,
            dtype=np.int64,
        ),
        file_name=np.asarray(
            file_name,
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
            annotation_width,
            dtype=np.int32,
        ),
        annotation_height=np.asarray(
            annotation_height,
            dtype=np.int32,
        ),
        bboxes=np.asarray(
            boxes,
            dtype=np.float32,
        ),
        detection_scores=np.asarray(
            scores,
            dtype=np.float32,
        ),
    )


def load_detection_cache(
    path: Path,
) -> dict[str, Any]:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            "image_id": int(data["image_id"]),
            "file_name": str(
                data["file_name"].item()
            ),
            "actual_width": int(
                data["actual_width"]
            ),
            "actual_height": int(
                data["actual_height"]
            ),
            "annotation_width": int(
                data["annotation_width"]
            ),
            "annotation_height": int(
                data["annotation_height"]
            ),
            "bboxes": np.asarray(
                data["bboxes"],
                dtype=np.float32,
            ),
            "detection_scores": np.asarray(
                data["detection_scores"],
                dtype=np.float32,
            ),
        }


def project_result_to_pitch(
    result: dict[str, Any],
    image_info: dict[str, Any],
) -> np.ndarray:
    keypoints = np.asarray(
        result["keypoints"],
        dtype=np.float32,
    ).reshape(-1, 3)

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

    if hasattr(pitch, "detach"):
        pitch = (
            pitch[:, :2]
            .detach()
            .cpu()
            .numpy()
        )
    else:
        pitch = np.asarray(
            pitch,
            dtype=np.float32,
        )[:, :2]

    return np.asarray(
        pitch[0],
        dtype=np.float32,
    )


def evaluate_locsim(
    annotation_path: Path,
    results_path: Path,
    fixed_score_threshold: float | None,
) -> dict[str, float]:
    coco_gt = COCO(
        str(annotation_path)
    )
    coco_det = coco_gt.loadRes(
        str(results_path)
    )

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
        evaluator.params.score_threshold = (
            fixed_score_threshold
        )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    result = {
        "mAP_LocSim_50_95": float(
            stats[0]
        ),
        "score_threshold": float(
            stats[15]
        ),
        "frame_accuracy_LocSim_50": float(
            stats[16]
        ),
    }

    # Some sskit versions expose the following extra statistics.
    if len(stats) > 14:
        result.update({
            "precision_LocSim_50": float(
                stats[12]
            ),
            "recall_LocSim_50": float(
                stats[13]
            ),
            "f1_LocSim_50": float(
                stats[14]
            ),
        })

    return result


def evaluate_standard_bbox_coco(
    annotation_path: Path,
    results_path: Path,
) -> dict[str, float]:
    """
    Standard COCO bbox AP/AR for the frozen YOLO detector itself.

    This is separate from SynLoc LocSim. It quantifies detector quality
    against GT bounding boxes before any ground-point estimator is involved.
    """
    coco_gt = COCO(
        str(annotation_path)
    )
    coco_det = coco_gt.loadRes(
        str(results_path)
    )

    evaluator = COCOeval(
        coco_gt,
        coco_det,
        "bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    return {
        "bbox_mAP_50_95": float(
            stats[0]
        ),
        "bbox_AP50": float(
            stats[1]
        ),
        "bbox_AP75": float(
            stats[2]
        ),
        "bbox_AR_max100": float(
            stats[8]
        ),
    }


def compute_baseline_style_stats(
    annotation_path: Path,
    results_path: Path,
    score_threshold: float,
    distance_threshold_m: float = 0.25,
) -> dict[str, float | int | None]:
    dataset = load_json(
        annotation_path
    )

    images_by_id = {
        int(image["id"]): image
        for image in dataset.get(
            "images",
            [],
        )
    }

    gt_by_image: dict[
        int,
        list[np.ndarray],
    ] = defaultdict(list)

    for annotation in dataset.get(
        "annotations",
        [],
    ):
        if int(
            annotation.get(
                "category_id",
                1,
            )
        ) != 1:
            continue

        if "position_on_pitch" not in annotation:
            continue

        gt_by_image[
            int(annotation["image_id"])
        ].append(
            np.asarray(
                annotation[
                    "position_on_pitch"
                ][:2],
                dtype=np.float32,
            )
        )

    predictions_by_image: dict[
        int,
        list[np.ndarray],
    ] = defaultdict(list)

    for result in load_json(
        results_path
    ):
        if (
            float(result["score"])
            < score_threshold
        ):
            continue

        image_id = int(
            result["image_id"]
        )

        predictions_by_image[
            image_id
        ].append(
            project_result_to_pitch(
                result,
                images_by_id[image_id],
            )
        )

    total_detected = 0
    total_missed = 0
    total_extra = 0
    distances_all: list[float] = []

    for image_id in images_by_id:
        gt = np.asarray(
            gt_by_image.get(
                image_id,
                [],
            ),
            dtype=np.float32,
        ).reshape(-1, 2)

        est = np.asarray(
            predictions_by_image.get(
                image_id,
                [],
            ),
            dtype=np.float32,
        ).reshape(-1, 2)

        if len(gt) == 0:
            total_extra += len(est)
            continue

        if len(est) == 0:
            total_missed += len(gt)
            continue

        (
            detected,
            missed,
            extra,
            distances,
            _,
        ) = match(
            gt,
            est,
            th=distance_threshold_m,
        )

        total_detected += int(
            detected
        )
        total_missed += int(
            missed
        )
        total_extra += int(
            extra
        )

        distances_all.extend(
            map(float, distances)
        )

    precision_denominator = (
        total_detected
        + total_extra
    )
    recall_denominator = (
        total_detected
        + total_missed
    )

    precision = (
        total_detected
        / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        total_detected
        / recall_denominator
        if recall_denominator
        else 0.0
    )

    mean_error = (
        float(
            np.mean(
                distances_all
            )
        )
        if distances_all
        else None
    )

    return {
        "distance_threshold_m": (
            distance_threshold_m
        ),
        "detected": total_detected,
        "missed": total_missed,
        "extra": total_extra,
        "precision": precision,
        "recall": recall,
        "mean_matched_localization_error_m": (
            mean_error
        ),
    }


def annotations_have_locsim_gt(
    annotation_path: Path | None,
) -> bool:
    if annotation_path is None:
        return False

    dataset = load_json(
        annotation_path
    )

    images = dataset.get(
        "images",
        [],
    )
    annotations = dataset.get(
        "annotations",
        [],
    )

    if not images or not annotations:
        return False

    image_ok = all(
        "camera_matrix" in image
        and "undist_poly" in image
        for image in images[:10]
    )

    annotation_ok = any(
        "position_on_pitch" in annotation
        for annotation in annotations
    )

    return image_ok and annotation_ok



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def get_model_fingerprint(
    model_argument: str,
) -> dict[str, Any]:
    model_path = Path(
        model_argument
    ).expanduser()

    if model_path.is_file():
        resolved = (
            model_path.resolve()
        )

        return {
            "argument": (
                model_argument
            ),
            "resolved_path": str(
                resolved
            ),
            "sha256": sha256_file(
                resolved
            ),
            "size_bytes": int(
                resolved.stat().st_size
            ),
        }

    # Model names such as yolo26n.pt may be resolved/downloaded by
    # Ultralytics. In that case the literal argument is still recorded,
    # but a local SHA-256 cannot be computed before model loading.
    return {
        "argument": (
            model_argument
        ),
        "resolved_path": None,
        "sha256": None,
        "size_bytes": None,
    }


def validate_existing_cache_configuration(
    summary_path: Path,
    *,
    model_fingerprint: dict[str, Any],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    overwrite: bool,
) -> None:
    if (
        not summary_path.is_file()
        or overwrite
    ):
        return

    previous = load_json(
        summary_path
    )

    mismatches: list[str] = []

    previous_fingerprint = (
        previous.get(
            "yolo_model_fingerprint",
            {},
        )
    )

    previous_sha = (
        previous_fingerprint.get(
            "sha256"
        )
    )
    current_sha = (
        model_fingerprint.get(
            "sha256"
        )
    )

    if (
        previous_sha is not None
        and current_sha is not None
        and previous_sha
        != current_sha
    ):
        mismatches.append(
            "YOLO model SHA-256 differs"
        )
    elif (
        previous_sha is None
        or current_sha is None
    ):
        previous_argument = (
            previous_fingerprint.get(
                "argument",
                previous.get(
                    "yolo_model"
                ),
            )
        )

        if (
            previous_argument
            != model_fingerprint[
                "argument"
            ]
        ):
            mismatches.append(
                "YOLO model argument differs"
            )

    settings = {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
    }

    for key, value in (
        settings.items()
    ):
        previous_value = (
            previous.get(key)
        )

        if (
            previous_value is not None
            and previous_value
            != value
        ):
            mismatches.append(
                f"{key}: cache={previous_value}, "
                f"requested={value}"
            )

    if mismatches:
        formatted = "\n  - ".join(
            mismatches
        )

        raise RuntimeError(
            "Existing canonical YOLO cache was created with a "
            "different detector/configuration:\n"
            f"  - {formatted}\n"
            "Use a different --output-dir, or deliberately pass "
            "--overwrite to rebuild the canonical cache."
        )

def main() -> None:
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )
    output_root = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    model_fingerprint = (
        get_model_fingerprint(
            args.yolo_model
        )
    )

    existing_summary_path = (
        output_root
        / args.split
        / "yolo_cache_summary.json"
    )

    validate_existing_cache_configuration(
        existing_summary_path,
        model_fingerprint=(
            model_fingerprint
        ),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        overwrite=args.overwrite,
    )

    image_directory = (
        resolve_image_directory(
            dataset_root,
            args.split,
        )
    )
    annotation_path = (
        resolve_annotation_file(
            dataset_root,
            args.split,
        )
    )

    metadata = (
        load_image_metadata(
            annotation_path
        )
    )

    cache_dir = (
        output_root
        / args.split
        / "yolo_raw"
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in image_directory.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in "
            f"{image_directory}"
        )

    device = select_device()

    print(
        f"Device:       {device}"
    )
    print(
        f"Split:        {args.split}"
    )
    print(
        f"Images:       "
        f"{image_directory}"
    )
    print(
        f"Annotations:  "
        f"{annotation_path}"
    )
    print(
        f"YOLO model:   "
        f"{args.yolo_model}"
    )
    print(
        f"Output:       "
        f"{output_root}"
    )
    print(
        f"Image count:  "
        f"{len(image_paths)}"
    )

    model = YOLO(
        args.yolo_model
    )

    processed = 0
    skipped = 0
    total_detections = 0

    for image_path in tqdm(
        image_paths,
        desc=f"{args.split} YOLO cache",
    ):
        relative_name = (
            image_path
            .relative_to(
                image_directory
            )
            .as_posix()
        )

        image_info = (
            metadata.get(
                relative_name
            )
            or metadata.get(
                image_path.name
            )
            or {}
        )

        image_id = int(
            image_info.get(
                "id",
                -1,
            )
        )

        cache_path = (
            cache_path_for_image(
                cache_dir,
                image_id,
                image_path,
            )
        )

        if (
            cache_path.is_file()
            and not args.overwrite
        ):
            skipped += 1
            cached = (
                load_detection_cache(
                    cache_path
                )
            )
            total_detections += len(
                cached[
                    "detection_scores"
                ]
            )
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: "
                f"{image_path}"
            )

        actual_height, actual_width = (
            image.shape[:2]
        )

        annotation_width = int(
            image_info.get(
                "width",
                actual_width,
            )
        )
        annotation_height = int(
            image_info.get(
                "height",
                actual_height,
            )
        )

        result = model.predict(
            source=image,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )
            scores = np.empty(
                (0,),
                dtype=np.float32,
            )
        else:
            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            scores = (
                result.boxes.conf
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

        save_detection_cache(
            cache_path,
            image_id=image_id,
            file_name=relative_name,
            actual_width=actual_width,
            actual_height=actual_height,
            annotation_width=annotation_width,
            annotation_height=annotation_height,
            boxes=boxes,
            scores=scores,
        )

        processed += 1
        total_detections += len(
            scores
        )

    summary: dict[str, Any] = {
        "split": args.split,
        "dataset_root": str(
            dataset_root
        ),
        "image_directory": str(
            image_directory
        ),
        "annotation_file": (
            str(annotation_path)
            if annotation_path
            else None
        ),
        "yolo_model": (
            args.yolo_model
        ),
        "yolo_model_fingerprint": (
            model_fingerprint
        ),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "image_count": len(
            image_paths
        ),
        "processed_this_run": (
            processed
        ),
        "skipped_existing": (
            skipped
        ),
        "total_cached_detections": (
            total_detections
        ),
        "cache_contract": {
            "canonical_source": (
                "All downstream bbox, pose, "
                "and SAM experiments must use "
                "these exact bboxes and scores."
            ),
            "arrays": [
                "image_id",
                "file_name",
                "actual_width",
                "actual_height",
                "annotation_width",
                "annotation_height",
                "bboxes",
                "detection_scores",
            ],
            "coordinates": (
                "actual image pixel coordinates"
            ),
        },
    }

    # ---------------------------------------------------------
    # Build one canonical bbox result file from the cache.
    # ---------------------------------------------------------
    bbox_results: list[
        dict[str, Any]
    ] = []
    detection_id = 1

    for cache_path in sorted(
        cache_dir.glob("*.npz")
    ):
        cached = load_detection_cache(
            cache_path
        )

        image_id = int(
            cached["image_id"]
        )

        if image_id < 0:
            continue

        actual_width = int(
            cached["actual_width"]
        )
        actual_height = int(
            cached["actual_height"]
        )
        annotation_width = int(
            cached[
                "annotation_width"
            ]
        )
        annotation_height = int(
            cached[
                "annotation_height"
            ]
        )

        for box, score in zip(
            cached["bboxes"],
            cached[
                "detection_scores"
            ],
        ):
            bbox_ann = (
                scale_box_to_annotation(
                    box,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            point_actual = (
                bbox_ground_point(
                    box
                )
            )
            point_ann = (
                scale_point_to_annotation(
                    point_actual,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            bbox_results.append(
                make_detection(
                    detection_id,
                    image_id,
                    float(score),
                    bbox_ann,
                    point_ann,
                )
            )
            detection_id += 1

    bbox_results_path = (
        output_root
        / args.split
        / "yolo_bbox_results.json"
    )
    save_json(
        bbox_results_path,
        bbox_results,
    )

    # ---------------------------------------------------------
    # Evaluate against GT when the split provides it.
    # ---------------------------------------------------------
    if (
        annotation_path is not None
        and bbox_results
    ):
        try:
            summary[
                "standard_bbox_detection"
            ] = (
                evaluate_standard_bbox_coco(
                    annotation_path,
                    bbox_results_path,
                )
            )
        except Exception as error:
            summary[
                "standard_bbox_detection_error"
            ] = repr(error)

    if (
        annotation_path is not None
        and bbox_results
        and annotations_have_locsim_gt(
            annotation_path
        )
    ):
        locsim = evaluate_locsim(
            annotation_path,
            bbox_results_path,
            args.bbox_score_threshold,
        )

        baseline = (
            compute_baseline_style_stats(
                annotation_path,
                bbox_results_path,
                locsim[
                    "score_threshold"
                ],
            )
        )

        summary[
            "bbox_bottom_center_locsim"
        ] = locsim
        summary[
            "bbox_bottom_center_0_25m"
        ] = baseline

        print(
            "\nCanonical YOLO bbox "
            "bottom-center score:"
        )
        print(
            "mAP-LocSim:        "
            f"{locsim['mAP_LocSim_50_95']:.6f}"
        )

        if (
            "precision_LocSim_50"
            in locsim
        ):
            print(
                "Precision @ 0.50: "
                f"{locsim['precision_LocSim_50']:.6f}"
            )
            print(
                "Recall @ 0.50:    "
                f"{locsim['recall_LocSim_50']:.6f}"
            )
            print(
                "F1 @ 0.50:        "
                f"{locsim['f1_LocSim_50']:.6f}"
            )

        print(
            "Frame accuracy:     "
            f"{locsim['frame_accuracy_LocSim_50']:.6f}"
        )
        print(
            "Selected threshold: "
            f"{locsim['score_threshold']:.10f}"
        )
        print(
            "0.25m detected:     "
            f"{baseline['detected']}"
        )
        print(
            "0.25m missed:       "
            f"{baseline['missed']}"
        )
        print(
            "0.25m extra:        "
            f"{baseline['extra']}"
        )
        print(
            "Mean loc. error:    "
            f"{baseline['mean_matched_localization_error_m']} m"
        )

    summary_path = (
        output_root
        / args.split
        / "yolo_cache_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print("\nSaved:")
    print(
        f"  Cache:   {cache_dir}"
    )
    print(
        f"  Results: {bbox_results_path}"
    )
    print(
        f"  Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
