from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from sskit import image_to_ground


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SAM ground-point localization error on SoccerNet SynLoc. "
            "Uses frozen YOLO detections and cached SAM masks."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--yolo-cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sam-cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.4302865565,
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--bottom-band-frac",
        type=float,
        default=0.05,
        help="Bottom band height as fraction of bbox height.",
    )
    parser.add_argument(
        "--bottom-y-percentile",
        type=float,
        default=98.0,
        help="Percentile used to estimate the bottom y of the mask.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional number of cache images to analyze for a quick smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_sam_localization_debug"),
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resolve_annotation_file(
    dataset_root: Path,
    split: str,
) -> Path:
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


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
        }


def scalar_int(value: np.ndarray) -> int:
    return int(np.asarray(value).item())


def scalar_str(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def xywh_to_xyxy(
    bbox: list[float] | np.ndarray,
) -> np.ndarray:
    x, y, width, height = map(float, bbox)
    return np.asarray(
        [x, y, x + width, y + height],
        dtype=np.float32,
    )


def scale_box_actual_to_annotation(
    bbox: np.ndarray,
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> np.ndarray:
    sx = annotation_width / actual_width
    sy = annotation_height / actual_height

    x1, y1, x2, y2 = map(float, bbox)

    return np.asarray(
        [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
        dtype=np.float32,
    )


def scale_point_actual_to_annotation(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> tuple[float, float]:
    return (
        float(point[0]) * annotation_width / actual_width,
        float(point[1]) * annotation_height / actual_height,
    )


def scale_point_annotation_to_actual(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
) -> tuple[float, float]:
    return (
        float(point[0]) * actual_width / annotation_width,
        float(point[1]) * actual_height / annotation_height,
    )


def pairwise_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros(
            (len(boxes_a), len(boxes_b)),
            dtype=np.float32,
        )

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    width = np.maximum(0.0, x2 - x1)
    height = np.maximum(0.0, y2 - y1)

    intersection = width * height

    area_a = np.maximum(
        0.0,
        (a[..., 2] - a[..., 0])
        * (a[..., 3] - a[..., 1]),
    )
    area_b = np.maximum(
        0.0,
        (b[..., 2] - b[..., 0])
        * (b[..., 3] - b[..., 1]),
    )

    union = area_a + area_b - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def match_detections_to_gt(
    detection_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    min_iou: float,
) -> dict[int, tuple[int, float]]:
    if len(detection_boxes) == 0 or len(gt_boxes) == 0:
        return {}

    ious = pairwise_iou(
        detection_boxes,
        gt_boxes,
    )

    detection_indices, gt_indices = (
        linear_sum_assignment(
            1.0 - ious
        )
    )

    matches: dict[int, tuple[int, float]] = {}

    for det_index, gt_index in zip(
        detection_indices,
        gt_indices,
    ):
        iou = float(
            ious[
                det_index,
                gt_index,
            ]
        )

        if iou >= min_iou:
            matches[int(det_index)] = (
                int(gt_index),
                iou,
            )

    return matches


def get_gt_ground_point_annotation(
    annotation: dict[str, Any],
) -> tuple[float, float] | None:
    raw = annotation.get("keypoints")

    if raw is None:
        return None

    try:
        keypoints = np.asarray(
            raw,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None

    if keypoints.ndim == 1:
        if keypoints.size < 6:
            return None

        x = float(keypoints[3])
        y = float(keypoints[4])

    elif (
        keypoints.ndim >= 2
        and keypoints.shape[0] >= 2
        and keypoints.shape[1] >= 2
    ):
        x = float(keypoints[1, 0])
        y = float(keypoints[1, 1])

    else:
        return None

    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        return None

    return x, y


def project_to_pitch(
    point_annotation: tuple[float, float],
    image_info: dict[str, Any],
) -> np.ndarray:
    x, y = point_annotation

    width = float(
        image_info["width"]
    )
    height = float(
        image_info["height"]
    )

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


def largest_connected_component(
    mask: np.ndarray,
) -> np.ndarray:
    binary = (
        np.asarray(mask) > 0
    ).astype(np.uint8)

    if not np.any(binary):
        return np.zeros_like(
            binary,
            dtype=bool,
        )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return binary.astype(bool)

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_label = (
        int(np.argmax(component_areas))
        + 1
    )

    return labels == largest_label


def get_sam_ground_point(
    mask: np.ndarray,
    bbox: np.ndarray,
    bottom_band_frac: float,
    bottom_y_percentile: float,
) -> tuple[float, float] | None:
    """
    Robust SAM ground-point estimator used in the previous evaluation:

    1. restrict mask to the YOLO bbox
    2. retain largest connected component
    3. y = requested lower-mask percentile (default 98th)
    4. x = median x in a bottom band equal to 5% of bbox height
    """
    x1, y1, x2, y2 = map(float, bbox)

    height, width = mask.shape[:2]

    ix1 = max(
        0,
        min(
            width - 1,
            int(np.floor(x1)),
        ),
    )
    iy1 = max(
        0,
        min(
            height - 1,
            int(np.floor(y1)),
        ),
    )
    ix2 = max(
        ix1 + 1,
        min(
            width,
            int(np.ceil(x2)),
        ),
    )
    iy2 = max(
        iy1 + 1,
        min(
            height,
            int(np.ceil(y2)),
        ),
    )

    # Process only the player's bbox region. The previous implementation
    # allocated a full 1920x1080 binary array and ran connected components
    # on that full frame for every detection. Cropping is equivalent:
    # pixels outside the bbox were always zeroed before component analysis.
    cropped_mask = np.ascontiguousarray(
        np.asarray(mask)[iy1:iy2, ix1:ix2] > 0,
        dtype=np.uint8,
    )

    component = largest_connected_component(cropped_mask)
    local_ys, local_xs = np.where(component)
    ys = local_ys + iy1
    xs = local_xs + ix1

    if len(xs) == 0:
        return None

    ground_y = float(
        np.percentile(
            ys.astype(
                np.float64
            ),
            bottom_y_percentile,
        )
    )

    bbox_height = max(
        y2 - y1,
        1.0,
    )

    band_height = max(
        bbox_height
        * bottom_band_frac,
        1.0,
    )

    band_min_y = (
        ground_y
        - band_height
    )

    band_mask = (
        ys >= band_min_y
    )

    if not np.any(band_mask):
        return None

    ground_x = float(
        np.median(
            xs[band_mask]
        )
    )

    return (
        ground_x,
        ground_y,
    )


def summarize_numeric(
    values: list[float],
) -> dict[str, Any]:
    array = np.asarray(
        [
            value
            for value in values
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if len(array) == 0:
        return {
            "count": 0,
        }

    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "count": len(rows),
        "sam_error_m": summarize_numeric(
            [
                row["sam_error_m"]
                for row in rows
            ]
        ),
        "signed_dx_px": summarize_numeric(
            [
                row["signed_dx_px"]
                for row in rows
            ]
        ),
        "signed_dy_px": summarize_numeric(
            [
                row["signed_dy_px"]
                for row in rows
            ]
        ),
        "abs_dx_px": summarize_numeric(
            [
                abs(
                    row["signed_dx_px"]
                )
                for row in rows
            ]
        ),
        "abs_dy_px": summarize_numeric(
            [
                abs(
                    row["signed_dy_px"]
                )
                for row in rows
            ]
        ),
    }


def main() -> None:
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )
    yolo_cache_root = (
        args.yolo_cache_root
        .expanduser()
        .resolve()
    )
    sam_cache_root = (
        args.sam_cache_root
        .expanduser()
        .resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_path = (
        resolve_annotation_file(
            dataset_root,
            args.split,
        )
    )
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
        list[dict[str, Any]],
    ] = defaultdict(list)

    for annotation in dataset.get(
        "annotations",
        [],
    ):
        if (
            int(
                annotation.get(
                    "category_id",
                    1,
                )
            )
            != 1
        ):
            continue

        if "bbox" not in annotation:
            continue

        if (
            "position_on_pitch"
            not in annotation
        ):
            continue

        gt_by_image[
            int(
                annotation[
                    "image_id"
                ]
            )
        ].append(
            annotation
        )

    yolo_dir = (
        yolo_cache_root
        / args.split
        / "yolo_raw"
    )
    sam_dir = (
        sam_cache_root
        / args.split
        / "sam_raw"
    )

    if not yolo_dir.is_dir():
        raise FileNotFoundError(
            f"YOLO cache not found: {yolo_dir}"
        )

    if not sam_dir.is_dir():
        raise FileNotFoundError(
            f"SAM cache not found: {sam_dir}"
        )

    rows: list[dict[str, Any]] = []

    total_detections = 0
    total_gt = 0
    bbox_matches = 0
    sam_valid = 0
    sam_invalid = 0
    missing_gt_ground = 0

    yolo_files = sorted(yolo_dir.glob("*.npz"))
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        yolo_files = yolo_files[:args.max_images]

    print(
        f"Processing {len(yolo_files)} SAM cache files "
        f"with bbox-only connected-component analysis...",
        flush=True,
    )
    for yolo_path in tqdm(
        yolo_files,
        desc="SAM localization",
        unit="image",
        mininterval=1.0,
    ):
        yolo = load_npz(
            yolo_path
        )

        image_id = scalar_int(
            yolo["image_id"]
        )

        if (
            image_id
            not in images_by_id
        ):
            continue

        sam_path = (
            sam_dir
            / yolo_path.name
        )

        if not sam_path.is_file():
            raise FileNotFoundError(
                f"SAM cache missing: {sam_path}"
            )

        sam = load_npz(
            sam_path
        )

        yolo_boxes = np.asarray(
            yolo["bboxes"],
            dtype=np.float32,
        )
        yolo_scores = np.asarray(
            yolo["detection_scores"],
            dtype=np.float32,
        )

        sam_boxes = np.asarray(
            sam["bboxes"],
            dtype=np.float32,
        )
        sam_scores = np.asarray(
            sam["detection_scores"],
            dtype=np.float32,
        )

        if not np.array_equal(
            yolo_boxes,
            sam_boxes,
        ):
            raise RuntimeError(
                "YOLO/SAM bbox cache mismatch: "
                f"{yolo_path.name}"
            )

        if not np.array_equal(
            yolo_scores,
            sam_scores,
        ):
            raise RuntimeError(
                "YOLO/SAM score cache mismatch: "
                f"{yolo_path.name}"
            )

        masks = np.asarray(
            sam["masks"]
        )

        if "mask_valid" in sam:
            mask_valid = np.asarray(
                sam["mask_valid"]
            ).astype(bool)
        else:
            mask_valid = np.asarray(
                [
                    np.any(mask)
                    for mask in masks
                ],
                dtype=bool,
            )

        selected_indices = (
            np.flatnonzero(
                yolo_scores
                >= args.score_threshold
            )
        )

        boxes = yolo_boxes[
            selected_indices
        ]
        selected_valid = mask_valid[
            selected_indices
        ]

        total_detections += len(
            boxes
        )

        actual_width = scalar_int(
            yolo["actual_width"]
        )
        actual_height = scalar_int(
            yolo["actual_height"]
        )
        annotation_width = scalar_int(
            yolo["annotation_width"]
        )
        annotation_height = scalar_int(
            yolo["annotation_height"]
        )

        detection_boxes_ann = (
            np.asarray(
                [
                    scale_box_actual_to_annotation(
                        box,
                        actual_width,
                        actual_height,
                        annotation_width,
                        annotation_height,
                    )
                    for box in boxes
                ],
                dtype=np.float32,
            ).reshape(
                -1,
                4,
            )
        )

        gt_annotations = (
            gt_by_image.get(
                image_id,
                [],
            )
        )

        total_gt += len(
            gt_annotations
        )

        gt_boxes = np.asarray(
            [
                xywh_to_xyxy(
                    annotation[
                        "bbox"
                    ]
                )
                for annotation
                in gt_annotations
            ],
            dtype=np.float32,
        ).reshape(
            -1,
            4,
        )

        matches = (
            match_detections_to_gt(
                detection_boxes_ann,
                gt_boxes,
                args.match_iou,
            )
        )

        bbox_matches += len(
            matches
        )

        image_info = (
            images_by_id[
                image_id
            ]
        )

        for det_index, (
            gt_index,
            match_iou,
        ) in matches.items():
            gt = gt_annotations[
                gt_index
            ]

            gt_ground_ann = (
                get_gt_ground_point_annotation(
                    gt
                )
            )

            if gt_ground_ann is None:
                missing_gt_ground += 1
                continue

            gt_ground_actual = (
                scale_point_annotation_to_actual(
                    gt_ground_ann,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            if not selected_valid[
                det_index
            ]:
                sam_invalid += 1
                continue

            sam_point = (
                get_sam_ground_point(
                    masks[
                        selected_indices[det_index]
                    ],
                    boxes[
                        det_index
                    ],
                    args.bottom_band_frac,
                    args.bottom_y_percentile,
                )
            )

            if sam_point is None:
                sam_invalid += 1
                continue

            sam_valid += 1

            signed_dx_px = (
                float(
                    sam_point[0]
                )
                - float(
                    gt_ground_actual[0]
                )
            )
            signed_dy_px = (
                float(
                    sam_point[1]
                )
                - float(
                    gt_ground_actual[1]
                )
            )

            sam_point_ann = (
                scale_point_actual_to_annotation(
                    sam_point,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            sam_pitch = (
                project_to_pitch(
                    sam_point_ann,
                    image_info,
                )
            )

            gt_pitch = np.asarray(
                gt[
                    "position_on_pitch"
                ][:2],
                dtype=np.float32,
            )

            sam_error_m = float(
                np.linalg.norm(
                    sam_pitch
                    - gt_pitch
                )
            )

            rows.append({
                "image_id": image_id,
                "file_name": scalar_str(
                    yolo["file_name"]
                ),
                "detection_index": int(
                    selected_indices[
                        det_index
                    ]
                ),
                "gt_annotation_id": int(
                    gt.get(
                        "id",
                        -1,
                    )
                ),
                "bbox_match_iou": float(
                    match_iou
                ),
                "sam_x_px": float(
                    sam_point[0]
                ),
                "sam_y_px": float(
                    sam_point[1]
                ),
                "gt_x_px": float(
                    gt_ground_actual[0]
                ),
                "gt_y_px": float(
                    gt_ground_actual[1]
                ),
                "signed_dx_px": (
                    signed_dx_px
                ),
                "signed_dy_px": (
                    signed_dy_px
                ),
                "abs_dx_px": abs(
                    signed_dx_px
                ),
                "abs_dy_px": abs(
                    signed_dy_px
                ),
                "sam_error_m": (
                    sam_error_m
                ),
            })

    csv_path = (
        output_dir
        / "sam_localization_errors.csv"
    )

    if rows:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                rows
            )

    within_025 = [
        row
        for row in rows
        if row["sam_error_m"]
        <= 0.25
    ]
    between_025_050 = [
        row
        for row in rows
        if (
            0.25
            < row["sam_error_m"]
            <= 0.50
        )
    ]
    within_050 = [
        row
        for row in rows
        if row["sam_error_m"]
        <= 0.50
    ]
    over_050 = [
        row
        for row in rows
        if row["sam_error_m"]
        > 0.50
    ]

    summary = {
        "configuration": {
            "score_threshold": (
                args.score_threshold
            ),
            "bbox_match_iou": (
                args.match_iou
            ),
            "bottom_band_frac": (
                args.bottom_band_frac
            ),
            "bottom_y_percentile": (
                args.bottom_y_percentile
            ),
            "signed_error_definition": (
                "prediction - GT; +dx = right of GT; "
                "+dy = lower/below GT in image"
            ),
        },
        "detection_matching": {
            "gt_players": total_gt,
            "yolo_detections_above_threshold": (
                total_detections
            ),
            "bbox_iou_matched_pairs": (
                bbox_matches
            ),
            "sam_valid_matched_pairs": (
                sam_valid
            ),
            "sam_invalid_matched_pairs": (
                sam_invalid
            ),
            "matched_pairs_missing_gt_ground_keypoint": (
                missing_gt_ground
            ),
        },
        "all_sam_matched_players": (
            summarize_rows(
                rows
            )
        ),
        "sam_within_0_25m": (
            summarize_rows(
                within_025
            )
        ),
        "sam_0_25_to_0_50m": (
            summarize_rows(
                between_025_050
            )
        ),
        "sam_within_0_50m": (
            summarize_rows(
                within_050
            )
        ),
        "sam_over_0_50m": (
            summarize_rows(
                over_050
            )
        ),
    }

    summary_path = (
        output_dir
        / "sam_localization_error_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print(
        "\n=== SAM localization analysis ==="
    )
    print(
        f"GT players:                     {total_gt}"
    )
    print(
        "YOLO detections >= threshold:   "
        f"{total_detections}"
    )
    print(
        f"BBox-IoU matched pairs:         {bbox_matches}"
    )
    print(
        f"Valid SAM matched pairs:        {sam_valid}"
    )
    print(
        f"Invalid SAM matched pairs:      {sam_invalid}"
    )
    print(
        f"SAM <= 0.25 m:                 {len(within_025)}"
    )
    print(
        "SAM 0.25-0.50 m:              "
        f"{len(between_025_050)}"
    )
    print(
        f"SAM <= 0.50 m:                 {len(within_050)}"
    )

    if rows:
        all_summary = (
            summary[
                "all_sam_matched_players"
            ]
        )

        print(
            "\n=== Signed image-space error ==="
        )
        print(
            "Mean signed dx:   "
            f"{all_summary['signed_dx_px']['mean']:.3f} px"
        )
        print(
            "Median signed dx: "
            f"{all_summary['signed_dx_px']['median']:.3f} px"
        )
        print(
            "Mean signed dy:   "
            f"{all_summary['signed_dy_px']['mean']:.3f} px"
        )
        print(
            "Median signed dy: "
            f"{all_summary['signed_dy_px']['median']:.3f} px"
        )
        print(
            "Mean abs dx:      "
            f"{all_summary['abs_dx_px']['mean']:.3f} px"
        )
        print(
            "Mean abs dy:      "
            f"{all_summary['abs_dy_px']['mean']:.3f} px"
        )

    if between_025_050:
        borderline = (
            summary[
                "sam_0_25_to_0_50m"
            ]
        )

        print(
            "\n=== Borderline 25-50 cm SAM cases ==="
        )
        print(
            "Mean signed dx:   "
            f"{borderline['signed_dx_px']['mean']:.3f} px"
        )
        print(
            "Median signed dx: "
            f"{borderline['signed_dx_px']['median']:.3f} px"
        )
        print(
            "Mean signed dy:   "
            f"{borderline['signed_dy_px']['mean']:.3f} px"
        )
        print(
            "Median signed dy: "
            f"{borderline['signed_dy_px']['median']:.3f} px"
        )
        print(
            "Mean abs dx:      "
            f"{borderline['abs_dx_px']['mean']:.3f} px"
        )
        print(
            "Mean abs dy:      "
            f"{borderline['abs_dy_px']['mean']:.3f} px"
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
