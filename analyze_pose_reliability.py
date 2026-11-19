from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from sskit import image_to_ground


SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "coco17": {
        "left_hip": 11,
        "right_hip": 12,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [],
        "right_foot": [],
    },
    "halpe26": {
        "left_hip": 11,
        "right_hip": 12,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [20, 22, 24],
        "right_foot": [21, 23, 25],
    },
    "coco25": {
        "left_hip": 11,
        "right_hip": 12,
        "left_ankle": 17,
        "right_ankle": 18,
        "left_foot": [19, 20, 21],
        "right_foot": [22, 23, 24],
    },
    "coco133": {
        "left_hip": 11,
        "right_hip": 12,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [17, 18, 19],
        "right_foot": [20, 21, 22],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-detection dataset for deciding when pose-derived "
            "pelvis localization is reliable on SoccerNet SynLoc."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--yolo-cache-root", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--pose-model", type=str, default="halpe26_m")
    parser.add_argument(
        "--pose-schema",
        choices=sorted(SCHEMA_DEFINITIONS),
        default="halpe26",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_pose_reliability"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.4302865565)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--keypoint-conf", type=float, default=0.30)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


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
        f"Could not find annotation file for split={split}. Tried: {candidates[split]}"
    )


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, Any] = {
            "image_id": int(data["image_id"]),
            "file_name": str(data["file_name"].item()),
            "actual_width": int(data["actual_width"]),
            "actual_height": int(data["actual_height"]),
            "annotation_width": int(data["annotation_width"]),
            "annotation_height": int(data["annotation_height"]),
            "bboxes": np.asarray(data["bboxes"], dtype=np.float32),
            "detection_scores": np.asarray(
                data["detection_scores"], dtype=np.float32
            ),
        }
        if "keypoints" in data:
            result["keypoints"] = np.asarray(data["keypoints"], dtype=np.float32)
        if "keypoint_scores" in data:
            result["keypoint_scores"] = np.asarray(
                data["keypoint_scores"], dtype=np.float32
            )
        return result


def xywh_to_xyxy(bbox: list[float] | np.ndarray) -> np.ndarray:
    x, y, width, height = map(float, bbox)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


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
    return np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


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


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

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
        (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]),
    )
    area_b = np.maximum(
        0.0,
        (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]),
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

    ious = pairwise_iou(detection_boxes, gt_boxes)
    detection_indices, gt_indices = linear_sum_assignment(1.0 - ious)

    matches: dict[int, tuple[int, float]] = {}

    for detection_index, gt_index in zip(detection_indices, gt_indices):
        iou = float(ious[detection_index, gt_index])
        if iou >= min_iou:
            matches[int(detection_index)] = (int(gt_index), iou)

    return matches


def bbox_bottom_center(bbox: np.ndarray) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, bbox)
    return (x1 + x2) / 2.0, y2


def confident_point(
    keypoints: np.ndarray,
    scores: np.ndarray,
    index: int,
    threshold: float,
) -> tuple[np.ndarray | None, float]:
    if index >= len(keypoints) or index >= len(scores):
        return None, float("nan")

    score = float(scores[index])
    if score < threshold:
        return None, score

    return np.asarray(keypoints[index], dtype=np.float32), score


def pelvis_center(
    keypoints: np.ndarray,
    scores: np.ndarray,
    schema: str,
    threshold: float,
) -> dict[str, Any]:
    definition = SCHEMA_DEFINITIONS[schema]
    left_index = int(definition["left_hip"])
    right_index = int(definition["right_hip"])

    left_point, left_score = confident_point(
        keypoints, scores, left_index, threshold
    )
    right_point, right_score = confident_point(
        keypoints, scores, right_index, threshold
    )

    if left_point is not None and right_point is not None:
        center = (left_point + right_point) / 2.0
        source = "both_hips"
    elif left_point is not None:
        center = left_point
        source = "left_hip_only"
    elif right_point is not None:
        center = right_point
        source = "right_hip_only"
    else:
        center = None
        source = "unavailable"

    finite_scores = [
        value
        for value in [left_score, right_score]
        if np.isfinite(value)
    ]

    return {
        "center": center,
        "source": source,
        "left_score": left_score,
        "right_score": right_score,
        "min_score": float(np.min(finite_scores)) if finite_scores else float("nan"),
        "mean_score": float(np.mean(finite_scores)) if finite_scores else float("nan"),
    }


def mean_confident_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
    threshold: float,
) -> tuple[np.ndarray | None, list[float]]:
    points: list[np.ndarray] = []
    valid_scores: list[float] = []

    for index in indices:
        point, score = confident_point(
            keypoints, scores, int(index), threshold
        )
        if point is not None:
            points.append(point)
            valid_scores.append(float(score))

    if not points:
        return None, valid_scores

    return np.mean(np.stack(points, axis=0), axis=0), valid_scores


def feet_center(
    keypoints: np.ndarray,
    scores: np.ndarray,
    schema: str,
    threshold: float,
) -> dict[str, Any]:
    """
    Center of visible foot keypoints.

    For Halpe26:
      left foot  = mean of confident [20,22,24]
      right foot = mean of confident [21,23,25]
      feet center = midpoint of left/right foot centers when both exist.
    """
    definition = SCHEMA_DEFINITIONS[schema]

    left_center, left_scores = mean_confident_keypoints(
        keypoints,
        scores,
        list(definition["left_foot"]),
        threshold,
    )
    right_center, right_scores = mean_confident_keypoints(
        keypoints,
        scores,
        list(definition["right_foot"]),
        threshold,
    )

    if left_center is not None and right_center is not None:
        center = (left_center + right_center) / 2.0
        source = "both_feet"
    elif left_center is not None:
        center = left_center
        source = "left_foot_only"
    elif right_center is not None:
        center = right_center
        source = "right_foot_only"
    else:
        center = None
        source = "unavailable"

    all_scores = left_scores + right_scores

    return {
        "center": center,
        "source": source,
        "left_center": left_center,
        "right_center": right_center,
        "left_valid_keypoints": len(left_scores),
        "right_valid_keypoints": len(right_scores),
        "valid_keypoints": len(all_scores),
        "min_score": float(np.min(all_scores)) if all_scores else float("nan"),
        "mean_score": float(np.mean(all_scores)) if all_scores else float("nan"),
        "max_score": float(np.max(all_scores)) if all_scores else float("nan"),
    }


def get_gt_ground_point(
    annotation: dict[str, Any],
) -> tuple[float, float] | None:
    raw = annotation.get("keypoints")
    if raw is None:
        return None

    try:
        keypoints = np.asarray(raw, dtype=np.float64)
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

    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    return x, y


def project_to_pitch(
    point_annotation: tuple[float, float],
    image_info: dict[str, Any],
) -> np.ndarray:
    x, y = point_annotation
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
        pitch = pitch[:, :2].detach().cpu().numpy()
    else:
        pitch = np.asarray(pitch, dtype=np.float32)[:, :2]

    return np.asarray(pitch[0], dtype=np.float32)


def pitch_error(
    point_actual: tuple[float, float] | None,
    *,
    actual_width: int,
    actual_height: int,
    annotation_width: int,
    annotation_height: int,
    image_info: dict[str, Any],
    gt_pitch: np.ndarray | None,
) -> tuple[float, float, float]:
    if point_actual is None or gt_pitch is None:
        return float("nan"), float("nan"), float("nan")

    point_annotation = scale_point_actual_to_annotation(
        point_actual,
        actual_width,
        actual_height,
        annotation_width,
        annotation_height,
    )

    predicted_pitch = project_to_pitch(point_annotation, image_info)
    delta = predicted_pitch - gt_pitch

    return (
        float(np.linalg.norm(delta)),
        float(delta[0]),
        float(delta[1]),
    )


def finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def point_value(point: np.ndarray | None, axis: int) -> float | str:
    if point is None:
        return ""
    value = float(point[axis])
    return value if np.isfinite(value) else ""


def ratio_or_blank(numerator: float, denominator: float) -> float | str:
    if not np.isfinite(numerator) or denominator <= 0:
        return ""
    return float(numerator / denominator)


def summarize_numeric(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    values: list[float] = []

    for row in rows:
        value = row.get(key)
        if value == "" or value is None:
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(numeric):
            values.append(numeric)

    if not values:
        return {"count": 0}

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    yolo_cache_root = args.yolo_cache_root.expanduser().resolve()
    pose_cache_root = args.pose_cache_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        / args.split
        / args.pose_model
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = resolve_annotation_file(dataset_root, args.split)
    dataset = load_json(annotation_path)

    images_by_id = {
        int(image["id"]): image
        for image in dataset.get("images", [])
    }

    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in dataset.get("annotations", []):
        if int(annotation.get("category_id", 1)) != 1:
            continue
        if "bbox" not in annotation:
            continue
        gt_by_image[int(annotation["image_id"])].append(annotation)

    yolo_cache_dir = yolo_cache_root / args.split / "yolo_raw"
    pose_cache_dir = (
        pose_cache_root
        / args.split
        / args.pose_model
        / "pose_raw"
    )

    if not yolo_cache_dir.is_dir():
        raise FileNotFoundError(f"YOLO cache not found: {yolo_cache_dir}")

    if not pose_cache_dir.is_dir():
        raise FileNotFoundError(f"Pose cache not found: {pose_cache_dir}")

    yolo_files = sorted(yolo_cache_dir.glob("*.npz"))

    rows: list[dict[str, Any]] = []

    total_detections = 0
    matched_detections = 0
    unmatched_detections = 0
    pelvis_available_count = 0
    feet_available_count = 0

    for yolo_path in yolo_files:
        yolo = load_cache(yolo_path)
        image_id = int(yolo["image_id"])

        if image_id not in images_by_id:
            continue

        pose_path = pose_cache_dir / yolo_path.name

        if not pose_path.is_file():
            raise FileNotFoundError(f"Pose cache missing: {pose_path}")

        pose = load_cache(pose_path)

        if not np.array_equal(yolo["bboxes"], pose["bboxes"]):
            raise RuntimeError(
                f"YOLO/Pose bbox cache mismatch: {yolo_path.name}"
            )

        if not np.array_equal(
            yolo["detection_scores"],
            pose["detection_scores"],
        ):
            raise RuntimeError(
                f"YOLO/Pose score cache mismatch: {yolo_path.name}"
            )

        scores = yolo["detection_scores"]
        selected_indices = np.flatnonzero(
            scores >= args.score_threshold
        )

        boxes = yolo["bboxes"][selected_indices]
        selected_scores = scores[selected_indices]
        keypoints = pose["keypoints"][selected_indices]
        keypoint_scores = pose["keypoint_scores"][selected_indices]

        total_detections += len(boxes)

        actual_width = int(yolo["actual_width"])
        actual_height = int(yolo["actual_height"])
        annotation_width = int(yolo["annotation_width"])
        annotation_height = int(yolo["annotation_height"])

        detection_boxes_annotation = np.asarray(
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
        ).reshape(-1, 4)

        gt_annotations = gt_by_image.get(image_id, [])
        gt_boxes = np.asarray(
            [
                xywh_to_xyxy(annotation["bbox"])
                for annotation in gt_annotations
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

        matches = match_detections_to_gt(
            detection_boxes_annotation,
            gt_boxes,
            args.match_iou,
        )

        image_info = images_by_id[image_id]

        for local_index, (
            box,
            score,
            person_keypoints,
            person_scores,
        ) in enumerate(
            zip(
                boxes,
                selected_scores,
                keypoints,
                keypoint_scores,
            )
        ):
            original_index = int(selected_indices[local_index])

            x1, y1, x2, y2 = map(float, box)
            bbox_width = max(x2 - x1, 1e-6)
            bbox_height = max(y2 - y1, 1e-6)
            bbox_center_x = (x1 + x2) / 2.0

            bbox_point = bbox_bottom_center(box)

            pelvis = pelvis_center(
                person_keypoints,
                person_scores,
                args.pose_schema,
                args.keypoint_conf,
            )
            feet = feet_center(
                person_keypoints,
                person_scores,
                args.pose_schema,
                args.keypoint_conf,
            )

            pelvis_point = pelvis["center"]
            feet_point = feet["center"]

            if pelvis_point is not None:
                pelvis_available_count += 1
            if feet_point is not None:
                feet_available_count += 1

            if pelvis_point is not None:
                pelvis_ground_point = (
                    float(np.clip(pelvis_point[0], x1, x2)),
                    float(y2),
                )
            else:
                pelvis_ground_point = None

            if feet_point is not None:
                feet_ground_point = (
                    float(feet_point[0]),
                    float(feet_point[1]),
                )
            else:
                feet_ground_point = None

            matched = local_index in matches

            gt_annotation_id: int | str = ""
            bbox_match_iou: float | str = ""
            gt_ground_x_px: float | str = ""
            gt_ground_y_px: float | str = ""
            gt_pitch_x_m: float | str = ""
            gt_pitch_y_m: float | str = ""

            bbox_error_m = float("nan")
            bbox_pitch_dx_m = float("nan")
            bbox_pitch_dy_m = float("nan")
            pelvis_error_m = float("nan")
            pelvis_pitch_dx_m = float("nan")
            pelvis_pitch_dy_m = float("nan")
            feet_error_m = float("nan")
            feet_pitch_dx_m = float("nan")
            feet_pitch_dy_m = float("nan")

            pelvis_dx_gt_px = float("nan")
            pelvis_dy_gt_px = float("nan")
            feet_dx_gt_px = float("nan")
            feet_dy_gt_px = float("nan")

            if matched:
                matched_detections += 1

                gt_index, match_iou = matches[local_index]
                gt = gt_annotations[gt_index]

                gt_annotation_id = int(gt.get("id", -1))
                bbox_match_iou = float(match_iou)

                gt_pitch: np.ndarray | None = None
                if (
                    "position_on_pitch" in gt
                    and len(gt["position_on_pitch"]) >= 2
                ):
                    gt_pitch = np.asarray(
                        gt["position_on_pitch"][:2],
                        dtype=np.float32,
                    )
                    gt_pitch_x_m = float(gt_pitch[0])
                    gt_pitch_y_m = float(gt_pitch[1])

                gt_ground_ann = get_gt_ground_point(gt)
                gt_ground_actual: tuple[float, float] | None = None

                if gt_ground_ann is not None:
                    gt_ground_actual = scale_point_annotation_to_actual(
                        gt_ground_ann,
                        actual_width,
                        actual_height,
                        annotation_width,
                        annotation_height,
                    )
                    gt_ground_x_px = float(gt_ground_actual[0])
                    gt_ground_y_px = float(gt_ground_actual[1])

                (
                    bbox_error_m,
                    bbox_pitch_dx_m,
                    bbox_pitch_dy_m,
                ) = pitch_error(
                    bbox_point,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    annotation_width=annotation_width,
                    annotation_height=annotation_height,
                    image_info=image_info,
                    gt_pitch=gt_pitch,
                )

                (
                    pelvis_error_m,
                    pelvis_pitch_dx_m,
                    pelvis_pitch_dy_m,
                ) = pitch_error(
                    pelvis_ground_point,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    annotation_width=annotation_width,
                    annotation_height=annotation_height,
                    image_info=image_info,
                    gt_pitch=gt_pitch,
                )

                (
                    feet_error_m,
                    feet_pitch_dx_m,
                    feet_pitch_dy_m,
                ) = pitch_error(
                    feet_ground_point,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    annotation_width=annotation_width,
                    annotation_height=annotation_height,
                    image_info=image_info,
                    gt_pitch=gt_pitch,
                )

                if gt_ground_actual is not None and pelvis_point is not None:
                    pelvis_dx_gt_px = (
                        float(pelvis_point[0]) - float(gt_ground_actual[0])
                    )
                    pelvis_dy_gt_px = (
                        float(pelvis_point[1]) - float(gt_ground_actual[1])
                    )

                if gt_ground_actual is not None and feet_point is not None:
                    feet_dx_gt_px = (
                        float(feet_point[0]) - float(gt_ground_actual[0])
                    )
                    feet_dy_gt_px = (
                        float(feet_point[1]) - float(gt_ground_actual[1])
                    )

            else:
                unmatched_detections += 1

            pelvis_x = (
                float(pelvis_point[0])
                if pelvis_point is not None
                else float("nan")
            )
            pelvis_y = (
                float(pelvis_point[1])
                if pelvis_point is not None
                else float("nan")
            )
            feet_x = (
                float(feet_point[0])
                if feet_point is not None
                else float("nan")
            )
            feet_y = (
                float(feet_point[1])
                if feet_point is not None
                else float("nan")
            )

            pelvis_feet_dx = (
                pelvis_x - feet_x
                if np.isfinite(pelvis_x) and np.isfinite(feet_x)
                else float("nan")
            )
            pelvis_feet_dy = (
                pelvis_y - feet_y
                if np.isfinite(pelvis_y) and np.isfinite(feet_y)
                else float("nan")
            )

            pelvis_shift_from_bbox_center = (
                pelvis_x - bbox_center_x
                if np.isfinite(pelvis_x)
                else float("nan")
            )

            pose_better_than_bbox: bool | str = ""
            error_improvement_m: float | str = ""

            if np.isfinite(pelvis_error_m) and np.isfinite(bbox_error_m):
                pose_better_than_bbox = bool(
                    pelvis_error_m < bbox_error_m
                )
                error_improvement_m = float(
                    bbox_error_m - pelvis_error_m
                )

            rows.append({
                "image_id": image_id,
                "file_name": yolo["file_name"],
                "detection_index": original_index,
                "yolo_score": float(score),

                "matched_gt": bool(matched),
                "gt_annotation_id": gt_annotation_id,
                "bbox_match_iou": bbox_match_iou,

                "bbox_x1_px": x1,
                "bbox_y1_px": y1,
                "bbox_x2_px": x2,
                "bbox_y2_px": y2,
                "bbox_width_px": bbox_width,
                "bbox_height_px": bbox_height,
                "bbox_center_x_px": bbox_center_x,

                "gt_ground_x_px": gt_ground_x_px,
                "gt_ground_y_px": gt_ground_y_px,
                "gt_pitch_x_m": gt_pitch_x_m,
                "gt_pitch_y_m": gt_pitch_y_m,

                "bbox_ground_x_px": float(bbox_point[0]),
                "bbox_ground_y_px": float(bbox_point[1]),
                "bbox_error_m": finite_or_blank(bbox_error_m),
                "bbox_pitch_dx_m": finite_or_blank(bbox_pitch_dx_m),
                "bbox_pitch_dy_m": finite_or_blank(bbox_pitch_dy_m),

                "pelvis_available": pelvis_point is not None,
                "pelvis_source": pelvis["source"],
                "pelvis_center_x_px": point_value(pelvis_point, 0),
                "pelvis_center_y_px": point_value(pelvis_point, 1),
                "left_hip_score": finite_or_blank(pelvis["left_score"]),
                "right_hip_score": finite_or_blank(pelvis["right_score"]),
                "hip_min_score": finite_or_blank(pelvis["min_score"]),
                "hip_mean_score": finite_or_blank(pelvis["mean_score"]),

                "pelvis_x_minus_bbox_center_px": finite_or_blank(
                    pelvis_shift_from_bbox_center
                ),
                "pelvis_x_minus_bbox_center_bbox_width": ratio_or_blank(
                    pelvis_shift_from_bbox_center,
                    bbox_width,
                ),
                "pelvis_x_relative_to_bbox": (
                    ratio_or_blank(pelvis_x - x1, bbox_width)
                    if np.isfinite(pelvis_x)
                    else ""
                ),
                "pelvis_y_relative_to_bbox": (
                    ratio_or_blank(pelvis_y - y1, bbox_height)
                    if np.isfinite(pelvis_y)
                    else ""
                ),

                "pelvis_ground_x_px": (
                    pelvis_ground_point[0]
                    if pelvis_ground_point is not None
                    else ""
                ),
                "pelvis_ground_y_px": (
                    pelvis_ground_point[1]
                    if pelvis_ground_point is not None
                    else ""
                ),
                "pelvis_ground_error_m": finite_or_blank(pelvis_error_m),
                "pelvis_ground_pitch_dx_m": finite_or_blank(
                    pelvis_pitch_dx_m
                ),
                "pelvis_ground_pitch_dy_m": finite_or_blank(
                    pelvis_pitch_dy_m
                ),
                "pelvis_center_dx_from_gt_ground_px": finite_or_blank(
                    pelvis_dx_gt_px
                ),
                "pelvis_center_dy_from_gt_ground_px": finite_or_blank(
                    pelvis_dy_gt_px
                ),

                "feet_available": feet_point is not None,
                "feet_source": feet["source"],
                "feet_center_x_px": point_value(feet_point, 0),
                "feet_center_y_px": point_value(feet_point, 1),
                "left_foot_center_x_px": point_value(feet["left_center"], 0),
                "left_foot_center_y_px": point_value(feet["left_center"], 1),
                "right_foot_center_x_px": point_value(feet["right_center"], 0),
                "right_foot_center_y_px": point_value(feet["right_center"], 1),
                "left_foot_valid_keypoints": feet["left_valid_keypoints"],
                "right_foot_valid_keypoints": feet["right_valid_keypoints"],
                "foot_valid_keypoints": feet["valid_keypoints"],
                "foot_min_score": finite_or_blank(feet["min_score"]),
                "foot_mean_score": finite_or_blank(feet["mean_score"]),
                "foot_max_score": finite_or_blank(feet["max_score"]),

                "feet_x_relative_to_bbox": (
                    ratio_or_blank(feet_x - x1, bbox_width)
                    if np.isfinite(feet_x)
                    else ""
                ),
                "feet_y_relative_to_bbox": (
                    ratio_or_blank(feet_y - y1, bbox_height)
                    if np.isfinite(feet_y)
                    else ""
                ),
                "feet_center_error_m": finite_or_blank(feet_error_m),
                "feet_pitch_dx_m": finite_or_blank(feet_pitch_dx_m),
                "feet_pitch_dy_m": finite_or_blank(feet_pitch_dy_m),
                "feet_center_dx_from_gt_ground_px": finite_or_blank(
                    feet_dx_gt_px
                ),
                "feet_center_dy_from_gt_ground_px": finite_or_blank(
                    feet_dy_gt_px
                ),

                "pelvis_minus_feet_x_px": finite_or_blank(pelvis_feet_dx),
                "pelvis_minus_feet_y_px": finite_or_blank(pelvis_feet_dy),
                "abs_pelvis_minus_feet_x_px": finite_or_blank(
                    abs(pelvis_feet_dx)
                    if np.isfinite(pelvis_feet_dx)
                    else float("nan")
                ),
                "pelvis_minus_feet_x_bbox_width": ratio_or_blank(
                    pelvis_feet_dx,
                    bbox_width,
                ),
                "pelvis_minus_feet_y_bbox_height": ratio_or_blank(
                    pelvis_feet_dy,
                    bbox_height,
                ),

                "pelvis_better_than_bbox": pose_better_than_bbox,
                "pelvis_improvement_over_bbox_m": error_improvement_m,
            })

    csv_path = output_dir / "pose_reliability_detections.csv"

    if rows:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    matched_rows = [
        row for row in rows
        if row["matched_gt"]
    ]
    usable_pelvis_rows = [
        row for row in matched_rows
        if row["pelvis_ground_error_m"] != ""
    ]
    pose_better_rows = [
        row for row in usable_pelvis_rows
        if row["pelvis_better_than_bbox"] is True
    ]
    pose_worse_rows = [
        row for row in usable_pelvis_rows
        if row["pelvis_better_than_bbox"] is False
    ]

    summary = {
        "configuration": {
            "split": args.split,
            "score_threshold": args.score_threshold,
            "bbox_match_iou": args.match_iou,
            "pose_model": args.pose_model,
            "pose_schema": args.pose_schema,
            "keypoint_confidence": args.keypoint_conf,
            "pelvis_ground_method": "pelvis center x + bbox bottom y",
            "feet_center_definition": (
                "Mean of confident keypoints within each foot; midpoint of "
                "left/right foot centers when both are available."
            ),
        },
        "counts": {
            "detections_above_score_threshold": total_detections,
            "matched_to_gt_by_bbox_iou": matched_detections,
            "unmatched_detections": unmatched_detections,
            "pelvis_available": pelvis_available_count,
            "feet_available": feet_available_count,
            "matched_with_usable_pelvis": len(usable_pelvis_rows),
            "pelvis_better_than_bbox": len(pose_better_rows),
            "pelvis_not_better_than_bbox": len(pose_worse_rows),
            "pelvis_better_rate": (
                len(pose_better_rows) / len(usable_pelvis_rows)
                if usable_pelvis_rows
                else None
            ),
        },
        "all_matched_with_pelvis": {
            "pelvis_ground_error_m": summarize_numeric(
                usable_pelvis_rows,
                "pelvis_ground_error_m",
            ),
            "bbox_error_m": summarize_numeric(
                usable_pelvis_rows,
                "bbox_error_m",
            ),
            "hip_min_score": summarize_numeric(
                usable_pelvis_rows,
                "hip_min_score",
            ),
            "hip_mean_score": summarize_numeric(
                usable_pelvis_rows,
                "hip_mean_score",
            ),
            "pelvis_x_minus_bbox_center_px": summarize_numeric(
                usable_pelvis_rows,
                "pelvis_x_minus_bbox_center_px",
            ),
            "abs_pelvis_minus_feet_x_px": summarize_numeric(
                usable_pelvis_rows,
                "abs_pelvis_minus_feet_x_px",
            ),
        },
        "pelvis_better_than_bbox_group": {
            "count": len(pose_better_rows),
            "hip_min_score": summarize_numeric(
                pose_better_rows,
                "hip_min_score",
            ),
            "hip_mean_score": summarize_numeric(
                pose_better_rows,
                "hip_mean_score",
            ),
            "pelvis_x_minus_bbox_center_px": summarize_numeric(
                pose_better_rows,
                "pelvis_x_minus_bbox_center_px",
            ),
            "abs_pelvis_minus_feet_x_px": summarize_numeric(
                pose_better_rows,
                "abs_pelvis_minus_feet_x_px",
            ),
            "pelvis_ground_error_m": summarize_numeric(
                pose_better_rows,
                "pelvis_ground_error_m",
            ),
            "pelvis_improvement_over_bbox_m": summarize_numeric(
                pose_better_rows,
                "pelvis_improvement_over_bbox_m",
            ),
        },
        "pelvis_not_better_than_bbox_group": {
            "count": len(pose_worse_rows),
            "hip_min_score": summarize_numeric(
                pose_worse_rows,
                "hip_min_score",
            ),
            "hip_mean_score": summarize_numeric(
                pose_worse_rows,
                "hip_mean_score",
            ),
            "pelvis_x_minus_bbox_center_px": summarize_numeric(
                pose_worse_rows,
                "pelvis_x_minus_bbox_center_px",
            ),
            "abs_pelvis_minus_feet_x_px": summarize_numeric(
                pose_worse_rows,
                "abs_pelvis_minus_feet_x_px",
            ),
            "pelvis_ground_error_m": summarize_numeric(
                pose_worse_rows,
                "pelvis_ground_error_m",
            ),
            "pelvis_improvement_over_bbox_m": summarize_numeric(
                pose_worse_rows,
                "pelvis_improvement_over_bbox_m",
            ),
        },
    }

    summary_path = output_dir / "pose_reliability_summary.json"
    save_json(summary_path, summary)

    print("\n=== Pose reliability dataset ===")
    print(f"Detections >= threshold: {total_detections}")
    print(f"BBox-IoU matched to GT:   {matched_detections}")
    print(f"Unmatched detections:     {unmatched_detections}")
    print(f"Pelvis available:         {pelvis_available_count}")
    print(f"Feet center available:    {feet_available_count}")

    if usable_pelvis_rows:
        rate = len(pose_better_rows) / len(usable_pelvis_rows)
        print(
            "Pelvis better than bbox: "
            f"{len(pose_better_rows)} / "
            f"{len(usable_pelvis_rows)} "
            f"({rate:.4f})"
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
