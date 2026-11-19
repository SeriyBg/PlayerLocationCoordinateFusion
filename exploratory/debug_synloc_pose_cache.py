from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sskit import image_to_ground
from tqdm import tqdm


# OpenCV uses BGR.
GT_COLOR = (0, 255, 0)       # green
BBOX_COLOR = (255, 0, 0)     # blue
METHOD_COLOR = (0, 0, 255)   # red
BOX_COLOR = (220, 220, 220)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


POSE_MODEL_SCHEMAS = {
    "coco17_m": "coco17",
    "halpe26_m": "halpe26",
    "halpe26_x_384": "halpe26",
    "vitpose25_l": "coco25",
    "dwpose133_l_384": "coco133",
    "rtmw133_l_384": "coco133",
}

SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "coco17": {
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [],
        "right_foot": [],
    },
    "halpe26": {
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [20, 22, 24],
        "right_foot": [21, 23, 25],
    },
    "coco25": {
        "left_ankle": 17,
        "right_ankle": 18,
        "left_foot": [19, 20, 21],
        "right_foot": [22, 23, 24],
    },
    "coco133": {
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [17, 18, 19],
        "right_foot": [20, 21, 22],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Debug cached SynLoc YOLO+pose predictions. For every detection, "
            "match it to a GT player by bbox IoU, compare bbox and pose "
            "ground-point errors in pitch-space metres, draw GT/bbox/pose "
            "points, and save a per-detection CSV."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help=(
            "Root passed to cache_synloc_pose.py, e.g. "
            "output/synloc_pose_cache"
        ),
    )
    p.add_argument(
        "--pose-model",
        choices=sorted(POSE_MODEL_SCHEMAS),
        default="halpe26_m",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../output/synloc_pose_debug"),
    )
    p.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.30,
        help="Same lower-body keypoint threshold used by the evaluator.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Ignore cached YOLO detections below this confidence.",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=0.30,
        help="Minimum bbox IoU for assigning a detection to GT.",
    )
    p.add_argument(
        "--better-margin-m",
        type=float,
        default=0.0,
        help=(
            "Pose is called BETTER only if its error is smaller than bbox "
            "error by more than this many metres."
        ),
    )
    p.add_argument(
        "--correct-threshold-m",
        type=float,
        default=0.25,
        help="Threshold used for bbox/method correctness transition columns.",
    )
    p.add_argument("--point-radius", type=int, default=2)
    p.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite already-rendered debug images.",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_directory(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }
    tried: list[Path] = []
    for name in aliases[split]:
        for candidate in (root / "images" / name, root / name):
            tried.append(candidate)
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Could not find image directory. Tried: {tried}")


def resolve_annotation_file(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }
    tried: list[Path] = []
    for name in aliases[split]:
        candidate = root / "annotations" / f"{name}.json"
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find annotation file. Tried: {tried}")


def extract_gt_ground_point(annotation: dict[str, Any]) -> tuple[float, float]:
    """
    SynLoc GT keypoint 1 is the image-space ground-projected point.
    Supports both nested and flat keypoint arrays.
    """
    keypoints = np.asarray(annotation["keypoints"], dtype=np.float32)

    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(-1, 3)

    if keypoints.shape[0] < 2:
        raise ValueError(
            f"Annotation {annotation.get('id')} has fewer than 2 keypoints."
        )

    return float(keypoints[1, 0]), float(keypoints[1, 1])


def annotation_bbox_to_actual(
    bbox_xywh: list[float],
    ann_width: int,
    ann_height: int,
    actual_width: int,
    actual_height: int,
) -> np.ndarray:
    x, y, w, h = map(float, bbox_xywh)
    sx = actual_width / float(ann_width)
    sy = actual_height / float(ann_height)

    return np.asarray(
        [x * sx, y * sy, (x + w) * sx, (y + h) * sy],
        dtype=np.float32,
    )


def annotation_point_to_actual(
    point: tuple[float, float],
    ann_width: int,
    ann_height: int,
    actual_width: int,
    actual_height: int,
) -> tuple[float, float]:
    sx = actual_width / float(ann_width)
    sy = actual_height / float(ann_height)
    return point[0] * sx, point[1] * sy


def actual_point_to_annotation(
    point: tuple[float, float],
    actual_width: int,
    actual_height: int,
    ann_width: int,
    ann_height: int,
) -> tuple[float, float]:
    sx = ann_width / float(actual_width)
    sy = ann_height / float(actual_height)
    return point[0] * sx, point[1] * sy


def bbox_ground_point(box: np.ndarray) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, y2


def best_lower_point_detail(
    keypoints: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
    threshold: float,
) -> tuple[np.ndarray, int, float] | None:
    candidates: list[tuple[np.ndarray, int, float]] = []

    for index in indices:
        if index >= len(scores) or index >= len(keypoints):
            continue

        score = float(scores[index])
        if score >= threshold:
            candidates.append((keypoints[index], index, score))

    if not candidates:
        return None

    return max(candidates, key=lambda item: float(item[0][1]))


def pose_ground_point_detail(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[tuple[float, float], str, dict[str, Any]]:
    """
    Same lower-body selection rule used by the pose evaluator, but returns
    diagnostic metadata about which keypoints were used.
    """
    definition = SCHEMA_DEFINITIONS[schema]

    left_foot = best_lower_point_detail(
        keypoints,
        scores,
        list(definition["left_foot"]),
        threshold,
    )
    right_foot = best_lower_point_detail(
        keypoints,
        scores,
        list(definition["right_foot"]),
        threshold,
    )

    left_ankle_index = int(definition["left_ankle"])
    right_ankle_index = int(definition["right_ankle"])

    left_ankle_score = (
        float(scores[left_ankle_index])
        if left_ankle_index < len(scores)
        else float("nan")
    )
    right_ankle_score = (
        float(scores[right_ankle_index])
        if right_ankle_index < len(scores)
        else float("nan")
    )

    diagnostics = {
        "left_ankle_score": left_ankle_score,
        "right_ankle_score": right_ankle_score,
        "left_foot_selected_index": (
            left_foot[1] if left_foot is not None else None
        ),
        "left_foot_selected_score": (
            left_foot[2] if left_foot is not None else None
        ),
        "right_foot_selected_index": (
            right_foot[1] if right_foot is not None else None
        ),
        "right_foot_selected_score": (
            right_foot[2] if right_foot is not None else None
        ),
        "keypoints_above_threshold": int(
            np.sum(np.asarray(scores) >= threshold)
        ),
        "mean_keypoint_score": (
            float(np.mean(scores)) if len(scores) else None
        ),
        "max_keypoint_score": (
            float(np.max(scores)) if len(scores) else None
        ),
    }

    if left_foot is not None and right_foot is not None:
        point = (left_foot[0] + right_foot[0]) / 2.0
        return (
            (float(point[0]), float(point[1])),
            "both_feet",
            diagnostics,
        )

    if left_foot is not None:
        return (
            (float(left_foot[0][0]), float(left_foot[0][1])),
            "left_foot",
            diagnostics,
        )

    if right_foot is not None:
        return (
            (float(right_foot[0][0]), float(right_foot[0][1])),
            "right_foot",
            diagnostics,
        )

    left_ankle = (
        keypoints[left_ankle_index]
        if left_ankle_index < len(keypoints)
        else None
    )
    right_ankle = (
        keypoints[right_ankle_index]
        if right_ankle_index < len(keypoints)
        else None
    )

    left_ok = (
        left_ankle is not None
        and left_ankle_score >= threshold
    )
    right_ok = (
        right_ankle is not None
        and right_ankle_score >= threshold
    )

    if left_ok and right_ok:
        point = (left_ankle + right_ankle) / 2.0
        return (
            (float(point[0]), float(point[1])),
            "both_ankles",
            diagnostics,
        )

    if left_ok:
        return (
            (float(left_ankle[0]), float(left_ankle[1])),
            "left_ankle",
            diagnostics,
        )

    if right_ok:
        return (
            (float(right_ankle[0]), float(right_ankle[1])),
            "right_ankle",
            diagnostics,
        )

    return (
        bbox_ground_point(bbox),
        "bbox_fallback",
        diagnostics,
    )


def pose_geometry_features(
    pose_point: tuple[float, float],
    bbox: np.ndarray,
) -> dict[str, float]:
    x1, y1, x2, y2 = map(float, bbox)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    px, py = pose_point
    bbox_center_x = (x1 + x2) / 2.0

    return {
        "bbox_width_px": width,
        "bbox_height_px": height,
        "relative_pose_y": (py - y1) / height,
        "relative_pose_x_offset": (
            abs(px - bbox_center_x) / width
        ),
        "pose_minus_bbox_y_px": py - y2,
        "pose_minus_bbox_x_px": px - bbox_center_x,
    }


def pairwise_iou(
    detections: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    if len(detections) == 0 or len(gt_boxes) == 0:
        return np.empty((len(detections), len(gt_boxes)), dtype=np.float32)

    a = detections[:, None, :]
    b = gt_boxes[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(
        0.0, a[..., 3] - a[..., 1]
    )
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(
        0.0, b[..., 3] - b[..., 1]
    )

    union = area_a + area_b - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def match_by_iou(
    detections: np.ndarray,
    gt_boxes: np.ndarray,
    minimum_iou: float,
) -> dict[int, tuple[int, float]]:
    if len(detections) == 0 or len(gt_boxes) == 0:
        return {}

    ious = pairwise_iou(detections, gt_boxes)
    row_indices, col_indices = linear_sum_assignment(1.0 - ious)

    matches: dict[int, tuple[int, float]] = {}
    for det_idx, gt_idx in zip(row_indices, col_indices):
        iou = float(ious[det_idx, gt_idx])
        if iou >= minimum_iou:
            matches[int(det_idx)] = (int(gt_idx), iou)

    return matches


def project_actual_point_to_pitch(
    point_actual: tuple[float, float],
    image_info: dict[str, Any],
    actual_width: int,
    actual_height: int,
) -> np.ndarray:
    ann_width = int(image_info["width"])
    ann_height = int(image_info["height"])

    x, y = actual_point_to_annotation(
        point_actual,
        actual_width,
        actual_height,
        ann_width,
        ann_height,
    )

    normalized = np.asarray(
        [[
            (x - (ann_width - 1.0) / 2.0) / ann_width,
            (y - (ann_height - 1.0) / 2.0) / ann_width,
        ]],
        dtype=np.float32,
    )

    pitch = image_to_ground(
        image_info["camera_matrix"],
        image_info["undist_poly"],
        normalized,
    )

    if hasattr(pitch, "detach"):
        pitch = pitch.detach().cpu().numpy()
    else:
        pitch = np.asarray(pitch)

    return np.asarray(pitch, dtype=np.float32)[0, :2]


def classify_comparison(
    bbox_error: float,
    pose_error: float,
    margin: float,
) -> str:
    improvement = bbox_error - pose_error

    if improvement > margin:
        return "better"
    if improvement < -margin:
        return "worse"
    return "equal"


def correctness_transition(
    bbox_error: float,
    pose_error: float,
    threshold: float,
) -> str:
    bbox_ok = bbox_error <= threshold
    pose_ok = pose_error <= threshold

    if bbox_ok and pose_ok:
        return "both_correct"
    if bbox_ok and not pose_ok:
        return "bbox_only_correct"
    if not bbox_ok and pose_ok:
        return "method_only_correct"
    return "both_wrong"


def draw_point(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    center = (int(round(point[0])), int(round(point[1])))
    # cv2.circle(image, center, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = max(0, min(x, image.shape[1] - tw - 4))
    y = max(th + 4, min(y, image.shape[0] - baseline - 4))

    cv2.rectangle(
        image,
        (x, y - th - 4),
        (x + tw + 4, y + baseline + 2),
        TEXT_BG,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - 2),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_legend(image: np.ndarray) -> None:
    entries = [
        ("GT", GT_COLOR),
        ("bbox", BBOX_COLOR),
        ("pose", METHOD_COLOR),
    ]

    x = 15
    y = 25

    for text, color in entries:
        cv2.circle(
            image,
            (x + 6, y - 5),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        x += 95


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        / args.split
        / args.pose_model
    )
    images_output = output_root / "images"
    images_output.mkdir(parents=True, exist_ok=True)

    image_dir = resolve_image_directory(dataset_root, args.split)
    annotation_path = resolve_annotation_file(dataset_root, args.split)
    dataset = load_json(annotation_path)

    images_by_id = {
        int(item["id"]): item
        for item in dataset.get("images", [])
    }

    annotations_by_image: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for annotation in dataset.get("annotations", []):
        if int(annotation.get("category_id", 1)) == 1:
            annotations_by_image[
                int(annotation["image_id"])
            ].append(annotation)

    schema = POSE_MODEL_SCHEMAS[args.pose_model]

    cache_dir = (
        cache_root
        / args.split
        / args.pose_model
        / "pose_raw"
    )
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Pose cache directory not found: {cache_dir}"
        )

    cache_files = sorted(cache_dir.glob("*.npz"))
    if not cache_files:
        raise RuntimeError(
            f"No .npz pose cache files found in {cache_dir}"
        )

    csv_path = output_root / "pose_detection_debug.csv"
    summary_path = output_root / "pose_debug_summary.json"

    fieldnames = [
        "image_id", "file_name", "detection_index",
        "yolo_confidence",
        "matched_gt", "gt_annotation_id", "bbox_match_iou",
        "comparison", "improvement_m", "transition_0_25m",
        "bbox_error_m", "pose_error_m",
        "bbox_correct_0_25m", "pose_correct_0_25m",
        "pose_available", "pose_ground_source",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_width_px", "bbox_height_px",
        "gt_image_x", "gt_image_y",
        "bbox_image_x", "bbox_image_y",
        "pose_image_x", "pose_image_y",
        "gt_pitch_x", "gt_pitch_y",
        "bbox_pitch_x", "bbox_pitch_y",
        "pose_pitch_x", "pose_pitch_y",
        "relative_pose_y", "relative_pose_x_offset",
        "pose_minus_bbox_y_px", "pose_minus_bbox_x_px",
        "left_ankle_score", "right_ankle_score",
        "left_foot_selected_index", "left_foot_selected_score",
        "right_foot_selected_index", "right_foot_selected_score",
        "keypoints_above_threshold",
        "mean_keypoint_score", "max_keypoint_score",
    ]

    counts = Counter()
    source_counts = Counter()
    bbox_errors: list[float] = []
    pose_errors: list[float] = []
    improvements: list[float] = []

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for cache_path in tqdm(
            cache_files,
            desc="Rendering pose debug",
        ):
            with np.load(
                cache_path,
                allow_pickle=False,
            ) as data:
                image_id = int(data["image_id"])
                file_name = str(data["file_name"].item())
                actual_width = int(data["actual_width"])
                actual_height = int(data["actual_height"])
                boxes = np.asarray(
                    data["bboxes"],
                    dtype=np.float32,
                )
                scores = np.asarray(
                    data["detection_scores"],
                    dtype=np.float32,
                )
                keypoints = np.asarray(
                    data["keypoints"],
                    dtype=np.float32,
                )
                kp_scores = np.asarray(
                    data["keypoint_scores"],
                    dtype=np.float32,
                )

            if image_id not in images_by_id:
                print(
                    f"WARNING: image_id={image_id} from "
                    f"{cache_path.name} not in annotations; skipping."
                )
                continue

            image_info = images_by_id[image_id]

            image_path = image_dir / file_name
            if not image_path.is_file():
                image_path = image_dir / Path(file_name).name

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            h, w = image.shape[:2]
            if (w, h) != (
                actual_width,
                actual_height,
            ):
                raise RuntimeError(
                    f"Cache/image dimension mismatch for {file_name}: "
                    f"cache={(actual_width, actual_height)}, "
                    f"image={(w, h)}"
                )

            gt_annotations = annotations_by_image.get(
                image_id,
                [],
            )
            ann_width = int(image_info["width"])
            ann_height = int(image_info["height"])

            gt_boxes = np.asarray(
                [
                    annotation_bbox_to_actual(
                        gt["bbox"],
                        ann_width,
                        ann_height,
                        w,
                        h,
                    )
                    for gt in gt_annotations
                ],
                dtype=np.float32,
            ).reshape(-1, 4)

            keep_indices = np.flatnonzero(
                scores >= args.min_score
            )

            kept_boxes = (
                boxes[keep_indices]
                if len(keep_indices)
                else np.empty(
                    (0, 4),
                    dtype=np.float32,
                )
            )

            local_matches = match_by_iou(
                kept_boxes,
                gt_boxes,
                args.match_iou,
            )

            matches = {
                int(keep_indices[local_det_idx]): match
                for local_det_idx, match
                in local_matches.items()
            }

            draw_legend(image)

            for det_idx in keep_indices:
                det_idx = int(det_idx)
                box = boxes[det_idx]
                score = float(scores[det_idx])

                x1, y1, x2, y2 = map(float, box)

                cv2.rectangle(
                    image,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    BOX_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                bbox_point = bbox_ground_point(box)
                draw_point(
                    image,
                    bbox_point,
                    BBOX_COLOR,
                    args.point_radius,
                )

                pose_available = (
                    det_idx < len(keypoints)
                    and det_idx < len(kp_scores)
                )

                if pose_available:
                    pose_point, source, diagnostics = (
                        pose_ground_point_detail(
                            keypoints[det_idx],
                            kp_scores[det_idx],
                            box,
                            schema,
                            args.keypoint_conf,
                        )
                    )
                    geometry = pose_geometry_features(
                        pose_point,
                        box,
                    )
                    draw_point(
                        image,
                        pose_point,
                        METHOD_COLOR,
                        args.point_radius,
                    )
                    source_counts[source] += 1
                else:
                    pose_point = None
                    source = "pose_unavailable"
                    diagnostics = {
                        "left_ankle_score": None,
                        "right_ankle_score": None,
                        "left_foot_selected_index": None,
                        "left_foot_selected_score": None,
                        "right_foot_selected_index": None,
                        "right_foot_selected_score": None,
                        "keypoints_above_threshold": None,
                        "mean_keypoint_score": None,
                        "max_keypoint_score": None,
                    }
                    geometry = {
                        "bbox_width_px": max(x2 - x1, 1.0),
                        "bbox_height_px": max(y2 - y1, 1.0),
                        "relative_pose_y": None,
                        "relative_pose_x_offset": None,
                        "pose_minus_bbox_y_px": None,
                        "pose_minus_bbox_x_px": None,
                    }

                row: dict[str, Any] = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "detection_index": det_idx,
                    "yolo_confidence": score,
                    "matched_gt": False,
                    "gt_annotation_id": None,
                    "bbox_match_iou": None,
                    "comparison": "unmatched",
                    "improvement_m": None,
                    "transition_0_25m": "unmatched",
                    "bbox_error_m": None,
                    "pose_error_m": None,
                    "bbox_correct_0_25m": None,
                    "pose_correct_0_25m": None,
                    "pose_available": pose_available,
                    "pose_ground_source": source,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "gt_image_x": None,
                    "gt_image_y": None,
                    "bbox_image_x": bbox_point[0],
                    "bbox_image_y": bbox_point[1],
                    "pose_image_x": (
                        pose_point[0]
                        if pose_point is not None
                        else None
                    ),
                    "pose_image_y": (
                        pose_point[1]
                        if pose_point is not None
                        else None
                    ),
                    "gt_pitch_x": None,
                    "gt_pitch_y": None,
                    "bbox_pitch_x": None,
                    "bbox_pitch_y": None,
                    "pose_pitch_x": None,
                    "pose_pitch_y": None,
                    **geometry,
                    **diagnostics,
                }

                match = matches.get(det_idx)

                if match is None:
                    counts["unmatched_detection"] += 1
                    label = (
                        f"D{det_idx} conf={score:.2f} "
                        f"{source} UNMATCHED"
                    )
                    # draw_label(
                    #     image,
                    #     label,
                    #     int(round(x1)),
                    #     int(round(y1)) - 3,
                    # )
                    writer.writerow(row)
                    continue

                gt_idx, match_iou = match
                gt = gt_annotations[gt_idx]

                gt_point_ann = extract_gt_ground_point(gt)
                gt_point = annotation_point_to_actual(
                    gt_point_ann,
                    ann_width,
                    ann_height,
                    w,
                    h,
                )

                gt_pitch = np.asarray(
                    gt["position_on_pitch"][:2],
                    dtype=np.float32,
                )

                draw_point(
                    image,
                    gt_point,
                    GT_COLOR,
                    args.point_radius,
                )

                bbox_pitch = project_actual_point_to_pitch(
                    bbox_point,
                    image_info,
                    w,
                    h,
                )

                bbox_error = float(
                    np.linalg.norm(
                        bbox_pitch - gt_pitch
                    )
                )

                row.update({
                    "matched_gt": True,
                    "gt_annotation_id": int(gt["id"]),
                    "bbox_match_iou": match_iou,
                    "gt_image_x": gt_point[0],
                    "gt_image_y": gt_point[1],
                    "gt_pitch_x": float(gt_pitch[0]),
                    "gt_pitch_y": float(gt_pitch[1]),
                    "bbox_pitch_x": float(bbox_pitch[0]),
                    "bbox_pitch_y": float(bbox_pitch[1]),
                    "bbox_error_m": bbox_error,
                    "bbox_correct_0_25m": (
                        bbox_error
                        <= args.correct_threshold_m
                    ),
                })

                bbox_errors.append(bbox_error)

                if pose_point is None:
                    counts["pose_unavailable"] += 1
                    row["comparison"] = "pose_unavailable"
                    row["transition_0_25m"] = "pose_unavailable"
                    label = (
                        f"D{det_idx} B={bbox_error:.2f}m "
                        "POSE=NA"
                    )
                    draw_label(
                        image,
                        label,
                        int(round(x1)),
                        int(round(y1)) - 3,
                    )
                    writer.writerow(row)
                    continue

                pose_pitch = project_actual_point_to_pitch(
                    pose_point,
                    image_info,
                    w,
                    h,
                )

                pose_error = float(
                    np.linalg.norm(
                        pose_pitch - gt_pitch
                    )
                )
                improvement = bbox_error - pose_error

                comparison = classify_comparison(
                    bbox_error,
                    pose_error,
                    args.better_margin_m,
                )

                transition = correctness_transition(
                    bbox_error,
                    pose_error,
                    args.correct_threshold_m,
                )

                row.update({
                    "comparison": comparison,
                    "improvement_m": improvement,
                    "transition_0_25m": transition,
                    "pose_error_m": pose_error,
                    "pose_correct_0_25m": (
                        pose_error
                        <= args.correct_threshold_m
                    ),
                    "pose_pitch_x": float(pose_pitch[0]),
                    "pose_pitch_y": float(pose_pitch[1]),
                })

                counts[comparison] += 1
                counts[transition] += 1
                pose_errors.append(pose_error)
                improvements.append(improvement)

                sign = "+" if improvement >= 0 else ""

                label = (
                    f"D{det_idx} B={bbox_error:.2f} "
                    f"P={pose_error:.2f}m "
                    f"{comparison.upper()} "
                    f"{sign}{improvement:.2f} "
                    f"[{source}]"
                )

                if comparison == "worse":
                    draw_label(
                        image,
                        label,
                        int(round(x1)),
                        int(round(y1)) - 3,
                    )

                writer.writerow(row)

            # Unmatched GT players: green crosses.
            matched_gt_indices = {
                gt_idx
                for gt_idx, _ in matches.values()
            }

            for gt_idx, gt in enumerate(
                gt_annotations
            ):
                if gt_idx in matched_gt_indices:
                    continue

                gt_point_ann = extract_gt_ground_point(
                    gt
                )
                gt_point = annotation_point_to_actual(
                    gt_point_ann,
                    ann_width,
                    ann_height,
                    w,
                    h,
                )

                gx = int(round(gt_point[0]))
                gy = int(round(gt_point[1]))

                cv2.drawMarker(
                    image,
                    (gx, gy),
                    GT_COLOR,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            output_path = images_output / (
                Path(file_name).stem
                + "_pose_debug.jpg"
            )

            if (
                args.overwrite_images
                or not output_path.exists()
            ):
                if not cv2.imwrite(
                    str(output_path),
                    image,
                ):
                    raise RuntimeError(
                        f"Failed to save image: "
                        f"{output_path}"
                    )

    summary = {
        "split": args.split,
        "pose_model": args.pose_model,
        "pose_schema": schema,
        "keypoint_confidence_threshold": (
            args.keypoint_conf
        ),
        "minimum_detection_score": args.min_score,
        "bbox_gt_match_iou_threshold": (
            args.match_iou
        ),
        "better_margin_m": (
            args.better_margin_m
        ),
        "correct_threshold_m": (
            args.correct_threshold_m
        ),
        "counts": dict(counts),
        "pose_ground_source_counts": (
            dict(source_counts)
        ),
        "mean_bbox_error_m": (
            float(np.mean(bbox_errors))
            if bbox_errors
            else None
        ),
        "mean_pose_error_m": (
            float(np.mean(pose_errors))
            if pose_errors
            else None
        ),
        "mean_improvement_m_bbox_minus_pose": (
            float(np.mean(improvements))
            if improvements
            else None
        ),
        "positive_improvement_means": (
            "pose is more accurate than bbox"
        ),
        "csv": str(csv_path),
        "images": str(images_output),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nFinished pose debugging.")
    print(f"Images:  {images_output}")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Better:  {counts['better']}")
    print(f"Worse:   {counts['worse']}")
    print(f"Equal:   {counts['equal']}")
    print(f"No pose: {counts['pose_unavailable']}")
    print(
        "Unmatched detections: "
        f"{counts['unmatched_detection']}"
    )
    print("Ground-point sources:")
    for source, count in source_counts.most_common():
        print(f"  {source:20s} {count}")


if __name__ == "__main__":
    main()
