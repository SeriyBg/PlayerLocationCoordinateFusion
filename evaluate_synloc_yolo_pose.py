from __future__ import annotations

# Enable CPU fallback for PyTorch operators that are not implemented on MPS,
# e.g. torchvision NMS on some macOS/PyTorch combinations.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from rtmlib import RTMPose, ViTPose
from tqdm import tqdm
from ultralytics import YOLO
from xtcocotools.coco import COCO

from sskit import image_to_ground, match
from sskit.coco import LocSimCOCOeval


# ---------------------------------------------------------------------------
# Pose model configurations copied from the single-image experiment
# ---------------------------------------------------------------------------

POSE_MODELS: dict[str, dict[str, Any]] = {
    "coco17_m": {
        "title": "RTMPose-M COCO-17",
        "pose_class": "rtmpose",
        "schema": "coco17",
        "onnx_model": (
            "https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-"
            "e48f03d0_20230504.zip"
        ),
        "model_input_size": (192, 256),
        "expected_keypoints": 17,
    },
    "halpe26_m": {
        "title": "RTMPose-M Body8-Halpe26",
        "pose_class": "rtmpose",
        "schema": "halpe26",
        "onnx_model": (
            "https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-"
            "256x192-4d3e73dd_20230605.zip"
        ),
        "model_input_size": (192, 256),
        "expected_keypoints": 26,
    },
    "halpe26_x_384": {
        "title": "RTMPose-X Body8-Halpe26 384x288",
        "pose_class": "rtmpose",
        "schema": "halpe26",
        "onnx_model": (
            "https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/onnx_sdk/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-"
            "384x288-7fb6e239_20230606.zip"
        ),
        "model_input_size": (288, 384),
        "expected_keypoints": 26,
    },
    "vitpose25_l": {
        "title": "ViTPose-L COCO+Feet-25",
        "pose_class": "vitpose",
        "schema": "coco25",
        "onnx_model": (
            "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/"
            "main/onnx/coco_25/vitpose-l-coco_25.onnx"
        ),
        "model_input_size": (192, 256),
        "expected_keypoints": 25,
    },
    "dwpose133_l_384": {
        "title": "DWPose-L WholeBody-133 384x288",
        "pose_class": "rtmpose",
        "schema": "coco133",
        "onnx_model": (
            "https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/onnx_sdk/"
            "rtmpose-l_simcc-ucoco_dw-ucoco_270e-"
            "384x288-2438fd99_20230728.zip"
        ),
        "model_input_size": (288, 384),
        "expected_keypoints": 133,
    },
    "rtmw133_l_384": {
        "title": "RTMW-L Cocktail14 WholeBody-133 384x288",
        "pose_class": "rtmpose",
        "schema": "coco133",
        "onnx_model": (
            "https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-"
            "384x288_20231122.zip"
        ),
        "model_input_size": (288, 384),
        "expected_keypoints": 133,
    },
}


SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    # All schemas used here retain the COCO body ordering for the hips:
    # 11 = left hip, 12 = right hip.
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
            "Evaluate one selected YOLO + pose-estimation pipeline on "
            "SoccerNet Spiideo SynLoc."
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
        help="Path to your fine-tuned YOLO best_original.pt.",
    )
    parser.add_argument(
        "--pose-model",
        choices=sorted(POSE_MODELS),
        default="halpe26_m",
        help="Exactly one pose checkpoint to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_pose_eval"),
    )

    parser.add_argument(
        "--pose-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional root created by cache_synloc_pose.py, e.g. "
            "output/synloc_pose_cache. Existing per-image .npz files are "
            "used instead of rerunning YOLO/pose. Missing cache files are "
            "computed and saved automatically."
        ),
    )

    parser.add_argument(
        "--ground-point-mode",
        choices=["foot_center", "pelvis_lowest", "pelvis_bbox_bottom", "foot_bbox_bottom"],
        default="pelvis_bbox_bottom",
        help=(
            "foot_center reproduces the old foot/ankle estimator. "
            "pelvis_lowest uses pelvis x (midpoint of confident hips) and "
            "the y coordinate of the lowest confident foot keypoint; if "
            "feet are unavailable it falls back to ankles, then bbox bottom."
        ),
    )

    # YOLO settings.
    parser.add_argument("--imgsz", type=int, default=1280)

    # 0.05 is preferable for mAP/F1-threshold evaluation because low-score
    # detections remain available to the evaluator. Use --conf 0.20 to exactly
    # reproduce the single-image script.
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)

    # Same keypoint threshold as the single-image script.
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save resumable progress every N images.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard a previous progress checkpoint.",
    )

    # Normally unset for validation so LocSimCOCOeval selects an F1-optimal
    # score threshold. Pass the validation threshold when evaluating test.
    parser.add_argument(
        "--bbox-score-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--pose-score-threshold",
        type=float,
        default=None,
    )

    return parser.parse_args()


def select_yolo_device() -> str | int:
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
            dataset_root / "images" / "valid",
            dataset_root / "images" / "validation",
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
        f"Could not find images for split '{split}'. "
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
        f"Could not find annotations for split '{split}'. "
        f"Tried: {candidates[split]}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    temp.replace(path)


def normalize_pose_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]

    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]

    if scores.ndim == 1:
        scores = scores[np.newaxis, ...]

    return keypoints, scores


def get_best_lower_point(
    keypoints: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
    threshold: float,
) -> np.ndarray | None:
    candidates = [
        keypoints[index]
        for index in indices
        if index < len(scores)
        and index < len(keypoints)
        and float(scores[index]) >= threshold
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda point: float(point[1]),
    )


def get_ground_point_foot_center(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, float, str]:
    """
    Original estimator from the previous evaluator.

    Prefer the lowest keypoint on each foot and average left/right feet.
    If feet are unavailable, use ankles, then bbox bottom-center.
    """
    definition = SCHEMA_DEFINITIONS[schema]

    left_foot = get_best_lower_point(
        keypoints,
        scores,
        list(definition["left_foot"]),
        threshold,
    )
    right_foot = get_best_lower_point(
        keypoints,
        scores,
        list(definition["right_foot"]),
        threshold,
    )

    if left_foot is not None and right_foot is not None:
        point = (left_foot + right_foot) / 2.0
        return float(point[0]), float(point[1]), "both feet"

    if left_foot is not None:
        return float(left_foot[0]), float(left_foot[1]), "left foot"

    if right_foot is not None:
        return float(right_foot[0]), float(right_foot[1]), "right foot"

    left_ankle_index = int(definition["left_ankle"])
    right_ankle_index = int(definition["right_ankle"])

    left_ankle = keypoints[left_ankle_index]
    right_ankle = keypoints[right_ankle_index]
    left_score = float(scores[left_ankle_index])
    right_score = float(scores[right_ankle_index])

    if left_score >= threshold and right_score >= threshold:
        point = (left_ankle + right_ankle) / 2.0
        return float(point[0]), float(point[1]), "both ankles"

    if left_score >= threshold:
        return float(left_ankle[0]), float(left_ankle[1]), "left ankle"

    if right_score >= threshold:
        return float(right_ankle[0]), float(right_ankle[1]), "right ankle"

    x1, _, x2, y2 = map(float, bbox)
    return (
        (x1 + x2) / 2.0,
        y2,
        "bbox bottom centre",
    )


def get_pelvis_x(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, str]:
    """
    Estimate the horizontal location of the pelvis.

    Prefer the midpoint of the two confident hip keypoints. If only one hip
    passes the confidence threshold, use that hip's x coordinate. If neither
    hip is reliable, fall back to the YOLO bbox center x.
    """
    definition = SCHEMA_DEFINITIONS[schema]

    left_index = int(definition["left_hip"])
    right_index = int(definition["right_hip"])

    left_ok = (
        left_index < len(keypoints)
        and left_index < len(scores)
        and float(scores[left_index]) >= threshold
    )
    right_ok = (
        right_index < len(keypoints)
        and right_index < len(scores)
        and float(scores[right_index]) >= threshold
    )

    if left_ok and right_ok:
        pelvis_x = (
            float(keypoints[left_index][0])
            + float(keypoints[right_index][0])
        ) / 2.0
        return pelvis_x, "both hips"

    if left_ok:
        return float(keypoints[left_index][0]), "left hip"

    if right_ok:
        return float(keypoints[right_index][0]), "right hip"

    x1, _, x2, _ = map(float, bbox)
    return (x1 + x2) / 2.0, "bbox centre x"


def get_lowest_support_y(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, str]:
    """
    Return the image y coordinate of the lowest reliable contact-like point.

    Priority:
      1. lowest confident foot keypoint over BOTH feet;
      2. lowest confident ankle;
      3. YOLO bbox bottom y.

    Unlike the old estimator, left/right feet are NOT averaged vertically.
    """
    definition = SCHEMA_DEFINITIONS[schema]

    foot_indices = (
        list(definition["left_foot"])
        + list(definition["right_foot"])
    )

    foot_candidates: list[tuple[float, int]] = []
    for index in foot_indices:
        if (
            index < len(keypoints)
            and index < len(scores)
            and float(scores[index]) >= threshold
        ):
            foot_candidates.append(
                (float(keypoints[index][1]), int(index))
            )

    if foot_candidates:
        lowest_y, lowest_index = max(
            foot_candidates,
            key=lambda item: item[0],
        )
        return lowest_y, f"lowest foot kpt {lowest_index}"

    ankle_indices = [
        int(definition["left_ankle"]),
        int(definition["right_ankle"]),
    ]

    ankle_candidates: list[tuple[float, int]] = []
    for index in ankle_indices:
        if (
            index < len(keypoints)
            and index < len(scores)
            and float(scores[index]) >= threshold
        ):
            ankle_candidates.append(
                (float(keypoints[index][1]), int(index))
            )

    if ankle_candidates:
        lowest_y, lowest_index = max(
            ankle_candidates,
            key=lambda item: item[0],
        )
        return lowest_y, f"lowest ankle kpt {lowest_index}"

    _, _, _, y2 = map(float, bbox)
    return y2, "bbox bottom y"


def get_ground_point_pelvis_lowest(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, float, str]:
    """
    Approximate the dataset's pelvis-to-ground target in image space.

    x := horizontal pelvis location
         (midpoint of confident left/right hips when possible)

    y := vertical level of the lowest confident foot keypoint
         (ankle/bbox fallback when foot keypoints are unavailable)

    This deliberately separates horizontal body-centre estimation from
    vertical ground-contact estimation.
    """
    pelvis_x, pelvis_source = get_pelvis_x(
        keypoints,
        scores,
        bbox,
        schema,
        threshold,
    )

    ground_y, support_source = get_lowest_support_y(
        keypoints,
        scores,
        bbox,
        schema,
        threshold,
    )

    return (
        float(pelvis_x),
        float(ground_y),
        f"{pelvis_source} + {support_source}",
    )


def get_ground_point_pelvis_bbox_bottom(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
) -> tuple[float, float, str]:
    """
    Estimate the ground point by projecting the pelvis horizontally
    onto the bottom edge of the YOLO bounding box.

    x:
        midpoint of left/right hips when both are confident;
        one hip if only one is confident;
        bbox center if neither hip is reliable.

    y:
        always the bottom edge of the YOLO bbox.
    """

    pelvis_x, pelvis_source = get_pelvis_x(
        keypoints=keypoints,
        scores=scores,
        bbox=bbox,
        schema=schema,
        threshold=threshold,
    )

    x1, _, x2, y2 = map(float, bbox)

    # Keep the estimated x inside the detected player's bbox.
    ground_x = float(
        np.clip(
            pelvis_x,
            x1,
            x2,
        )
    )

    ground_y = float(y2)

    return (
        ground_x,
        ground_y,
        f"{pelvis_source} + bbox bottom",
    )


def get_ground_point(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float,
    mode: str,
) -> tuple[float, float, str]:
    if mode == "foot_center":
        return get_ground_point_foot_center(
            keypoints,
            scores,
            bbox,
            schema,
            threshold,
        )

    if mode == "pelvis_lowest":
        return get_ground_point_pelvis_lowest(
            keypoints,
            scores,
            bbox,
            schema,
            threshold,
        )

    if mode == "pelvis_bbox_bottom":
        return get_ground_point_pelvis_bbox_bottom(
            keypoints,
            scores,
            bbox,
            schema,
            threshold,
        )

    raise ValueError(
        f"Unknown ground-point mode: {mode}"
    )


def get_pose_cache_path(
    cache_root: Path,
    split: str,
    pose_model: str,
    image_info: dict[str, Any],
) -> Path:
    image_id = int(image_info["id"])
    image_stem = Path(str(image_info["file_name"])).stem

    return (
        cache_root.resolve()
        / split
        / pose_model
        / "pose_raw"
        / f"{image_id:06d}_{image_stem}.npz"
    )


def load_pose_cache(
    cache_path: Path,
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with np.load(
        cache_path,
        allow_pickle=False,
    ) as data:
        actual_width = int(data["actual_width"])
        actual_height = int(data["actual_height"])
        bboxes = np.asarray(
            data["bboxes"],
            dtype=np.float32,
        )
        detection_scores = np.asarray(
            data["detection_scores"],
            dtype=np.float32,
        )
        keypoints = np.asarray(
            data["keypoints"],
            dtype=np.float32,
        )
        keypoint_scores = np.asarray(
            data["keypoint_scores"],
            dtype=np.float32,
        )

    return (
        actual_width,
        actual_height,
        bboxes,
        detection_scores,
        keypoints,
        keypoint_scores,
    )


def save_pose_cache(
    cache_path: Path,
    image_info: dict[str, Any],
    actual_width: int,
    actual_height: int,
    bboxes: np.ndarray,
    detection_scores: np.ndarray,
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
) -> None:
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
            bboxes,
            dtype=np.float32,
        ),
        detection_scores=np.asarray(
            detection_scores,
            dtype=np.float32,
        ),
        keypoints=np.asarray(
            keypoints,
            dtype=np.float32,
        ),
        keypoint_scores=np.asarray(
            keypoint_scores,
            dtype=np.float32,
        ),
    )



def bbox_ground_point(
    bbox: np.ndarray,
) -> tuple[float, float]:
    x1, _, x2, y2 = map(float, bbox)
    return (x1 + x2) / 2.0, y2


def clip_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, image_width - 1
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, image_height - 1
    )
    return boxes


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

    return [x1, y1, x2 - x1, y2 - y1]


def make_detection(
    detection_id: int,
    image_id: int,
    score: float,
    bbox_xywh: list[float],
    ground_point: tuple[float, float],
) -> dict[str, Any]:
    x, y, width, height = bbox_xywh
    ground_x, ground_y = ground_point

    # The SynLoc evaluator can project an image-space keypoint onto the pitch.
    # Index 1 is used, matching the devkit convention. Index 0 is a dummy bbox
    # center so that keypoint 1 stores our estimated ground-contact location.
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
        "area": float(max(width, 0.0) * max(height, 0.0)),
        "keypoints": [
            float(center_x),
            float(center_y),
            1.0,
            float(ground_x),
            float(ground_y),
            1.0,
        ],
    }


def create_pose_model(
    config: dict[str, Any],
) -> RTMPose | ViTPose:
    common_kwargs = {
        "onnx_model": config["onnx_model"],
        "model_input_size": config["model_input_size"],
        "backend": "onnxruntime",
        "device": "cpu",
    }

    if config["pose_class"] == "rtmpose":
        return RTMPose(
            **common_kwargs,
            to_openpose=False,
        )

    if config["pose_class"] == "vitpose":
        return ViTPose(**common_kwargs)

    raise ValueError(
        f"Unsupported pose class: {config['pose_class']}"
    )


def save_progress(
    path: Path,
    completed_image_ids: set[int],
    bbox_results: list[dict[str, Any]],
    pose_results: list[dict[str, Any]],
    next_detection_id: int,
    ground_source_counts: Counter,
) -> None:
    save_json_atomic(
        path,
        {
            "completed_image_ids": sorted(completed_image_ids),
            "bbox_results": bbox_results,
            "pose_results": pose_results,
            "next_detection_id": next_detection_id,
            "ground_source_counts": dict(ground_source_counts),
        },
    )


def run_inference(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Counter,
    Counter,
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

    config = POSE_MODELS[args.pose_model]

    # Separate output subdirectories by ground-point rule so an experiment
    # with pelvis_lowest cannot overwrite an earlier foot_center experiment.
    output_dir = (
        args.output_dir.resolve()
        / args.split
        / args.pose_model
        / args.ground_point_mode
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bbox_results_path = (
        output_dir
        / "yolo_bbox_results.json"
    )
    pose_results_path = (
        output_dir
        / f"yolo_pose_{args.pose_model}_results.json"
    )
    progress_path = output_dir / "progress.json"
    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    completed_image_ids: set[int] = set()
    bbox_results: list[dict[str, Any]] = []
    pose_results: list[dict[str, Any]] = []
    next_detection_id = 1
    ground_source_counts: Counter = Counter()
    cache_stats: Counter = Counter()

    if (
        progress_path.is_file()
        and not args.restart
    ):
        state = load_json(progress_path)

        completed_image_ids = set(
            map(
                int,
                state.get(
                    "completed_image_ids",
                    [],
                ),
            )
        )
        bbox_results = state.get(
            "bbox_results",
            [],
        )
        pose_results = state.get(
            "pose_results",
            [],
        )
        next_detection_id = int(
            state.get(
                "next_detection_id",
                1,
            )
        )
        ground_source_counts.update(
            state.get(
                "ground_source_counts",
                {},
            )
        )

        print(
            f"Resuming: "
            f"{len(completed_image_ids)}/"
            f"{len(images)} images completed."
        )

    pose_cache_root = (
        args.pose_cache_root.resolve()
        if args.pose_cache_root is not None
        else None
    )

    if pose_cache_root is not None:
        cache_directory = (
            pose_cache_root
            / args.split
            / args.pose_model
            / "pose_raw"
        )
        cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Pose cache:       "
            f"{cache_directory}"
        )

        cache_summary_path = (
            pose_cache_root
            / args.split
            / args.pose_model
            / "pose_cache_summary.json"
        )

        if cache_summary_path.is_file():
            cache_summary = load_json(
                cache_summary_path
            )

            cached_conf = cache_summary.get(
                "conf"
            )
            if (
                cached_conf is not None
                and args.conf
                < float(cached_conf)
            ):
                raise ValueError(
                    "The cache was generated with YOLO "
                    f"conf={cached_conf}, but this run requests "
                    f"conf={args.conf}. The missing lower-score "
                    "detections cannot be recovered from the cache. "
                    "Regenerate the cache with a confidence <= "
                    "the requested value."
                )

            for key, requested in (
                ("imgsz", args.imgsz),
                ("iou", args.iou),
                ("max_det", args.max_det),
            ):
                cached = cache_summary.get(key)
                if (
                    cached is not None
                    and cached != requested
                ):
                    print(
                        "WARNING: cached YOLO setting "
                        f"{key}={cached}, while this run "
                        f"requests {key}={requested}. "
                        "Cached frames will use the cached "
                        "predictions."
                    )

    yolo_device = select_yolo_device()

    print(f"YOLO device:      {yolo_device}")
    print("Pose device:      cpu (ONNX Runtime)")
    print(f"Split:            {args.split}")
    print(f"Images:           {image_directory}")
    print(f"Annotations:      {annotation_path}")
    print(f"YOLO model:       {args.yolo_model}")
    print(
        f"Pose model:       "
        f"{args.pose_model} "
        f"({config['title']})"
    )
    print(
        f"Pose input:       "
        f"{config['model_input_size']}"
    )
    print(
        f"Ground-point:     "
        f"{args.ground_point_mode}"
    )
    print(
        f"Keypoint conf:    "
        f"{args.keypoint_conf}"
    )
    print(f"YOLO conf:        {args.conf}")
    print(f"YOLO imgsz:       {args.imgsz}")
    print(f"Output:           {output_dir}")

    # Lazy initialization: if every pending frame exists in cache, neither
    # YOLO nor the pose model is loaded at all.
    yolo_model: YOLO | None = None
    pose_model: RTMPose | ViTPose | None = None

    def ensure_models_loaded() -> tuple[
        YOLO,
        RTMPose | ViTPose,
    ]:
        nonlocal yolo_model, pose_model

        if yolo_model is None:
            print(
                "\nLoading YOLO because a "
                "cache miss was encountered..."
            )
            yolo_model = YOLO(
                args.yolo_model
            )

        if pose_model is None:
            print(
                "Loading pose checkpoint because a "
                "cache miss was encountered..."
            )
            pose_model = create_pose_model(
                config
            )

        return yolo_model, pose_model

    pending_images = [
        image_info
        for image_info in images
        if int(image_info["id"])
        not in completed_image_ids
    ]

    expected_keypoints = int(
        config["expected_keypoints"]
    )

    for processed_count, image_info in enumerate(
        tqdm(
            pending_images,
            desc=(
                f"{args.split} "
                f"{args.pose_model} "
                f"{args.ground_point_mode}"
            ),
        ),
        start=1,
    ):
        image_id = int(image_info["id"])
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

        cache_path: Path | None = None
        if pose_cache_root is not None:
            cache_path = get_pose_cache_path(
                pose_cache_root,
                args.split,
                args.pose_model,
                image_info,
            )

        # -------------------------------------------------------------
        # Fast path: reuse cached YOLO boxes + pose outputs.
        # -------------------------------------------------------------
        if (
            cache_path is not None
            and cache_path.is_file()
        ):
            (
                actual_width,
                actual_height,
                bboxes,
                detection_scores,
                keypoints,
                keypoint_scores,
            ) = load_pose_cache(
                cache_path
            )

            cache_stats["hits"] += 1

            # A cache produced at a LOWER confidence can safely be filtered
            # upward to reproduce a higher requested confidence threshold.
            keep = (
                detection_scores
                >= float(args.conf)
            )

            bboxes = bboxes[keep]
            detection_scores = (
                detection_scores[keep]
            )
            keypoints = keypoints[keep]
            keypoint_scores = (
                keypoint_scores[keep]
            )

            bboxes = clip_boxes(
                bboxes,
                actual_width,
                actual_height,
            )

        # -------------------------------------------------------------
        # Slow path: infer only this missing frame and optionally cache it.
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
                pose_model_loaded,
            ) = ensure_models_loaded()

            yolo_result = (
                yolo_model_loaded.predict(
                    source=image,
                    classes=[0],
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    device=yolo_device,
                    verbose=False,
                )[0]
            )

            if (
                yolo_result.boxes is None
                or len(
                    yolo_result.boxes
                ) == 0
            ):
                bboxes = np.empty(
                    (0, 4),
                    dtype=np.float32,
                )
                detection_scores = np.empty(
                    (0,),
                    dtype=np.float32,
                )
                keypoints = np.empty(
                    (
                        0,
                        expected_keypoints,
                        2,
                    ),
                    dtype=np.float32,
                )
                keypoint_scores = np.empty(
                    (
                        0,
                        expected_keypoints,
                    ),
                    dtype=np.float32,
                )
            else:
                bboxes = (
                    yolo_result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                detection_scores = (
                    yolo_result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                bboxes = clip_boxes(
                    bboxes,
                    actual_width,
                    actual_height,
                )

                (
                    keypoints,
                    keypoint_scores,
                ) = pose_model_loaded(
                    image,
                    bboxes=bboxes.tolist(),
                )

                (
                    keypoints,
                    keypoint_scores,
                ) = normalize_pose_output(
                    keypoints,
                    keypoint_scores,
                )

            if cache_path is not None:
                save_pose_cache(
                    cache_path=cache_path,
                    image_info=image_info,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    bboxes=bboxes,
                    detection_scores=(
                        detection_scores
                    ),
                    keypoints=keypoints,
                    keypoint_scores=(
                        keypoint_scores
                    ),
                )
                cache_stats[
                    "new_files_written"
                ] += 1

        annotation_width = int(
            image_info["width"]
        )
        annotation_height = int(
            image_info["height"]
        )

        if len(bboxes) == 0:
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
                    pose_results,
                    next_detection_id,
                    ground_source_counts,
                )

            continue

        if (
            keypoints.ndim != 3
            or keypoints.shape[1]
            != expected_keypoints
            or keypoints.shape[2] != 2
        ):
            raise RuntimeError(
                f"{config['title']} returned "
                f"keypoints with shape "
                f"{keypoints.shape}; expected "
                f"(N, {expected_keypoints}, 2). "
                f"Image: {image_path}"
            )

        if len(keypoints) != len(bboxes):
            raise RuntimeError(
                f"{config['title']} returned "
                f"{len(keypoints)} poses for "
                f"{len(bboxes)} YOLO boxes. "
                f"Image: {image_path}"
            )

        for (
            bbox,
            detection_score,
            person_keypoints,
            person_scores,
        ) in zip(
            bboxes,
            detection_scores,
            keypoints,
            keypoint_scores,
        ):
            bbox_xywh_ann = (
                scale_box_to_annotation(
                    bbox,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            bbox_point_actual = (
                bbox_ground_point(
                    bbox
                )
            )
            bbox_point_ann = (
                scale_point_to_annotation(
                    bbox_point_actual,
                    actual_width,
                    actual_height,
                    annotation_width,
                    annotation_height,
                )
            )

            (
                pose_x,
                pose_y,
                source,
            ) = get_ground_point(
                keypoints=person_keypoints,
                scores=person_scores,
                bbox=bbox,
                schema=str(
                    config["schema"]
                ),
                threshold=(
                    args.keypoint_conf
                ),
                mode=(
                    args.ground_point_mode
                ),
            )

            ground_source_counts[
                source
            ] += 1

            pose_point_ann = (
                scale_point_to_annotation(
                    (pose_x, pose_y),
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
                    score=float(
                        detection_score
                    ),
                    bbox_xywh=(
                        bbox_xywh_ann
                    ),
                    ground_point=(
                        bbox_point_ann
                    ),
                )
            )

            pose_results.append(
                make_detection(
                    detection_id=(
                        next_detection_id
                    ),
                    image_id=image_id,
                    score=float(
                        detection_score
                    ),
                    bbox_xywh=(
                        bbox_xywh_ann
                    ),
                    ground_point=(
                        pose_point_ann
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
                pose_results,
                next_detection_id,
                ground_source_counts,
            )

    save_json_atomic(
        bbox_results_path,
        bbox_results,
    )
    save_json_atomic(
        pose_results_path,
        pose_results,
    )

    save_progress(
        progress_path,
        completed_image_ids,
        bbox_results,
        pose_results,
        next_detection_id,
        ground_source_counts,
    )

    print(
        f"\nBBox detections: "
        f"{len(bbox_results)}"
    )
    print(
        f"Pose detections: "
        f"{len(pose_results)}"
    )

    if pose_cache_root is not None:
        print(
            "Pose cache:"
            f"\n  hits:              "
            f"{cache_stats['hits']}"
            f"\n  misses:            "
            f"{cache_stats['misses']}"
            f"\n  new files written: "
            f"{cache_stats['new_files_written']}"
        )

    print("Ground-point sources:")
    for source, count in (
        ground_source_counts.most_common()
    ):
        print(
            f"  {source:35s} "
            f"{count}"
        )

    return (
        annotation_path,
        bbox_results_path,
        pose_results_path,
        summary_path,
        ground_source_counts,
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

    # Our estimated image-space ground point is keypoint index 1.
    evaluator.params.position_from_keypoint_index = 1

    if fixed_score_threshold is not None:
        evaluator.params.score_threshold = (
            fixed_score_threshold
        )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "mAP_LocSim_50_95": float(
            evaluator.stats[0]
        ),
        "score_threshold": float(
            evaluator.stats[15]
        ),
        "frame_accuracy_LocSim_50": float(
            evaluator.stats[16]
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

    # This follows the normalized image-coordinate convention used by sskit.
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

    return (
        pitch[:, :2]
        .detach()
        .cpu()
        .numpy()[0]
    )


def compute_baseline_style_stats(
    annotation_path: Path,
    results_path: Path,
    score_threshold: float,
    distance_threshold_m: float = 0.5,
) -> dict[str, float | int | None]:
    """
    Compute detected/missed/extra counts and matched localization error
    using the same 0.25 m matching threshold style as the SynLoc baseline.
    """
    dataset = load_json(annotation_path)

    images_by_id = {
        int(image["id"]): image
        for image in dataset.get("images", [])
    }

    gt_by_image: dict[
        int,
        list[np.ndarray],
    ] = defaultdict(list)

    for annotation in dataset.get(
        "annotations", []
    ):
        if int(
            annotation.get("category_id", 1)
        ) != 1:
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

    for result in load_json(results_path):
        if float(result["score"]) < score_threshold:
            continue

        image_id = int(result["image_id"])

        predictions_by_image[image_id].append(
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
            gt_by_image.get(image_id, []),
            dtype=np.float32,
        ).reshape(-1, 2)

        est = np.asarray(
            predictions_by_image.get(
                image_id, []
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

        total_detected += int(detected)
        total_missed += int(missed)
        total_extra += int(extra)
        distances_all.extend(
            map(float, distances)
        )

    precision_denominator = (
        total_detected + total_extra
    )
    recall_denominator = (
        total_detected + total_missed
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
        float(np.mean(distances_all))
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


def print_summary(
    title: str,
    data: dict[str, Any],
) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    locsim = data["locsim"]
    baseline = data["baseline_style"]

    print(
        "mAP-LocSim @ 0.50:0.95:  "
        f"{locsim['mAP_LocSim_50_95']:.6f}"
    )
    print(
        "Frame accuracy @ 0.50:    "
        f"{locsim['frame_accuracy_LocSim_50']:.6f}"
    )
    print(
        "Selected score threshold: "
        f"{locsim['score_threshold']:.6f}"
    )

    print("\n0.25 m matching:")
    print(
        f"Detected:                  "
        f"{baseline['detected']}"
    )
    print(
        f"Missed:                    "
        f"{baseline['missed']}"
    )
    print(
        f"Extra:                     "
        f"{baseline['extra']}"
    )
    print(
        f"Precision:                 "
        f"{baseline['precision']:.6f}"
    )
    print(
        f"Recall:                    "
        f"{baseline['recall']:.6f}"
    )

    mean_error = baseline[
        "mean_matched_localization_error_m"
    ]

    if mean_error is None:
        print("Mean localization error:   n/a")
    else:
        print(
            "Mean localization error:   "
            f"{mean_error:.6f} m"
        )


def main() -> None:
    args = parse_args()

    (
        annotation_path,
        bbox_results_path,
        pose_results_path,
        summary_path,
        ground_source_counts,
        cache_stats,
    ) = run_inference(args)

    print(
        "\nEvaluating YOLO bbox "
        "bottom-center..."
    )
    bbox_locsim = evaluate_locsim(
        annotation_path,
        bbox_results_path,
        args.bbox_score_threshold,
    )
    bbox_baseline = (
        compute_baseline_style_stats(
            annotation_path,
            bbox_results_path,
            bbox_locsim["score_threshold"],
        )
    )

    print(
        f"\nEvaluating YOLO + "
        f"{args.pose_model} "
        f"[{args.ground_point_mode}]..."
    )
    pose_locsim = evaluate_locsim(
        annotation_path,
        pose_results_path,
        args.pose_score_threshold,
    )
    pose_baseline = (
        compute_baseline_style_stats(
            annotation_path,
            pose_results_path,
            pose_locsim["score_threshold"],
        )
    )

    config = POSE_MODELS[args.pose_model]

    summary = {
        "split": args.split,
        "yolo_model": args.yolo_model,
        "pose_model": args.pose_model,
        "pose_title": config["title"],
        "pose_schema": config["schema"],
        "pose_input_size": (
            config["model_input_size"]
        ),
        "keypoint_confidence": (
            args.keypoint_conf
        ),
        "ground_point_mode": (
            args.ground_point_mode
        ),
        "pose_cache_root": (
            str(args.pose_cache_root.resolve())
            if args.pose_cache_root is not None
            else None
        ),
        "pose_cache_stats_this_run": dict(
            cache_stats
        ),
        "yolo_imgsz": args.imgsz,
        "yolo_inference_conf": args.conf,
        "yolo_nms_iou": args.iou,
        "ground_point_sources": dict(
            ground_source_counts
        ),
        "bbox": {
            "results_file": str(
                bbox_results_path
            ),
            "locsim": bbox_locsim,
            "baseline_style": (
                bbox_baseline
            ),
        },
        "pose": {
            "results_file": str(
                pose_results_path
            ),
            "locsim": pose_locsim,
            "baseline_style": (
                pose_baseline
            ),
        },
    }

    save_json_atomic(
        summary_path,
        summary,
    )

    print_summary(
        "YOLO — bbox bottom-center",
        summary["bbox"],
    )
    print_summary(
        (
            f"YOLO + {config['title']} "
            f"[{args.ground_point_mode}]"
        ),
        summary["pose"],
    )

    print("\nGround-point source counts:")
    for source, count in (
        ground_source_counts.most_common()
    ):
        print(f"  {source:20s} {count}")

    print("\nSaved:")
    print(f"  {bbox_results_path}")
    print(f"  {pose_results_path}")
    print(f"  {summary_path}")

    if args.split == "val":
        print(
            "\nFor an unbiased test-set run, "
            "reuse these validation thresholds:"
        )
        print(
            "  --bbox-score-threshold "
            f"{bbox_locsim['score_threshold']:.10f}"
        )
        print(
            "  --pose-score-threshold "
            f"{pose_locsim['score_threshold']:.10f}"
        )


if __name__ == "__main__":
    main()
