from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from rtmlib import RTMPose, ViTPose
from ultralytics import YOLO


IMAGE_PATH = Path(
    "data/spiideo/images/test/000017.jpg"
)

OUTPUT_DIR = Path("../output")

# YOLO_MODEL = "yolo26n.pt"
YOLO_MODEL = "models/yolo26n_fullhd/yolo26n_fullhd-2/weights/best_original.pt"
YOLO_CONFIDENCE = 0.20
YOLO_IMAGE_SIZE = 1280
KEYPOINT_CONFIDENCE = 0.30

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

# Models to execute. Add/remove names without changing the main loop.
ACTIVE_POSE_MODELS = (
    "coco17_m",
    "halpe26_m",
    "halpe26_x_384",
    "vitpose25_l",
    # The 133-point models are substantially heavier; enable as needed.
    # "dwpose133_l_384",
    # "rtmw133_l_384",
)


COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

HALPE26_EDGES = COCO17_EDGES + [
    (17, 18),  # head -> neck
    (18, 5),   # neck -> left shoulder
    (18, 6),   # neck -> right shoulder
    (18, 19),  # neck -> pelvis
    (19, 11),  # pelvis -> left hip
    (19, 12),  # pelvis -> right hip
    (15, 20),  # left ankle -> left big toe
    (15, 22),  # left ankle -> left small toe
    (15, 24),  # left ankle -> left heel
    (16, 21),  # right ankle -> right big toe
    (16, 23),  # right ankle -> right small toe
    (16, 25),  # right ankle -> right heel
]


# ViTPose COCO+Feet-25 has a different body ordering from COCO-17.
COCO25_EDGES = [
    (17, 15), (15, 12),
    (18, 16), (16, 13),
    (12, 14), (13, 14),
    (5, 14),
    (6, 5), (7, 5),
    (6, 8), (8, 10),
    (7, 9), (9, 11),
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4),
    (17, 21), (19, 20), (19, 21),
    (18, 24), (22, 23), (22, 24),
    (5, 0),
]

# COCO-WholeBody keeps the first 17 COCO body points and adds six feet.
COCO133_EDGES = COCO17_EDGES + [
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
]

SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "coco17": {
        "edges": COCO17_EDGES,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [],
        "right_foot": [],
    },
    "halpe26": {
        "edges": HALPE26_EDGES,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [20, 22, 24],
        "right_foot": [21, 23, 25],
    },
    "coco25": {
        "edges": COCO25_EDGES,
        "left_ankle": 17,
        "right_ankle": 18,
        "left_foot": [19, 20, 21],
        "right_foot": [22, 23, 24],
    },
    "coco133": {
        "edges": COCO133_EDGES,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": [17, 18, 19],
        "right_foot": [20, 21, 22],
    },
}


def select_yolo_device() -> str | int:
    """Select the best device for Ultralytics YOLO."""
    if torch.cuda.is_available():
        return 0

    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def normalize_pose_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize RTMPose output to:
        keypoints: (people, keypoints, 2)
        scores:    (people, keypoints)
    """
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
    """Return the confident candidate with the largest image y-coordinate."""
    candidates = [
        keypoints[index]
        for index in indices
        if float(scores[index]) >= threshold
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda point: float(point[1]))


def get_ground_point(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
    schema: str,
    threshold: float = 0.3,
) -> tuple[float, float, str]:
    """Estimate a pitch-contact point for any configured schema."""
    if schema not in SCHEMA_DEFINITIONS:
        raise ValueError(f"Unsupported keypoint schema: {schema}")

    definition = SCHEMA_DEFINITIONS[schema]

    left_foot = get_best_lower_point(
        keypoints,
        scores,
        indices=list(definition["left_foot"]),
        threshold=threshold,
    )
    right_foot = get_best_lower_point(
        keypoints,
        scores,
        indices=list(definition["right_foot"]),
        threshold=threshold,
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

    x1, _, x2, y2 = bbox
    return (
        float((x1 + x2) / 2.0),
        float(y2),
        "bbox bottom centre",
    )


def draw_pose_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    edges: list[tuple[int, int]],
    threshold: float,
) -> np.ndarray:
    """
    Draw either a 17-point or 26-point skeleton.

    A custom drawer is used because support for Halpe26 visualization
    depends on the installed rtmlib version.
    """
    output = image.copy()

    for person_keypoints, person_scores in zip(keypoints, scores):
        for start_index, end_index in edges:
            if (
                float(person_scores[start_index]) < threshold
                or float(person_scores[end_index]) < threshold
            ):
                continue

            start = tuple(
                np.rint(person_keypoints[start_index]).astype(int)
            )
            end = tuple(
                np.rint(person_keypoints[end_index]).astype(int)
            )

            cv2.line(
                output,
                start,
                end,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for point, score in zip(person_keypoints, person_scores):
            if float(score) < threshold:
                continue

            centre = tuple(np.rint(point).astype(int))

            cv2.circle(
                output,
                centre,
                3,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    return output


def add_title(image: np.ndarray, title: str) -> None:
    """Draw a readable title on the image."""
    cv2.rectangle(
        image,
        (0, 0),
        (620, 52),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visualize_model_result(
    image: np.ndarray,
    model_name: str,
    model_title: str,
    schema: str,
    bboxes: np.ndarray,
    detection_scores: np.ndarray,
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
) -> np.ndarray:
    """Create a complete visualization for one pose checkpoint."""
    if schema not in SCHEMA_DEFINITIONS:
        raise ValueError(f"Unsupported keypoint schema: {schema}")

    edges = list(SCHEMA_DEFINITIONS[schema]["edges"])

    visualization = draw_pose_skeleton(
        image=image,
        keypoints=keypoints,
        scores=keypoint_scores,
        edges=edges,
        threshold=KEYPOINT_CONFIDENCE,
    )

    add_title(visualization, model_title)

    for person_index, (
        bbox,
        detection_score,
        person_keypoints,
        person_scores,
    ) in enumerate(
        zip(
            bboxes,
            detection_scores,
            keypoints,
            keypoint_scores,
        ),
        start=1,
    ):
        x1, y1, x2, y2 = bbox.astype(int)

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            visualization,
            f"person {person_index}: {detection_score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        ground_x, ground_y, ground_source = get_ground_point(
            keypoints=person_keypoints,
            scores=person_scores,
            bbox=bbox,
            schema=schema,
            threshold=KEYPOINT_CONFIDENCE,
        )

        cv2.circle(
            visualization,
            (round(ground_x), round(ground_y)),
            6,
            (0, 0, 255),
            -1,
        )

        print(
            f"[{model_name}] Person {person_index}: "
            f"ground point=({ground_x:.1f}, {ground_y:.1f}), "
            f"source={ground_source}"
        )

    return visualization


def resize_for_display(
    image: np.ndarray,
    max_width: int = 1800,
    max_height: int = 1000,
) -> np.ndarray:
    """Resize only the preview window, preserving the saved resolution."""
    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )



def create_pose_model(config: dict[str, Any]) -> RTMPose | ViTPose:
    """Create the correct rtmlib low-level model for a config entry."""
    common_kwargs = {
        "onnx_model": config["onnx_model"],
        "model_input_size": config["model_input_size"],
        "backend": "onnxruntime",
        "device": "cpu",
    }

    pose_class = str(config["pose_class"])

    if pose_class == "rtmpose":
        return RTMPose(
            **common_kwargs,
            to_openpose=False,
        )

    if pose_class == "vitpose":
        return ViTPose(**common_kwargs)

    raise ValueError(f"Unsupported pose class: {pose_class}")


def make_comparison_grid(
    images: list[np.ndarray],
    columns: int = 2,
) -> np.ndarray:
    """Arrange an arbitrary number of same-sized images in a grid."""
    if not images:
        raise ValueError("No visualizations were generated")

    height, width = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns

    blank = np.zeros_like(images[0])
    padded = images + [blank] * (rows * columns - len(images))

    row_images = []
    for row_index in range(rows):
        row = padded[
            row_index * columns:(row_index + 1) * columns
        ]
        row_images.append(np.hstack(row))

    return np.vstack(row_images)

def main() -> None:
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Detect people once with YOLO
    # ---------------------------------------------------------

    yolo = YOLO(YOLO_MODEL)

    yolo_result = yolo.predict(
        source=image,
        classes=[0],
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        device=select_yolo_device(),
        verbose=False,
    )[0]

    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        print("No players were detected.")
        return

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

    print(f"YOLO detected {len(bboxes)} people.")

    # ---------------------------------------------------------
    # 2. Run both pose checkpoints on exactly the same boxes
    # ---------------------------------------------------------

    visualizations: list[np.ndarray] = []

    for model_name in ACTIVE_POSE_MODELS:
        config = POSE_MODELS[model_name]
        print(f"\nLoading {config['title']}...")

        pose_model = create_pose_model(config)

        keypoints, keypoint_scores = pose_model(
            image,
            bboxes=bboxes.tolist(),
        )

        keypoints, keypoint_scores = normalize_pose_output(
            keypoints,
            keypoint_scores,
        )

        expected_keypoints = int(config["expected_keypoints"])

        if keypoints.shape[1] != expected_keypoints:
            raise RuntimeError(
                f"{config['title']} returned "
                f"{keypoints.shape[1]} keypoints; "
                f"expected {expected_keypoints}."
            )

        if len(keypoints) != len(bboxes):
            raise RuntimeError(
                f"{config['title']} returned poses for "
                f"{len(keypoints)} people, but YOLO produced "
                f"{len(bboxes)} boxes."
            )

        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Scores shape: {keypoint_scores.shape}")

        visualization = visualize_model_result(
            image=image,
            model_name=model_name,
            model_title=str(config["title"]),
            schema=str(config["schema"]),
            bboxes=bboxes,
            detection_scores=detection_scores,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
        )

        output_path = (
            OUTPUT_DIR
            / f"{IMAGE_PATH.stem}_{model_name}.jpg"
        )

        if not cv2.imwrite(str(output_path), visualization):
            raise RuntimeError(
                f"Could not save output to {output_path}"
            )

        print(f"Saved: {output_path}")
        visualizations.append(visualization)

        # Release each ONNX model before loading the next one.
        del pose_model

    # ---------------------------------------------------------
    # 3. Save and show side-by-side comparison
    # ---------------------------------------------------------

    comparison = make_comparison_grid(visualizations, columns=2)

    comparison_path = (
        OUTPUT_DIR
        / f"{IMAGE_PATH.stem}_pose_comparison.jpg"
    )

    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError(
            f"Could not save output to {comparison_path}"
        )

    print(f"\nComparison saved to: {comparison_path}")

    cv2.imshow(
        "Pose checkpoint comparison",
        resize_for_display(comparison),
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()