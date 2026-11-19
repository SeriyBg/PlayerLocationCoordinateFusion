from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rtmlib import RTMPose, ViTPose
from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pose estimation on the exact frozen YOLO detections stored "
            "by 01_cache_synloc_yolo.py. This script never runs YOLO."
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
        help=(
            "Root produced by 01_cache_synloc_yolo.py, e.g. "
            "output/synloc_yolo_cache."
        ),
    )
    parser.add_argument(
        "--pose-model",
        choices=sorted(POSE_MODELS),
        default="halpe26_m",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_pose_cache"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute pose cache files that already exist.",
    )

    return parser.parse_args()


def resolve_image_directory(
    root: Path,
    split: str,
) -> Path:
    aliases = {
        "train": ["train"],
        "val": [
            "val",
            "valid",
            "validation",
        ],
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
        f"Could not find image directory. Tried: {tried}"
    )


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def load_yolo_cache(
    path: Path,
) -> dict[str, Any]:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            "image_id": int(
                data["image_id"]
            ),
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


def normalize_pose_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )
    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if keypoints.ndim == 2:
        keypoints = (
            keypoints[
                np.newaxis,
                ...,
            ]
        )

    if (
        scores.ndim == 3
        and scores.shape[-1] == 1
    ):
        scores = scores[..., 0]

    if scores.ndim == 1:
        scores = (
            scores[
                np.newaxis,
                ...,
            ]
        )

    return keypoints, scores


def create_pose_model(
    config: dict[str, Any],
) -> RTMPose | ViTPose:
    kwargs = {
        "onnx_model": (
            config["onnx_model"]
        ),
        "model_input_size": (
            config["model_input_size"]
        ),
        "backend": "onnxruntime",
        "device": "cpu",
    }

    if (
        config["pose_class"]
        == "rtmpose"
    ):
        return RTMPose(
            **kwargs,
            to_openpose=False,
        )

    if (
        config["pose_class"]
        == "vitpose"
    ):
        return ViTPose(
            **kwargs
        )

    raise ValueError(
        "Unsupported pose class: "
        f"{config['pose_class']}"
    )


def save_pose_cache(
    path: Path,
    *,
    source_yolo_cache: str,
    yolo_data: dict[str, Any],
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        image_id=np.asarray(
            yolo_data["image_id"],
            dtype=np.int64,
        ),
        file_name=np.asarray(
            yolo_data["file_name"],
        ),
        actual_width=np.asarray(
            yolo_data[
                "actual_width"
            ],
            dtype=np.int32,
        ),
        actual_height=np.asarray(
            yolo_data[
                "actual_height"
            ],
            dtype=np.int32,
        ),
        annotation_width=np.asarray(
            yolo_data[
                "annotation_width"
            ],
            dtype=np.int32,
        ),
        annotation_height=np.asarray(
            yolo_data[
                "annotation_height"
            ],
            dtype=np.int32,
        ),
        bboxes=np.asarray(
            yolo_data["bboxes"],
            dtype=np.float32,
        ),
        detection_scores=np.asarray(
            yolo_data[
                "detection_scores"
            ],
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
        source_yolo_cache=np.asarray(
            source_yolo_cache,
        ),
    )


def verify_exact_detection_identity(
    yolo_data: dict[str, Any],
    pose_cache_path: Path,
) -> None:
    """
    Verify that an existing pose cache still contains the exact same
    frozen YOLO boxes and scores as the current canonical YOLO cache.
    """
    with np.load(
        pose_cache_path,
        allow_pickle=False,
    ) as data:
        cached_boxes = np.asarray(
            data["bboxes"],
            dtype=np.float32,
        )
        cached_scores = np.asarray(
            data["detection_scores"],
            dtype=np.float32,
        )

    if not np.array_equal(
        cached_boxes,
        yolo_data["bboxes"],
    ):
        raise RuntimeError(
            "Existing pose cache does not contain the exact same "
            f"YOLO boxes as canonical cache: {pose_cache_path}"
        )

    if not np.array_equal(
        cached_scores,
        yolo_data[
            "detection_scores"
        ],
    ):
        raise RuntimeError(
            "Existing pose cache does not contain the exact same "
            f"YOLO scores as canonical cache: {pose_cache_path}"
        )


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
    output_root = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    image_directory = (
        resolve_image_directory(
            dataset_root,
            args.split,
        )
    )

    yolo_cache_dir = (
        yolo_cache_root
        / args.split
        / "yolo_raw"
    )

    if not yolo_cache_dir.is_dir():
        raise FileNotFoundError(
            "Canonical YOLO cache not found: "
            f"{yolo_cache_dir}"
        )

    yolo_summary_path = (
        yolo_cache_root
        / args.split
        / "yolo_cache_summary.json"
    )

    if not yolo_summary_path.is_file():
        raise FileNotFoundError(
            "YOLO cache summary not found: "
            f"{yolo_summary_path}"
        )

    yolo_summary = load_json(
        yolo_summary_path
    )

    config = POSE_MODELS[
        args.pose_model
    ]
    expected_keypoints = int(
        config[
            "expected_keypoints"
        ]
    )

    output_dir = (
        output_root
        / args.split
        / args.pose_model
        / "pose_raw"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    yolo_cache_files = sorted(
        yolo_cache_dir.glob(
            "*.npz"
        )
    )

    if not yolo_cache_files:
        raise RuntimeError(
            "No canonical YOLO cache files "
            f"found in {yolo_cache_dir}"
        )

    print(
        "This script does NOT run YOLO."
    )
    print(
        f"Canonical YOLO cache: "
        f"{yolo_cache_dir}"
    )
    print(
        f"Images:               "
        f"{image_directory}"
    )
    print(
        f"Pose model:           "
        f"{args.pose_model} "
        f"({config['title']})"
    )
    print(
        "Pose device:          "
        "cpu (ONNX Runtime)"
    )
    print(
        f"Output:               "
        f"{output_dir}"
    )
    print(
        f"Cache files:          "
        f"{len(yolo_cache_files)}"
    )

    pose_model = create_pose_model(
        config
    )

    processed = 0
    skipped = 0
    total_detections = 0
    total_poses = 0

    for yolo_cache_path in tqdm(
        yolo_cache_files,
        desc=(
            f"{args.split} "
            f"{args.pose_model}"
        ),
    ):
        yolo_data = load_yolo_cache(
            yolo_cache_path
        )

        output_path = (
            output_dir
            / yolo_cache_path.name
        )

        if (
            output_path.is_file()
            and not args.overwrite
        ):
            verify_exact_detection_identity(
                yolo_data,
                output_path,
            )
            skipped += 1
            total_detections += len(
                yolo_data["bboxes"]
            )
            with np.load(
                output_path,
                allow_pickle=False,
            ) as data:
                total_poses += len(
                    data["keypoints"]
                )
            continue

        file_name = yolo_data[
            "file_name"
        ]

        image_path = (
            image_directory
            / file_name
        )

        if not image_path.is_file():
            image_path = (
                image_directory
                / Path(
                    file_name
                ).name
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

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

        if (
            actual_width
            != yolo_data[
                "actual_width"
            ]
            or actual_height
            != yolo_data[
                "actual_height"
            ]
        ):
            raise RuntimeError(
                "Image dimensions differ from "
                "canonical YOLO cache for "
                f"{file_name}."
            )

        boxes = yolo_data[
            "bboxes"
        ]

        total_detections += len(
            boxes
        )

        if len(boxes) == 0:
            keypoints = np.empty(
                (
                    0,
                    expected_keypoints,
                    2,
                ),
                dtype=np.float32,
            )
            keypoint_scores = (
                np.empty(
                    (
                        0,
                        expected_keypoints,
                    ),
                    dtype=np.float32,
                )
            )
        else:
            (
                keypoints,
                keypoint_scores,
            ) = pose_model(
                image,
                bboxes=boxes.tolist(),
            )

            (
                keypoints,
                keypoint_scores,
            ) = normalize_pose_output(
                keypoints,
                keypoint_scores,
            )

            if (
                keypoints.ndim != 3
                or keypoints.shape[1]
                != expected_keypoints
                or keypoints.shape[2]
                != 2
            ):
                raise RuntimeError(
                    f"{config['title']} returned "
                    f"{keypoints.shape}; expected "
                    f"(N,{expected_keypoints},2). "
                    f"Image: {image_path}"
                )

            if (
                len(keypoints)
                != len(boxes)
            ):
                raise RuntimeError(
                    "Pose/detection count mismatch. "
                    f"YOLO boxes={len(boxes)}, "
                    f"poses={len(keypoints)}, "
                    f"image={image_path}. "
                    "Stopping rather than breaking "
                    "cross-method detection identity."
                )

        save_pose_cache(
            output_path,
            source_yolo_cache=(
                str(
                    yolo_cache_path
                )
            ),
            yolo_data=yolo_data,
            keypoints=keypoints,
            keypoint_scores=(
                keypoint_scores
            ),
        )

        processed += 1
        total_poses += len(
            keypoints
        )

    summary = {
        "split": args.split,
        "pose_model": args.pose_model,
        "pose_title": config[
            "title"
        ],
        "pose_schema": config[
            "schema"
        ],
        "pose_input_size": config[
            "model_input_size"
        ],
        "expected_keypoints": (
            expected_keypoints
        ),
        "canonical_yolo_cache_root": (
            str(yolo_cache_root)
        ),
        "canonical_yolo_model": (
            yolo_summary.get(
                "yolo_model"
            )
        ),
        "canonical_yolo_settings": {
            "imgsz": yolo_summary.get(
                "imgsz"
            ),
            "conf": yolo_summary.get(
                "conf"
            ),
            "iou": yolo_summary.get(
                "iou"
            ),
            "max_det": yolo_summary.get(
                "max_det"
            ),
        },
        "processed_this_run": (
            processed
        ),
        "skipped_existing": (
            skipped
        ),
        "total_frozen_yolo_detections": (
            total_detections
        ),
        "total_pose_records": (
            total_poses
        ),
        "alignment_guarantee": (
            "bboxes and detection_scores are copied verbatim "
            "from the canonical YOLO cache; pose index i "
            "corresponds to canonical YOLO detection i."
        ),
        "cache_arrays": [
            "image_id",
            "file_name",
            "actual_width",
            "actual_height",
            "annotation_width",
            "annotation_height",
            "bboxes",
            "detection_scores",
            "keypoints",
            "keypoint_scores",
            "source_yolo_cache",
        ],
    }

    summary_path = (
        output_dir.parent
        / "pose_cache_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print("\nFinished.")
    print(
        f"Frozen YOLO detections: "
        f"{total_detections}"
    )
    print(
        f"Pose records:           "
        f"{total_poses}"
    )
    print(
        f"Processed this run:     "
        f"{processed}"
    )
    print(
        f"Skipped existing:       "
        f"{skipped}"
    )
    print(
        f"Cache: {output_dir}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
