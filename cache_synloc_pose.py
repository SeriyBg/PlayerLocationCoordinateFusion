from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from rtmlib import RTMPose, ViTPose
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cache YOLO detections plus raw pose keypoints/scores "
            "for one selected pose model on a SynLoc split."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--yolo-model", required=True)
    p.add_argument("--pose-model", choices=sorted(POSE_MODELS), default="halpe26_m")
    p.add_argument("--split", choices=["train", "val", "test", "challenge"], default="val")
    p.add_argument("--output-dir", type=Path, default=Path("output/synloc_pose_cache"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def select_yolo_device():
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
    tried = []
    for name in aliases[split]:
        for candidate in (root / "images" / name, root / name):
            tried.append(candidate)
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Could not find image directory. Tried: {tried}")


def resolve_annotation_file(root: Path, split: str) -> Path | None:
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
        "challenge": ["challenge"],
    }
    for name in aliases[split]:
        path = root / "annotations" / f"{name}.json"
        if path.is_file():
            return path
    return None


def load_image_metadata(annotation_path: Path | None):
    if annotation_path is None:
        return {}
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data.get("images", []):
        result[str(item["file_name"])] = item
        result[Path(str(item["file_name"])).name] = item
    return result


def normalize_pose_output(keypoints, scores):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim == 1:
        scores = scores[np.newaxis, ...]

    return keypoints, scores


def create_pose_model(config):
    kwargs = {
        "onnx_model": config["onnx_model"],
        "model_input_size": config["model_input_size"],
        "backend": "onnxruntime",
        "device": "cpu",
    }

    if config["pose_class"] == "rtmpose":
        return RTMPose(**kwargs, to_openpose=False)
    if config["pose_class"] == "vitpose":
        return ViTPose(**kwargs)

    raise ValueError(f"Unsupported pose class: {config['pose_class']}")


def main():
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    image_dir = resolve_image_directory(root, args.split)
    annotation_path = resolve_annotation_file(root, args.split)
    metadata = load_image_metadata(annotation_path)
    config = POSE_MODELS[args.pose_model]

    cache_dir = (
        args.output_dir.expanduser().resolve()
        / args.split
        / args.pose_model
        / "pose_raw"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    yolo_device = select_yolo_device()
    expected_k = int(config["expected_keypoints"])

    print(f"YOLO device:  {yolo_device}")
    print("Pose device:  cpu (ONNX Runtime)")
    print(f"Images:       {image_dir}")
    print(f"Annotations:  {annotation_path}")
    print(f"YOLO model:   {args.yolo_model}")
    print(f"Pose model:   {args.pose_model} ({config['title']})")
    print(f"Output:       {cache_dir}")
    print(f"Image count:  {len(image_paths)}")

    yolo = YOLO(args.yolo_model)
    pose = create_pose_model(config)

    total_boxes = 0
    total_poses = 0
    mismatches = 0
    skipped = 0

    for image_path in tqdm(image_paths, desc=f"{args.split} {args.pose_model} cache"):
        relative_name = image_path.relative_to(image_dir).as_posix()
        info = metadata.get(relative_name) or metadata.get(image_path.name) or {}
        image_id = int(info.get("id", -1))
        stem = f"{image_id:06d}_{image_path.stem}" if image_id >= 0 else image_path.stem
        out_path = cache_dir / f"{stem}.npz"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")

        h, w = image.shape[:2]
        ann_w = int(info.get("width", w))
        ann_h = int(info.get("height", h))

        result = yolo.predict(
            source=image,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=yolo_device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            keypoints = np.empty((0, expected_k, 2), dtype=np.float32)
            kp_scores = np.empty((0, expected_k), dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            total_boxes += len(boxes)

            keypoints, kp_scores = pose(image, bboxes=boxes.tolist())
            keypoints, kp_scores = normalize_pose_output(keypoints, kp_scores)

            if keypoints.shape[1:] != (expected_k, 2):
                raise RuntimeError(
                    f"{config['title']} returned {keypoints.shape}; "
                    f"expected (N,{expected_k},2). Image: {image_path}"
                )

            total_poses += len(keypoints)
            if len(keypoints) != len(boxes):
                mismatches += 1

        np.savez_compressed(
            out_path,
            image_id=np.asarray(image_id, dtype=np.int64),
            file_name=np.asarray(relative_name),
            actual_width=np.asarray(w, dtype=np.int32),
            actual_height=np.asarray(h, dtype=np.int32),
            annotation_width=np.asarray(ann_w, dtype=np.int32),
            annotation_height=np.asarray(ann_h, dtype=np.int32),
            bboxes=boxes,
            detection_scores=scores,
            keypoints=keypoints,
            keypoint_scores=kp_scores,
        )

    summary = {
        "split": args.split,
        "dataset_root": str(root),
        "image_directory": str(image_dir),
        "annotation_file": str(annotation_path) if annotation_path else None,
        "yolo_model": args.yolo_model,
        "pose_model": args.pose_model,
        "pose_title": config["title"],
        "pose_schema": config["schema"],
        "pose_input_size": config["model_input_size"],
        "expected_keypoints": expected_k,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "image_count": len(image_paths),
        "skipped_existing": skipped,
        "yolo_boxes_this_run": total_boxes,
        "poses_this_run": total_poses,
        "images_with_box_pose_count_mismatch": mismatches,
        "cache_arrays": [
            "image_id", "file_name",
            "actual_width", "actual_height",
            "annotation_width", "annotation_height",
            "bboxes", "detection_scores",
            "keypoints", "keypoint_scores"
        ],
        "coordinates": "actual image pixel coordinates",
    }

    with (cache_dir.parent / "pose_cache_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished.")
    print(f"YOLO boxes this run: {total_boxes}")
    print(f"Poses this run:      {total_poses}")
    print(f"Mismatched images:   {mismatches}")
    print(f"Skipped existing:    {skipped}")


if __name__ == "__main__":
    main()
