from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import SAM, YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Cache YOLO detections and raw SAM masks for a SynLoc split."
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--yolo-model", required=True)
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--split", choices=["train", "val", "test", "challenge"], default="val")
    p.add_argument("--output-dir", type=Path, default=Path("output/synloc_sam_cache"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def select_device():
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


def resize_mask(mask, width, height):
    if mask.shape == (height, width):
        return mask.astype(np.uint8)
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def main():
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    image_dir = resolve_image_directory(root, args.split)
    annotation_path = resolve_annotation_file(root, args.split)
    metadata = load_image_metadata(annotation_path)

    cache_dir = args.output_dir.expanduser().resolve() / args.split / "sam_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    device = select_device()
    print(f"Device:       {device}")
    print(f"Images:       {image_dir}")
    print(f"Annotations:  {annotation_path}")
    print(f"YOLO model:   {args.yolo_model}")
    print(f"SAM model:    {args.sam_model}")
    print(f"Output:       {cache_dir}")
    print(f"Image count:  {len(image_paths)}")

    yolo = YOLO(args.yolo_model)
    sam = SAM(args.sam_model)

    total_boxes = 0
    total_masks = 0
    mismatches = 0
    skipped = 0

    for image_path in tqdm(image_paths, desc=f"{args.split} SAM cache"):
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
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            total_boxes += len(boxes)

            sam_results = sam.predict(
                source=image,
                bboxes=boxes.tolist(),
                device=device,
                verbose=False,
                save=False,
            )

            if not sam_results or sam_results[0].masks is None:
                masks = np.empty((0, h, w), dtype=np.uint8)
            else:
                raw = sam_results[0].masks.data.detach().cpu().numpy()
                masks = (
                    np.stack([resize_mask(m, w, h) for m in raw], axis=0)
                    if len(raw)
                    else np.empty((0, h, w), dtype=np.uint8)
                )

            total_masks += len(masks)
            if len(masks) != len(boxes):
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
            masks=masks,
        )

    summary = {
        "split": args.split,
        "dataset_root": str(root),
        "image_directory": str(image_dir),
        "annotation_file": str(annotation_path) if annotation_path else None,
        "yolo_model": args.yolo_model,
        "sam_model": args.sam_model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "image_count": len(image_paths),
        "skipped_existing": skipped,
        "yolo_boxes_this_run": total_boxes,
        "sam_masks_this_run": total_masks,
        "images_with_box_mask_count_mismatch": mismatches,
        "cache_arrays": [
            "image_id", "file_name",
            "actual_width", "actual_height",
            "annotation_width", "annotation_height",
            "bboxes", "detection_scores", "masks"
        ],
        "mask_shape": "(N,H,W), uint8 0/1",
        "coordinates": "actual image pixel coordinates",
    }

    with (cache_dir.parent / "sam_cache_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished.")
    print(f"YOLO boxes this run: {total_boxes}")
    print(f"SAM masks this run:  {total_masks}")
    print(f"Mismatched images:   {mismatches}")
    print(f"Skipped existing:    {skipped}")


if __name__ == "__main__":
    main()
