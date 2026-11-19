from __future__ import annotations

# Enable CPU fallback for torchvision operators not implemented on MPS.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import SAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM on the exact frozen YOLO detections stored by "
            "01_cache_synloc_yolo.py. This script never runs YOLO."
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
        "--sam-model",
        type=str,
        default="mobile_sam.pt",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/synloc_sam_cache"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute SAM cache files that already exist.",
    )

    return parser.parse_args()


def select_device() -> str | int:
    if torch.cuda.is_available():
        return 0

    mps = getattr(
        torch.backends,
        "mps",
        None,
    )

    if (
        mps is not None
        and mps.is_available()
    ):
        return "mps"

    return "cpu"


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
            tried.append(
                candidate
            )

            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        "Could not find image directory. "
        f"Tried: {tried}"
    )


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


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


def resize_mask_to_image(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if (
        mask.shape
        == (
            image_height,
            image_width,
        )
    ):
        return (
            mask
            .astype(
                np.uint8
            )
        )

    resized = cv2.resize(
        mask.astype(
            np.uint8
        ),
        (
            image_width,
            image_height,
        ),
        interpolation=(
            cv2.INTER_NEAREST
        ),
    )

    return resized.astype(
        np.uint8
    )


def save_sam_cache(
    path: Path,
    *,
    source_yolo_cache: str,
    yolo_data: dict[str, Any],
    masks: np.ndarray,
    mask_valid: np.ndarray,
) -> None:
    """
    masks[i] is always aligned with canonical YOLO detection i.

    If SAM failed to produce a usable mask for a detection,
    mask_valid[i] == False and masks[i] is all zeros.
    """
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
            yolo_data[
                "bboxes"
            ],
            dtype=np.float32,
        ),
        detection_scores=np.asarray(
            yolo_data[
                "detection_scores"
            ],
            dtype=np.float32,
        ),
        masks=np.asarray(
            masks,
            dtype=np.uint8,
        ),
        mask_valid=np.asarray(
            mask_valid,
            dtype=bool,
        ),
        source_yolo_cache=np.asarray(
            source_yolo_cache,
        ),
    )


def verify_exact_detection_identity(
    yolo_data: dict[str, Any],
    sam_cache_path: Path,
) -> None:
    """
    Ensure an existing SAM cache was produced from the exact current
    canonical YOLO boxes and confidence scores.
    """
    with np.load(
        sam_cache_path,
        allow_pickle=False,
    ) as data:
        cached_boxes = np.asarray(
            data["bboxes"],
            dtype=np.float32,
        )
        cached_scores = np.asarray(
            data[
                "detection_scores"
            ],
            dtype=np.float32,
        )

    if not np.array_equal(
        cached_boxes,
        yolo_data["bboxes"],
    ):
        raise RuntimeError(
            "Existing SAM cache does not contain the exact same "
            f"YOLO boxes as canonical cache: {sam_cache_path}"
        )

    if not np.array_equal(
        cached_scores,
        yolo_data[
            "detection_scores"
        ],
    ):
        raise RuntimeError(
            "Existing SAM cache does not contain the exact same "
            f"YOLO scores as canonical cache: {sam_cache_path}"
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

    output_dir = (
        output_root
        / args.split
        / "sam_raw"
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

    device = select_device()

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
        f"SAM model:            "
        f"{args.sam_model}"
    )
    print(
        f"SAM device:           "
        f"{device}"
    )
    print(
        f"Output:               "
        f"{output_dir}"
    )
    print(
        f"Cache files:          "
        f"{len(yolo_cache_files)}"
    )

    sam_model = SAM(
        args.sam_model
    )

    processed = 0
    skipped = 0
    total_detections = 0
    total_valid_masks = 0
    total_missing_masks = 0
    count_mismatch_images = 0

    for yolo_cache_path in tqdm(
        yolo_cache_files,
        desc=(
            f"{args.split} SAM cache"
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

            with np.load(
                output_path,
                allow_pickle=False,
            ) as data:
                mask_valid = np.asarray(
                    data["mask_valid"],
                    dtype=bool,
                )

            total_detections += len(
                yolo_data["bboxes"]
            )
            total_valid_masks += int(
                mask_valid.sum()
            )
            total_missing_masks += int(
                (~mask_valid).sum()
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

        number_of_detections = len(
            boxes
        )
        total_detections += (
            number_of_detections
        )

        # Index-aligned arrays: mask i always belongs to YOLO detection i.
        masks = np.zeros(
            (
                number_of_detections,
                actual_height,
                actual_width,
            ),
            dtype=np.uint8,
        )

        mask_valid = np.zeros(
            (
                number_of_detections,
            ),
            dtype=bool,
        )

        if number_of_detections > 0:
            prediction = (
                sam_model.predict(
                    source=image,
                    bboxes=(
                        boxes.tolist()
                    ),
                    device=device,
                    verbose=False,
                    save=False,
                )
            )

            raw_masks: np.ndarray | None = None

            if (
                prediction
                and prediction[0].masks
                is not None
            ):
                raw_masks = (
                    prediction[0]
                    .masks.data
                    .detach()
                    .cpu()
                    .numpy()
                )

            if raw_masks is not None:
                if (
                    len(raw_masks)
                    != number_of_detections
                ):
                    count_mismatch_images += 1

                usable_count = min(
                    len(raw_masks),
                    number_of_detections,
                )

                for index in range(
                    usable_count
                ):
                    resized_mask = (
                        resize_mask_to_image(
                            raw_masks[index],
                            actual_width,
                            actual_height,
                        )
                    )

                    if int(
                        resized_mask.sum()
                    ) > 0:
                        masks[index] = (
                            resized_mask
                        )
                        mask_valid[
                            index
                        ] = True

        valid_count = int(
            mask_valid.sum()
        )
        missing_count = (
            number_of_detections
            - valid_count
        )

        total_valid_masks += (
            valid_count
        )
        total_missing_masks += (
            missing_count
        )

        save_sam_cache(
            output_path,
            source_yolo_cache=(
                str(
                    yolo_cache_path
                )
            ),
            yolo_data=yolo_data,
            masks=masks,
            mask_valid=(
                mask_valid
            ),
        )

        processed += 1

    summary = {
        "split": args.split,
        "sam_model": args.sam_model,
        "sam_device": str(
            device
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
        "total_valid_sam_masks": (
            total_valid_masks
        ),
        "total_missing_sam_masks": (
            total_missing_masks
        ),
        "images_with_sam_count_mismatch": (
            count_mismatch_images
        ),
        "alignment_guarantee": (
            "bboxes and detection_scores are copied verbatim "
            "from the canonical YOLO cache; masks[i] belongs to "
            "canonical YOLO detection i. mask_valid[i] indicates "
            "whether SAM produced a non-empty mask."
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
            "masks",
            "mask_valid",
            "source_yolo_cache",
        ],
        "mask_format": (
            "Full-resolution uint8 binary masks. "
            "Shape = (N,H,W). Invalid/missing masks are all zeros."
        ),
    }

    summary_path = (
        output_dir.parent
        / "sam_cache_summary.json"
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
        f"Valid SAM masks:        "
        f"{total_valid_masks}"
    )
    print(
        f"Missing SAM masks:      "
        f"{total_missing_masks}"
    )
    print(
        f"Count-mismatch images:  "
        f"{count_mismatch_images}"
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
