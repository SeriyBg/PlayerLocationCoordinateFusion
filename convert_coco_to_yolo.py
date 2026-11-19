from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


SYNLOC_ROOT = Path(
    os.getenv(
        "SYNLOC_ROOT",
        "/workspace/data/SoccerNet/SpiideoSynLoc",
    )
)

YOLO_ROOT = Path(
    os.getenv(
        "YOLO_ROOT",
        "/workspace/data/SoccerNet/SpiideoSynLoc_yolo",
    )
)


def resolve_image_directory(dataset_root: Path, split: str) -> Path:
    candidates = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
    }

    for directory_name in candidates[split]:
        candidate = dataset_root / directory_name

        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find the {split} image directory under "
        f"{dataset_root}. Tried: {candidates[split]}"
    )


def resolve_annotation_file(dataset_root: Path, split: str) -> Path:
    candidates = {
        "train": [dataset_root / "annotations" / "train.json"],
        "val": [
            dataset_root / "annotations" / "val.json",
            dataset_root / "annotations" / "valid.json",
            dataset_root / "annotations" / "validation.json",
        ],
    }

    for candidate in candidates[split]:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find the {split} annotation file. "
        f"Tried: {candidates[split]}"
    )


def create_directory_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        if target.resolve() == source:
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(
            f"{target} already exists and is not a symbolic link. "
            "Remove it or choose another YOLO_ROOT."
        )

    target.symlink_to(source, target_is_directory=True)


def load_annotations(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_person_category_ids(dataset: dict) -> set[int]:
    person_ids = {
        int(category["id"])
        for category in dataset.get("categories", [])
        if category.get("name", "").lower()
        in {"person", "player", "athlete"}
    }

    return person_ids or {1}


def clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def convert_bbox_to_yolo(
    bbox: list[float],
    annotation_width: int,
    annotation_height: int,
    actual_width: int,
    actual_height: int,
) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None

    x, y, width, height = map(float, bbox)

    if (
        width <= 0
        or height <= 0
        or annotation_width <= 0
        or annotation_height <= 0
        or actual_width <= 0
        or actual_height <= 0
    ):
        return None

    scale_x = actual_width / annotation_width
    scale_y = actual_height / annotation_height

    x1 = clip(x * scale_x, 0.0, float(actual_width))
    y1 = clip(y * scale_y, 0.0, float(actual_height))
    x2 = clip((x + width) * scale_x, 0.0, float(actual_width))
    y2 = clip((y + height) * scale_y, 0.0, float(actual_height))

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 1.0 or box_height <= 1.0:
        return None

    return (
        clip(((x1 + x2) / 2.0) / actual_width, 0.0, 1.0),
        clip(((y1 + y2) / 2.0) / actual_height, 0.0, 1.0),
        clip(box_width / actual_width, 0.0, 1.0),
        clip(box_height / actual_height, 0.0, 1.0),
    )


def prepare_split(split: str, source_root: Path, output_root: Path) -> None:
    image_directory = resolve_image_directory(source_root, split)
    annotation_path = resolve_annotation_file(source_root, split)

    print(f"\nPreparing split: {split}")
    print(f"Images:      {image_directory}")
    print(f"Annotations: {annotation_path}")

    dataset = load_annotations(annotation_path)
    person_category_ids = find_person_category_ids(dataset)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in dataset.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))

        if category_id in person_category_ids:
            annotations_by_image[int(annotation["image_id"])].append(
                annotation
            )

    output_image_directory = output_root / "images" / split
    output_label_directory = output_root / "labels" / split

    create_directory_symlink(image_directory, output_image_directory)
    output_label_directory.mkdir(parents=True, exist_ok=True)

    image_count = 0
    bbox_count = 0
    invalid_bbox_count = 0
    dimension_mismatch_count = 0

    for image_info in dataset.get("images", []):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        source_image_path = image_directory / file_name

        if not source_image_path.is_file():
            raise FileNotFoundError(
                f"Image listed in JSON does not exist: {source_image_path}"
            )

        annotation_width = int(image_info["width"])
        annotation_height = int(image_info["height"])

        with Image.open(source_image_path) as image:
            actual_width, actual_height = image.size

        if (
            actual_width != annotation_width
            or actual_height != annotation_height
        ):
            dimension_mismatch_count += 1

        label_path = output_label_directory / Path(file_name).with_suffix(
            ".txt"
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            converted_bbox = convert_bbox_to_yolo(
                bbox=annotation["bbox"],
                annotation_width=annotation_width,
                annotation_height=annotation_height,
                actual_width=actual_width,
                actual_height=actual_height,
            )

            if converted_bbox is None:
                invalid_bbox_count += 1
                continue

            x_center, y_center, width, height = converted_bbox
            lines.append(
                f"0 {x_center:.8f} {y_center:.8f} "
                f"{width:.8f} {height:.8f}"
            )
            bbox_count += 1

        label_path.write_text("\n".join(lines), encoding="utf-8")
        image_count += 1

    print(f"Images processed:             {image_count}")
    print(f"Player boxes written:         {bbox_count}")
    print(f"Invalid boxes skipped:        {invalid_bbox_count}")
    print(
        "Images requiring dimension scaling: "
        f"{dimension_mismatch_count}"
    )


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "synloc.yaml"
    configuration = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "player"},
    }

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            configuration,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return yaml_path


def main() -> None:
    if not SYNLOC_ROOT.is_dir():
        raise FileNotFoundError(
            f"SynLoc dataset directory not found: {SYNLOC_ROOT}"
        )

    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        prepare_split(
            split=split,
            source_root=SYNLOC_ROOT,
            output_root=YOLO_ROOT,
        )

    yaml_path = write_dataset_yaml(YOLO_ROOT)

    print("\nDataset preparation completed.")
    print(f"YOLO dataset: {YOLO_ROOT}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
