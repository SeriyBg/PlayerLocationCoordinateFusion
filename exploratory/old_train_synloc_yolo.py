from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


DATASET_YAML = Path(
    "data/spiideo/synloc.yaml"
)

PROJECT_DIRECTORY = Path(
    "../runs/synloc"
)


def select_device() -> str | int:
    if torch.cuda.is_available():
        return 0

    mps_backend = getattr(torch.backends, "mps", None)

    if (
        mps_backend is not None
        and mps_backend.is_available()
    ):
        return "mps"

    return "cpu"


def main() -> None:
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    device = select_device()

    print(f"Training device: {device}")
    print(f"Dataset: {DATASET_YAML}")

    # .pt means fine-tuning from pretrained COCO weights.
    model = YOLO("../yolo26n.pt")

    model.train(
        data=str(DATASET_YAML),

        # Training duration
        epochs=1,#100,
        patience=20,

        # Small football players benefit from a larger inference size.
        imgsz=640,#1280,

        # Reduce to 1 if the Mac runs out of unified memory.
        batch=2,

        device=device,

        # workers=0 is slower but avoids many macOS multiprocessing
        # problems during the first run.
        workers=0,

        # Output
        project=str(PROJECT_DIRECTORY),
        name="yolo26n_fullhd",
        exist_ok=False,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Optimizer and scheduling
        optimizer="auto",
        cos_lr=True,

        # Dataset handling
        cache=False,
        rect=False,

        # Moderate augmentation
        mosaic=0.5,
        close_mosaic=10,
        fliplr=0.5,
        degrees=0.0,
        perspective=0.0,

        # Logging
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()