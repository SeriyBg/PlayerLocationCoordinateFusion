from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO


DATASET_YAML = Path(
    os.getenv(
        "DATASET_YAML",
        "/workspace/data/SoccerNet/SpiideoSynLoc_yolo/synloc.yaml",
    )
)
PROJECT_DIRECTORY = Path(
    os.getenv("PROJECT_DIRECTORY", "/workspace/runs/synloc")
)
MODEL = os.getenv("MODEL", "/app/yolo26n.pt")


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_cache() -> bool | str:
    value = os.getenv("CACHE", "false").strip().lower()

    if value in {"0", "false", "no", "off"}:
        return False

    if value in {"1", "true", "yes", "on"}:
        return True

    if value in {"ram", "disk"}:
        return value

    raise ValueError(
        "CACHE must be false, true, ram, or disk; "
        f"received: {value!r}"
    )


def select_device() -> str | int:
    requested = os.getenv("DEVICE")

    if requested:
        return int(requested) if requested.isdigit() else requested

    if torch.cuda.is_available():
        return 0

    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def main() -> None:
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    device = select_device()
    PROJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"Torch version:    {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    print(f"Training device:  {device}")
    print(f"Dataset:          {DATASET_YAML}")
    print(f"Model:            {MODEL}")
    print(f"Output directory: {PROJECT_DIRECTORY}")

    model = YOLO(MODEL)

    model.train(
        data=str(DATASET_YAML),
        epochs=env_int("EPOCHS", 100),
        patience=env_int("PATIENCE", 20),
        imgsz=env_int("IMGSZ", 1280),
        batch=env_int("BATCH", 2),
        device=device,
        workers=env_int("WORKERS", 8),
        project=str(PROJECT_DIRECTORY),
        name=os.getenv("RUN_NAME", "yolo26n_fullhd"),
        exist_ok=env_bool("EXIST_OK", False),
        seed=env_int("SEED", 42),
        deterministic=env_bool("DETERMINISTIC", True),
        optimizer=os.getenv("OPTIMIZER", "auto"),
        cos_lr=env_bool("COS_LR", True),
        cache=env_cache(),
        rect=env_bool("RECT", False),
        mosaic=env_float("MOSAIC", 0.5),
        close_mosaic=env_int("CLOSE_MOSAIC", 10),
        fliplr=env_float("FLIPLR", 0.5),
        degrees=env_float("DEGREES", 0.0),
        perspective=env_float("PERSPECTIVE", 0.0),
        amp=env_bool("AMP", True),
        plots=env_bool("PLOTS", True),
        verbose=env_bool("VERBOSE", True),
    )


if __name__ == "__main__":
    main()
