from __future__ import annotations

import os
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader


SOCCERNET_ROOT = Path(
    os.getenv("SOCCERNET_ROOT", "/workspace/data/SoccerNet")
)
SYNLOC_VERSION = os.getenv("SYNLOC_VERSION", "fullhd").strip()
SYNLOC_SPLITS = [
    split.strip()
    for split in os.getenv(
        "SYNLOC_SPLITS",
        "train,valid,test,challenge",
    ).split(",")
    if split.strip()
]


def extract_archives(directory: Path) -> None:
    archives = sorted(directory.glob("*.zip"))

    if not archives:
        print(f"No ZIP archives found under {directory}.")
        print("They may already have been extracted.")
        return

    for archive in archives:
        marker = directory / f".{archive.name}.extracted"

        if marker.is_file():
            print(f"Already extracted: {archive.name}")
            continue

        print(f"Extracting: {archive}")

        with zipfile.ZipFile(archive, "r") as zip_file:
            zip_file.extractall(directory)

        marker.write_text("ok\n", encoding="utf-8")


def main() -> None:
    SOCCERNET_ROOT.mkdir(parents=True, exist_ok=True)

    if not SYNLOC_SPLITS:
        raise ValueError("SYNLOC_SPLITS must contain at least one split.")

    print(f"SoccerNet directory: {SOCCERNET_ROOT}")
    print(f"SynLoc version:      {SYNLOC_VERSION}")
    print(f"Splits:              {SYNLOC_SPLITS}")

    downloader = SoccerNetDownloader(
        LocalDirectory=str(SOCCERNET_ROOT)
    )

    kwargs: dict[str, object] = {
        "task": "SpiideoSynLoc",
        "split": SYNLOC_SPLITS,
    }

    if SYNLOC_VERSION:
        kwargs["version"] = SYNLOC_VERSION

    downloader.downloadDataTask(**kwargs)
    extract_archives(SOCCERNET_ROOT)

    expected_root = SOCCERNET_ROOT / "SpiideoSynLoc"

    if expected_root.is_dir():
        print(f"SynLoc dataset ready at: {expected_root}")
    else:
        print(
            "Download/extraction completed, but the expected directory "
            f"was not found: {expected_root}"
        )
        print(
            "Inspect the downloaded archive layout and set SYNLOC_ROOT "
            "to the actual extracted dataset directory."
        )


if __name__ == "__main__":
    main()
