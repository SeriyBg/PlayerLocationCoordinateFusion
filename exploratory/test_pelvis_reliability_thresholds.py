from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pelvis reliability as a function of the horizontal "
            "distance between pelvis center and feet center."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="pose_reliability_detections.csv",
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 10],
        help=(
            "Pixel thresholds to test. Example: "
            "--thresholds 1 2 3 4 5 6 8 10"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pelvis_reliability_by_px.csv"),
    )

    return parser.parse_args()


def parse_bool(value: str) -> bool | None:
    value = value.strip().lower()

    if value == "true":
        return True
    if value == "false":
        return False

    return None


def parse_float(value: str) -> float | None:
    value = value.strip()

    if value == "":
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    if not np.isfinite(number):
        return None

    return number


def main() -> None:
    args = parse_args()

    rows: list[tuple[float, bool]] = []

    with args.csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "abs_pelvis_minus_feet_x_px",
            "pelvis_better_than_bbox",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            distance = parse_float(
                row["abs_pelvis_minus_feet_x_px"]
            )
            better = parse_bool(
                row["pelvis_better_than_bbox"]
            )

            if distance is None or better is None:
                continue

            rows.append(
                (
                    distance,
                    better,
                )
            )

    if not rows:
        raise RuntimeError(
            "No usable rows found in the CSV."
        )

    distances = np.asarray(
        [row[0] for row in rows],
        dtype=np.float64,
    )
    better = np.asarray(
        [row[1] for row in rows],
        dtype=bool,
    )

    thresholds = sorted(
        set(
            float(value)
            for value in args.thresholds
        )
    )

    results: list[dict[str, float | int]] = []

    total = len(rows)

    print(
        "\n=== Pelvis reliability by pelvis-feet X distance ==="
    )
    print(
        f"Usable matched detections: {total}"
    )
    print()
    print(
        f"{'Threshold':>12} "
        f"{'Cases':>10} "
        f"{'Coverage':>10} "
        f"{'Pelvis better':>15} "
        f"{'Pelvis worse':>14} "
        f"{'Success %':>10}"
    )
    print("-" * 78)

    for threshold in thresholds:
        mask = distances <= threshold

        count = int(
            np.sum(mask)
        )
        better_count = int(
            np.sum(
                better[mask]
            )
        )
        worse_count = (
            count
            - better_count
        )

        coverage = (
            count
            / total
            if total
            else 0.0
        )

        success_rate = (
            better_count
            / count
            if count
            else float("nan")
        )

        results.append({
            "threshold_px": threshold,
            "cases": count,
            "coverage": coverage,
            "pelvis_better": better_count,
            "pelvis_worse_or_equal": worse_count,
            "success_rate": success_rate,
        })

        success_text = (
            f"{success_rate * 100:.2f}%"
            if np.isfinite(success_rate)
            else "n/a"
        )

        print(
            f"{threshold:>10.2f}px "
            f"{count:>10d} "
            f"{coverage * 100:>9.2f}% "
            f"{better_count:>15d} "
            f"{worse_count:>14d} "
            f"{success_text:>10}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold_px",
                "cases",
                "coverage",
                "pelvis_better",
                "pelvis_worse_or_equal",
                "success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
