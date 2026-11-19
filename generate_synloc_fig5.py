#!/usr/bin/env python3
"""Plot Fig. 5 from the validation bootstrap_comparisons.csv export.

    python -m pip install matplotlib
    python generate_synloc_fig5.py 'bootstrap_comparisons(1).csv'

The CSV supplies the observed gains and paired bootstrap percentile intervals.
"""

import argparse
import csv
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


LABELS = {
    "bbox": "vs. bbox",
    "hybrid_a": "vs. Hybrid",
    "hard_fusion": "vs. hard fusion",
}
GAIN = "delta_percentage_points"
LOWER = "ci_lower_percentage_points"
UPPER = "ci_upper_percentage_points"
PNG_DPI = 600


def read_comparisons(path):
    comparisons = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = {"comparator", "proposed", GAIN, LOWER, UPPER}
        missing = sorted(columns - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        for row in reader:
            key = row["comparator"]
            if key not in LABELS or row["proposed"] != "continuous_fusion":
                continue
            if key in comparisons:
                raise ValueError(f"Duplicate comparison for {key!r}.")
            gain, lower, upper = (float(row[column]) for column in (GAIN, LOWER, UPPER))
            if not all(isfinite(value) for value in (gain, lower, upper)) or lower > upper:
                raise ValueError(f"Invalid gain or interval for {key!r}.")
            comparisons[key] = (gain, lower, upper)

    missing = LABELS.keys() - comparisons.keys()
    if missing:
        raise ValueError(f"Missing continuous-fusion comparisons: {', '.join(sorted(missing))}")
    return [comparisons[key] for key in LABELS]


def plot_figure(comparisons, output_dir):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"}):
        fig, ax = plt.subplots(figsize=(7.15, 3.25), layout="constrained")
        positions = list(reversed(range(len(comparisons))))
        gains, lower, upper = zip(*comparisons)
        interval_color = "#276e97"

        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e7eb", linewidth=0.7)
        ax.axvline(0, color="#69767f", linestyle="--", linewidth=1.0)
        ax.hlines(positions, lower, upper, color=interval_color, linewidth=1.6, zorder=3)
        for endpoints in (lower, upper):
            ax.plot(endpoints, positions, "|", color=interval_color, markersize=7, markeredgewidth=1.3, zorder=3)
        ax.plot(gains, positions, "o", color="#194d6c", markersize=5.4, zorder=4)

        for y, gain in zip(positions, gains):
            ax.annotate(f"{gain:+.3f} pp", (gain, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="#19252e",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

        minimum = min(0, *lower, *gains)
        maximum = max(0, *upper, *gains)
        span = maximum - minimum or 1
        ax.set_xlim(minimum - 0.06 * span, maximum + 0.10 * span)
        ax.set_ylim(-0.35, positions[0] + 0.45)
        ax.set_yticks(positions, list(LABELS.values()))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlabel("Improvement in dataset-level mAP-LocSim (percentage points)", labelpad=7)
        ax.tick_params(direction="out", length=3.5, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "fig5_bootstrap_ci"
        fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
        fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Validation bootstrap comparisons CSV")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        plot_figure(read_comparisons(args.csv), args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
