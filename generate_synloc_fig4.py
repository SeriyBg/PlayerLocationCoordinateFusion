#!/usr/bin/env python3
"""Plot SynLoc validation sensitivity from the three evaluation CSV exports.

    python -m pip install matplotlib
    python generate_synloc_fig4.py 'threshold_comparison(1).csv' \
        'comparison(4).csv' 'comparison(5).csv'

Use validation exports for all three inputs. Scores use the 0–1 mAP scale.
"""

import argparse
import csv
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


METRIC = "mAP_LocSim_50_95"
HEIGHT = "height_threshold_px"
GATE = "area_gate_source_px2"
PROFILES = ("proposed", "linear_40_90")
HEIGHT_RANGE = (50, 90)
AREA_RANGE = (300, 1000)

# Validation references from the continuous-fusion evaluation summary.
HYBRID_A_MAP = 0.5627036263330816
CONTINUOUS_MAP = 0.604249210980587  # Linear 40–90 px Y and linear 200–700 px² X.

BLUE = "#16589b"
TEAL = "#117c66"
RUST = "#ba4a23"
INK = "#19252e"


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return list(reader)


def read_score(row, path):
    score = float(row[METRIC])
    if not 0 <= score <= 1:
        raise ValueError(f"{path} contains an invalid {METRIC}: {score}")
    hybrid_delta = row.get("delta_mAP_vs_hybrid_a")
    if hybrid_delta and not isclose(score - float(hybrid_delta), HYBRID_A_MAP, rel_tol=0, abs_tol=5e-5):
        raise ValueError(f"{path} does not match the validation Hybrid reference. Use the val export.")
    return score


def add_point(series, threshold, score, path):
    previous = series.setdefault(threshold, score)
    if not isclose(previous, score, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{path} gives a conflicting mAP-LocSim at threshold {threshold:g}.")


def series_xy(series, description):
    if not series:
        raise ValueError(f"No matching rows for {description}.")
    return tuple(zip(*sorted(series.items())))


def load_height_sweep(path):
    points = {}
    for row in read_csv(path, (HEIGHT, METRIC)):
        threshold = float(row[HEIGHT])
        if HEIGHT_RANGE[0] <= threshold <= HEIGHT_RANGE[1]:
            add_point(points, threshold, read_score(row, path), path)
    return series_xy(points, "the 50–90 px height sweep")


def load_area_sweeps(paths):
    points = {profile: {} for profile in PROFILES}
    for path in paths:
        for row in read_csv(path, ("strategy", "soft_y_profile", GATE, METRIC)):
            profile = row["soft_y_profile"]
            if row["strategy"] != "combined" or profile not in points:
                continue
            gate = float(row[GATE])
            if AREA_RANGE[0] <= gate <= AREA_RANGE[1]:
                add_point(points[profile], gate, read_score(row, path), path)
    return {profile: series_xy(points[profile], profile) for profile in PROFILES}


def style_axis(ax, title, xlabel, x_ticks, values, y_spacing):
    low, high = min(values), max(values)
    padding = max(0.002, 0.12 * (high - low))
    ax.set(xlabel=xlabel, ylabel="Validation mAP-LocSim",
           ylim=(low - padding, high + padding))
    ax.set_title(title, loc="left", fontweight="semibold", pad=10)
    ax.set_xticks(x_ticks)
    ax.yaxis.set_major_locator(MultipleLocator(y_spacing))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dbe2e8", linewidth=0.64)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#52606b")
        ax.spines[side].set_linewidth(0.72)
    ax.tick_params(direction="out", length=3, width=0.7, color="#52606b", pad=4)


def plot_figure(height, area, output_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "path",
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.15, 3.05), layout="constrained",
                                     gridspec_kw={"width_ratios": (1, 1.10)})
    fig.get_layout_engine().set(w_pad=0.09, h_pad=0.09, wspace=0.09)
    reference = dict(color=RUST, linewidth=1.4, linestyle=(0, (1.1, 2.2)), dash_capstyle="round")

    heights, height_scores = height
    ax_a.plot(heights, height_scores, color=BLUE, marker="o", linewidth=1.8, markersize=4.2,
              label="Y step sweep")
    ax_a.axhline(HYBRID_A_MAP, label=f"Hybrid ({HYBRID_A_MAP:.4f})", **reference)
    style_axis(ax_a, "(a)  Y step threshold", "Height threshold (px)", range(50, 91, 10),
               (*height_scores, HYBRID_A_MAP), y_spacing=0.005)
    ax_a.set_xlim(48, 92)
    ax_a.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="none", fontsize=8)

    stepped_gates, stepped_scores = area["proposed"]
    linear_gates, linear_scores = area["linear_40_90"]
    ax_b.plot(stepped_gates, stepped_scores, color=TEAL, marker="s", linewidth=1.6, markersize=3.5,
              linestyle="--", label="Stepped Y profile")
    ax_b.plot(linear_gates, linear_scores, color=BLUE, marker="o", linewidth=1.8, markersize=4.2,
              markerfacecolor="white", label="Linear Y profile (40–90 px)")
    ax_b.axhline(CONTINUOUS_MAP, label=f"Continuous fusion ({CONTINUOUS_MAP:.4f})", **reference)
    style_axis(ax_b, "(b)  Hard X gate", r"Area threshold (px$^2$)", range(300, 1001, 100),
               (*stepped_scores, *linear_scores, CONTINUOUS_MAP), y_spacing=0.005)
    ax_b.set_xlim(270, 1030)
    ax_b.legend(loc="lower left", frameon=True, framealpha=1, edgecolor="none", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig4_fusion_sensitivity"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("height_csv", type=Path, help="Validation threshold_comparison.csv")
    parser.add_argument("fusion_csv", type=Path, help="Validation fusion_experiments_val_all_comparison.csv")
    parser.add_argument("area_sweep_csv", type=Path, help="Validation area_sweep_400_1000_val_combined_comparison.csv")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    try:
        height = load_height_sweep(args.height_csv)
        area = load_area_sweeps((args.fusion_csv, args.area_sweep_csv))
        plot_figure(height, area, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
