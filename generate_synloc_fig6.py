#!/usr/bin/env python3
"""Render Fig. 6 from SynLoc validation frames and the worst-localizations CSV.

    python -m pip install matplotlib pillow
    python generate_synloc_fig6.py /path/to/spiideo/images/val \
        'worst_localizations(2).csv'

The default failure indices are zero-based data rows, excluding the header.
Use the validation images at the source resolution recorded in the CSV.
"""

import argparse
import csv
from dataclasses import dataclass
from math import ceil, floor, isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import patheffects, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


FAILURE_INDICES = (27, 42, 100)
Y_TRANSITION = (40.0, 90.0)  # Source-image box height, px.
X_TRANSITION = (200.0, 700.0)  # Source-image box area, px².
CROP_ASPECT = 1.28  # Width / height.
FULL_HD = (1920, 1080)
INK = "#152022"
BOX_COLOR = "#6ee7ff"
MARKERS = {
    "BBox point": dict(marker="s", ms=8.5, mfc="none", mec=BOX_COLOR, mew=1.35, zorder=5),
    "Mask point": dict(marker="o", ms=11, mfc="none", mec="#ffbe51", mew=1.35, zorder=6),
    "Fused point": dict(marker="D", ms=5.1, mfc="#bcff72", mec=INK, mew=0.6, zorder=7),
    "Ground truth": dict(marker="x", ms=6.0, mfc="none", mec="#ff80c5", mew=1.35, zorder=8),
}
BOX_COLS = ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
SAM_COLS = ("sam_x_px", "sam_y_px")
GT_COLS = ("gt_x_px", "gt_y_px")
CSV_COLS = {
    "file_name", "detection_index", "match_status", "image_width_px", "image_height_px",
    *BOX_COLS, *SAM_COLS, *GT_COLS,
}


@dataclass(frozen=True)
class Example:
    file_name: str
    detection_index: int
    image_size: tuple[int, int]
    box: tuple[float, float, float, float]
    sam: tuple[float, float] | None
    gt: tuple[float, float]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def bbox_point(self):
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2

    @property
    def fused_point(self):
        u_b, v_b = self.bbox_point
        if self.sam is None:
            return u_b, v_b
        u_s, v_s = self.sam
        area = (self.box[2] - self.box[0]) * self.height
        alpha = clipped_weight(self.height, Y_TRANSITION)
        beta = clipped_weight(area, X_TRANSITION)
        return (1 - beta) * u_b + beta * u_s, (1 - alpha) * v_b + alpha * v_s


# From the full validation export adaptive_70px_per_detection_errors.csv.
# Rows 5492 and 56730 reach the SAM point under the final rule. Row 20755
# takes SAM X and bbox Y. The export therefore contains the exact pitch
# error for the final point in all three cases, each below 0.25 m.
SUCCESS_CASES = (
    Example(
        "000358.jpg", 1, FULL_HD,  # GT annotation 5733. 0.412 to 0.026 m.
        (546.32666015625, 674.8709106445312, 599.5916748046875, 770.1637573242188),
        (593.0, 764.0), (594.1673889160156, 764.3748168945312),
    ),
    Example(
        "003681.jpg", 5, FULL_HD,  # GT annotation 59365. 0.300 to 0.079 m.
        (1399.4822998046875, 661.1990966796875, 1452.0997314453125, 802.7430419921875),
        (1431.0, 793.0), (1433.5597229003906, 789.8450317382812),
    ),
    Example(
        "001337.jpg", 16, FULL_HD,  # GT annotation 21667. 0.531 to 0.111 m.
        (645.34228515625, 492.61541748046875, 674.6102294921875, 528.9750366210938),
        (669.0, 521.0), (668.9379272460938, 528.6969442367554),
    ),
)


def clipped_weight(value, transition):
    start, end = transition
    return min(1.0, max(0.0, (value - start) / (end - start)))


def load_failures(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(CSV_COLS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)

    failures = []
    for index in FAILURE_INDICES:
        if not 0 <= index < len(rows):
            raise ValueError(f"Record index {index} is outside the {len(rows)} data rows in {path}.")
        row = rows[index]
        if row["match_status"] != "matched":
            raise ValueError(f"Record {index} needs a matched ground-truth point.")

        box = tuple(float(row[col]) for col in BOX_COLS)
        sam = tuple(float(row[col]) for col in SAM_COLS) if all(row[col] for col in SAM_COLS) else None
        gt = tuple(float(row[col]) for col in GT_COLS)
        image_size = (int(row["image_width_px"]), int(row["image_height_px"]))
        valid_box = box[0] < box[2] and box[1] < box[3]
        if min(image_size) <= 0 or not valid_box or not all(map(isfinite, (*box, *gt, *(sam or ())))):
            raise ValueError(f"Invalid box, dimensions or coordinates in record {index}.")
        failures.append(Example(row["file_name"], int(row["detection_index"]), image_size, box, sam, gt))
    return failures


def crop_bounds(example, points):
    """Return (left, top, right, bottom) in source-image coordinates."""
    x1, y1, x2, y2 = example.box
    xs, ys = zip((x1, y1), (x2, y2), *points)
    pad = max(6.0, 0.20 * example.height)
    width, height = example.image_size
    desired_height = max((max(xs) - min(xs) + 2 * pad) / CROP_ASPECT, max(ys) - min(ys) + 2 * pad)
    crop_height = min(desired_height, height, width / CROP_ASPECT)
    crop_width = crop_height * CROP_ASPECT
    left = (min(xs) + max(xs) - crop_width) / 2
    top = (min(ys) + max(ys) - crop_height) / 2
    left = min(max(left, -0.5), width - 0.5 - crop_width)
    top = min(max(top, -0.5), height - 0.5 - crop_height)
    return left, top, left + crop_width, top + crop_height


def draw_example(ax, images_dir, example, label, show_vertical_error):
    fused = example.fused_point
    points = (example.bbox_point, example.sam, fused, example.gt)
    left, top, right, bottom = crop_bounds(example, [p for p in points if p is not None])

    image_path = images_dir / example.file_name
    with Image.open(image_path) as image:
        if image.size != example.image_size:
            raise ValueError(f"{image_path} is {image.size}, but the diagnostic coordinates use "
                             f"{example.image_size}. Use the frames used for inference.")
        x0, y0 = max(0, floor(left)), max(0, floor(top))
        x_end = min(image.width, ceil(right) + 1)
        y_end = min(image.height, ceil(bottom) + 1)
        pixels = image.crop((x0, y0, x_end, y_end)).convert("RGB")

    extent = (x0 - 0.5, x_end - 0.5, y_end - 0.5, y0 - 0.5)
    ax.imshow(pixels, extent=extent, interpolation="nearest", zorder=0)
    x1, y1, x2, y2 = example.box
    outline = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=BOX_COLOR, linewidth=1.0, zorder=4)
    outline.set_path_effects([patheffects.withStroke(linewidth=1.9, foreground=INK)])
    ax.add_patch(outline)

    for point, style in zip(points, MARKERS.values()):
        if point is not None:
            ax.plot(*point, linestyle="none", **style,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground=INK)])

    subtitle = rf"({label})   $h={example.height:.1f}$ px"
    if show_vertical_error:
        vertical_error = abs(fused[1] - example.gt[1])
        subtitle += rf"     $|\Delta v|={vertical_error:.1f}$ px"
    ax.set_title(subtitle, loc="left", fontsize=8.6, color=INK, pad=4)
    ax.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    ax.set_axis_off()


def plot_figure(images_dir, failures, output_dir):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path"}):
        fig = plt.figure(figsize=(7.15, 4.95), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.045, h_pad=0.045, wspace=0.04, hspace=0.025)
        grid = fig.add_gridspec(5, 3, height_ratios=(0.10, 1, 0.10, 1, 0.13))
        for row, (heading, examples) in enumerate((
            ("Successful localizations", SUCCESS_CASES), ("Vertical-error cases", failures),
        )):
            heading_ax = fig.add_subplot(grid[2 * row, :])
            heading_ax.set_axis_off()
            heading_ax.text(0, 0.4, heading, va="center", fontsize=10.6, weight="semibold", color=INK)
            for col, example in enumerate(examples):
                label = "abcdef"[row * 3 + col]
                ax = fig.add_subplot(grid[2 * row + 1, col])
                draw_example(ax, images_dir, example, label, show_vertical_error=(row == 1))
                print(f"({label}) {example.file_name}, detection {example.detection_index}")

        legend_ax = fig.add_subplot(grid[4, :])
        legend_ax.set_axis_off()
        handles = [
            Line2D([], [], linestyle="none", label=label, **style,
                   path_effects=[patheffects.withStroke(linewidth=2, foreground=INK)])
            for label, style in MARKERS.items()
        ]
        legend_ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                         handlelength=1.65, handletextpad=0.45, columnspacing=1.65)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "fig6_qualitative_localizations"
        fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
        fig.savefig(stem.with_suffix(".svg"), dpi=600, facecolor="white", metadata={"Date": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", type=Path, help="Directory containing the SynLoc validation images")
    parser.add_argument("worst_csv", type=Path, help="Validation worst_localizations.csv")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    try:
        failures = load_failures(args.worst_csv)
        plot_figure(args.images_dir, failures, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
