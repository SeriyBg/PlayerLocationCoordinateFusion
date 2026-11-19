from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

OUT = Path.cwd()
BASE = "fig3_fusion_weights"
INK = "#19252e"
BLUE = "#16589b"
TEAL = "#117c66"
RUST = "#ba4a23"

HEIGHT_LIMIT = 160
AREA_LIMIT = 900
HEIGHT_RAMP = (40, 90)
AREA_RAMP = (200, 700)
HEIGHT_STEP = 70
AREA_GATE = 400
STEPPED_HEIGHTS = np.array([0, 40, 50, 60, 70, 140, HEIGHT_LIMIT])
STEPPED_WEIGHTS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])


def ramp(x, lower, upper):
    return np.clip((x - lower) / (upper - lower), 0, 1)


def style_axis(ax, title, ylabel, xlabel, limit, ticks):
    ax.set_title(title, loc="left", fontweight="semibold", pad=10)
    ax.set(xlim=(0, limit), ylim=(-0.075, 1.075), xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(ticks)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dbe2e8", linewidth=0.64)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#52606b")
        ax.spines[side].set_linewidth(0.72)
    ax.tick_params(direction="out", length=3.0, width=0.7, color="#52606b", pad=4)
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 6


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "mathtext.fontset": "stix",
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.labelcolor": INK,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 8.7,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "svg.fonttype": "path",
        "figure.facecolor": "white",
    })
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.15, 3.03), layout="constrained", gridspec_kw={"width_ratios": (1.12, 1)}
    )
    fig.get_layout_engine().set(w_pad=0.09, h_pad=0.09, wspace=0.075)

    proposed = dict(color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=3, label="Proposed linear")
    hard = dict(color=RUST, linewidth=1.55, linestyle=(0, (1.1, 2.2)), dash_capstyle="round", zorder=5)
    h = np.array([0, *HEIGHT_RAMP, HEIGHT_LIMIT])
    ax_a.plot(h, ramp(h, *HEIGHT_RAMP), **proposed)
    ax_a.step(STEPPED_HEIGHTS, STEPPED_WEIGHTS, where="post", color=TEAL, linewidth=1.55,
              linestyle=(0, (5.0, 2.6)), zorder=4, label="Stepped profile")
    ax_a.step([0, HEIGHT_STEP, HEIGHT_LIMIT], [0, 1, 1], where="post", label="70 px step", **hard)
    style_axis(ax_a, "(a)  Vertical weight", r"$\alpha(h)$", r"Bounding-box height $h$ (px)",
               HEIGHT_LIMIT, [0, 40, 50, 60, 70, 90, 140, HEIGHT_LIMIT])
    ax_a.tick_params(axis="x", labelsize=8.1)

    area = np.concatenate(([0], np.linspace(*AREA_RAMP, num=501), [AREA_LIMIT]))
    beta = ramp(area, *AREA_RAMP)
    beta_s = beta ** 2 * (3 - 2 * beta)

    ax_b.plot(area, beta, **proposed)
    ax_b.plot(
        area, beta_s, color=TEAL, linewidth=1.55,
        linestyle=(0, (5.0, 2.6)), zorder=4, label="Smoothstep",
    )
    ax_b.step(
        [0, AREA_GATE, AREA_LIMIT], [0, 1, 1],
        where="post", label=r"400 px$^2$ gate", **hard,
    )
    style_axis(ax_b, "(b)  Horizontal weight", r"$\beta(A)$", r"Bounding-box area $A$ (px$^2$)",
               AREA_LIMIT, [0, 200, 400, 700, AREA_LIMIT])

    for ax in (ax_a, ax_b):
        ax.legend(loc="lower right", bbox_to_anchor=(0.983, 0.125), borderaxespad=0.0,
                  frameon=True, framealpha=1, facecolor="white", edgecolor="none", fancybox=False,
                  borderpad=0.3, fontsize=8.0, handlelength=2.6, handletextpad=0.7, labelspacing=0.62)

    title = "Figure 3. Fusion weights as functions of player size"
    fig.savefig(OUT / f"{BASE}.svg", metadata={"Title": title, "Date": None})
    fig.savefig(OUT / f"{BASE}.png", dpi=600, metadata={"Title": title})
    plt.close(fig)


if __name__ == "__main__":
    main()
