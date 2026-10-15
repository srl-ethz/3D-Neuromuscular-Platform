"""The three-panel axon summary, drawn from the CSV files in <csv-dir>.

Usage:
    python scripts/neuron-analysis/make_figures.py --data-dir <csv-dir> --out-dir <output>
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy import stats


LW = 0.35
BAR_LW = 0.30
ERR_LW = 0.30
CAPSIZE = 1.2
FONT_SIZE = 5

FIGSIZE_IN = (3.0, 1.3)
PANEL_GAP_IN = 0.15

INK = "black"

GROUP_KEYS = ["Flat", "Grooves"]
CONDITION_PREFIX = {"I": "Flat", "G": "Grooves"}
DOT_FACE = {"Flat": "none", "Grooves": "#808080"}
DOT_SIZE = 7
DOT_JITTER = 0.12
MEAN_HALF_WIDTH = 0.22

TICK_LABELS = ["Isotropic", "Anisotropic"]

# Matplotlib's text boxes are slightly smaller than the glyph ink (descenders,
# side bearings), so leave a real margin or the edge labels get clipped.
FONT_BOX_EXTRA_IN = dict(left=0.03, top=0.03, bottom=0.03, right=0.03)


def register_arial():
    """Make genuine Arial available to matplotlib, wherever it is installed."""
    if any(f.name == "Arial" for f in fm.fontManager.ttflist):
        return True
    patterns = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        os.path.expanduser("~/Library/Fonts/Arial.ttf"),
        os.path.expanduser("~/.local/share/fonts/Arial.ttf"),
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/**/[Aa]rial.ttf",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            fm.fontManager.addfont(path)
            return True
    return False


def resolve_font_family():
    """Return the font to draw with, or abort rather than substitute silently."""
    if register_arial():
        return "Arial"
    if os.environ.get("PREVIEW_FONT_FALLBACK") == "1":
        print("!! PREVIEW ONLY: Arial not found, rendering in Liberation Sans "
              "(metrically identical). Do not use this output in the paper.")
        return "Liberation Sans"
    raise SystemExit(
        "Arial not found. Install Arial, or set PREVIEW_FONT_FALLBACK=1 for a "
        "layout preview that must not be used in the paper.")


def frame(ax):
    """Open frame: left and bottom spines only, at the shared hairline width."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(INK)
        spine.set_linewidth(LW)


def fit_panels(fig, axes, gap_in=PANEL_GAP_IN, pad=None):
    """Place the panels so the artwork, labels included, fills the figure size."""
    pad = FONT_BOX_EXTRA_IN if pad is None else pad
    width_in, height_in = fig.get_size_inches()
    inv = fig.dpi_scale_trans.inverted()
    n = len(axes)

    for _ in range(3):
        renderer = fig.canvas.get_renderer()
        boxes = [(ax.get_tightbbox(renderer).transformed(inv),
                  ax.get_window_extent(renderer).transformed(inv)) for ax in axes]
        left_over = [a.x0 - t.x0 for t, a in boxes]
        right_over = [max(t.x1 - a.x1, 0) for t, a in boxes]
        bottom = max(a.y0 - t.y0 for t, a in boxes) + pad["bottom"]
        top = max(t.y1 - a.y1 for t, a in boxes) + pad["top"]

        panel_w = (width_in - pad["left"] - pad["right"] - gap_in * (n - 1)
                   - sum(left_over) - sum(right_over)) / n

        x = pad["left"]
        for i, ax in enumerate(axes):
            x += left_over[i]
            ax.set_position([x / width_in, bottom / height_in,
                             panel_w / width_in,
                             (height_in - bottom - top) / height_in])
            x += panel_w + right_over[i] + gap_in


def _condition(sample):
    """The condition a sample belongs to, from the first letter of its name."""
    try:
        return CONDITION_PREFIX[sample[:1].upper()]
    except KeyError:
        raise ValueError(f"sample {sample!r} does not start with one of "
                         f"{sorted(CONDITION_PREFIX)}") from None


def _load(data_dir, filename, value_column):
    """``{condition: array of one value per sample}`` from a CSV."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run the analysis scripts first.")

    by_sample = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            by_sample.setdefault(row["sample"], []).append(float(row[value_column]))

    grouped = {key: [] for key in GROUP_KEYS}
    for sample in sorted(by_sample):
        grouped[_condition(sample)].append(float(np.mean(by_sample[sample])))
    return {key: np.asarray(values, dtype=float) for key, values in grouped.items()}


def load_all(data_dir):
    """The three panels' data, each one value per sample."""
    return [
        (_load(data_dir, "axon_alignment.csv", "alignment_index"),
         "Alignment index"),
        (_load(data_dir, "axon_area.csv", "area_mm2"),
         "Axon area (mm\u00b2)"),
        (_load(data_dir, "axon_length.csv", "median_length_mm"),
         "Axon length - Median (mm)"),
    ]


def draw_panel(ax, data, ylabel):
    """One dot per sample, thick line at the mean, whiskers at +/- 1 SD."""
    values = [data[key] for key in GROUP_KEYS]
    means = [v.mean() for v in values]
    sds = [v.std(ddof=1) for v in values]

    t_stat, p_value = stats.ttest_ind(data["Grooves"], data["Flat"], equal_var=False)

    rng = np.random.default_rng(0)
    for i, key in enumerate(GROUP_KEYS):
        v = values[i]
        ax.scatter(i + rng.uniform(-DOT_JITTER, DOT_JITTER, len(v)), v, s=DOT_SIZE,
                   facecolors=DOT_FACE[key], edgecolors=INK,
                   linewidths=BAR_LW, zorder=3)
        ax.plot([i - MEAN_HALF_WIDTH, i + MEAN_HALF_WIDTH], [means[i]] * 2,
                color=INK, lw=LW * 2, solid_capstyle="butt", zorder=4)
        ax.errorbar(i, means[i], yerr=sds[i], fmt="none", ecolor=INK,
                    elinewidth=ERR_LW, capsize=CAPSIZE * 1.5, capthick=ERR_LW,
                    zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticklabels(TICK_LABELS, fontsize=FONT_SIZE, color=INK)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, color=INK, labelpad=1)
    ax.tick_params(labelsize=FONT_SIZE, colors=INK, width=LW, length=2, pad=1.5)
    plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE, color=INK)
    frame(ax)

    data_top = max(max(v.max() for v in values),
                   max(m + s for m, s in zip(means, sds)))
    data_bottom = min(0, min(v.min() for v in values))
    y_range = data_top - data_bottom

    if p_value < 0.05:
        bracket_y = data_top + y_range * 0.10
        text_y = bracket_y + y_range * 0.03
        axis_top = text_y + y_range * 0.12
        ax.plot([0, 1], [bracket_y, bracket_y], color=INK, linewidth=LW,
                solid_capstyle="butt")
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.text(0.5, text_y, stars, ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold", color=INK)
    else:
        axis_top = data_top + y_range * 0.08

    ax.set_ylim(0, axis_top)
    ax.grid(False)
    ax.set_axisbelow(False)

    print(f"{ylabel}: Welch t = {t_stat:+.3f}, p = {p_value:.4g}  |  "
          f"Flat mean = {means[0]:.3f} SD = {sds[0]:.3f} (n = {len(values[0])})  |  "
          f"Grooves mean = {means[1]:.3f} SD = {sds[1]:.3f} (n = {len(values[1])})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": resolve_font_family(),
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    os.makedirs(args.out_dir, exist_ok=True)

    panels = load_all(args.data_dir)

    print("\n=== GROUP COMPARISONS (Welch's t-test, Grooves vs Flat) ===")
    fig, axes = plt.subplots(1, len(panels), figsize=FIGSIZE_IN)
    for ax, (data, ylabel) in zip(axes, panels):
        draw_panel(ax, data, ylabel)

    fit_panels(fig, axes)

    path = os.path.join(args.out_dir, "axon_summary_combined.png")
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
