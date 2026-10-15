"""Every published panel, drawn from the CSV files in <csv-dir>.

Usage:
    python scripts/muscle-analysis/make_figures.py --data-dir <csv-dir> --out-dir <output>
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

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
HATCH_LW = 0.25

FONT_SIZE = 5


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
    """Return the font family to draw with, or abort if Arial is missing."""
    if register_arial():
        return "Arial"
    if os.environ.get("PREVIEW_FONT_FALLBACK") == "1":
        print("!! PREVIEW ONLY: Arial not found, rendering in Liberation Sans "
              "(metrically identical). Do not use this output in the paper.")
        return "Liberation Sans"
    raise SystemExit(
        "Arial not found. Install Arial, or set PREVIEW_FONT_FALLBACK=1 for a "
        "layout preview that must not be used in the paper."
    )


def base_rcparams(family):
    """Base rcParams: real Arial, type-42 (embedded TrueType) text in the PDF."""
    return {
        "font.family": family,
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": HATCH_LW,
    }


def frame(ax, color="black"):
    """Open frame: left and bottom spines only, at the shared hairline width."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(color)
        spine.set_linewidth(LW)


FONT_BOX_EXTRA_IN = dict(left=0.017, top=0.013, bottom=0.004, right=0.0)


def fit_axes(fig, axes, size_in, gap_in=0.3, pad=None):
    """Place panels so the drawn artwork, labels included, fills ``size_in``."""
    pad = FONT_BOX_EXTRA_IN if pad is None else pad
    width_in, height_in = size_in
    inv = fig.dpi_scale_trans.inverted()
    n = len(axes)

    for _ in range(3):
        renderer = fig.canvas.get_renderer()
        boxes = [(ax.get_tightbbox(renderer).transformed(inv),
                  ax.get_window_extent(renderer).transformed(inv)) for ax in axes]
        left_over = [axes_box.x0 - tight.x0 for tight, axes_box in boxes]
        right_over = [max(tight.x1 - axes_box.x1, 0) for tight, axes_box in boxes]
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


GROUP_KEYS = ["Flat", "Grooves"]
DISPLAY_GROUPS = ["Flat", "Grooves"]
CONDITION_PREFIX = {"F": "Flat", "G": "Grooves"}

INK = "black"
DOT_FACE = {"Flat": "none", "Grooves": "#808080"}
DOT_SIZE = 7
DOT_JITTER = 0.12
MEAN_HALF_WIDTH = 0.22

PAIR_FIGSIZE_IN = (3.0, 1.3)
SINGLE_FIGSIZE_IN = (1.5, 1.3)

BAND_GREY = "#d9d9d9"
FLAT_BAND = "#ebebeb"
FLAT_LINE = "#808080"

METRICS = {
    "fusion_index": ("fusion_index_pct", "Fusion index (%)"),
    "fiber_density": ("fiber_density_pct", "Fiber density (%)"),
    "alignment_index": ("alignment_index", "Alignment index"),
}

HIST_BIN_EDGES = np.linspace(-90, 90, 37)

DIST_BINS = [(b, b + 15) for b in range(0, 150, 15)]
DIST_CENTERS = np.array([(lo + hi) / 2 for lo, hi in DIST_BINS])


def _float(value):
    return float(value) if value not in ("", "nan", "NaN", None) else np.nan


def _condition(sample):
    """The condition a sample belongs to, from the first letter of its name."""
    try:
        return CONDITION_PREFIX[sample[:1].upper()]
    except KeyError:
        raise ValueError(f"sample {sample!r} does not start with one of "
                         f"{sorted(CONDITION_PREFIX)}") from None


def _rows(data_dir, stem):
    path = os.path.join(data_dir, f"{stem}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_metric(data_dir, stem, column):
    """``{condition: one value per sample}``."""
    grouped = defaultdict(list)
    for row in _rows(data_dir, stem):
        grouped[row["sample"]].append(_float(row[column]))

    out = {key: [] for key in GROUP_KEYS}
    for sample in sorted(grouped):
        out[_condition(sample)].append(float(np.nanmean(grouped[sample])))
    return out


def load_orientation_distribution(data_dir):
    """``{condition: array(n_curves, n_bins)}`` of per-replicate distributions."""
    rows = _rows(data_dir, "orientation_distribution")
    if rows is None:
        return None

    curves = defaultdict(dict)
    for row in rows:
        key = (row["sample"], int(row["replicate"]))
        curves[key][_float(row["deviation_deg"])] = _float(row["frequency_pct"])

    out = defaultdict(list)
    for (sample, _replicate), by_bin in sorted(curves.items()):
        out[_condition(sample)].append([by_bin[c] for c in sorted(by_bin)])
    return {key: np.array(values) for key, values in out.items()}


def load_wall_distance(data_dir):
    """Grooved curves and flat baseline, one value per sample."""
    profile = _rows(data_dir, "groove_alignment_profile")
    flat = _rows(data_dir, "flat_alignment")
    if profile is None or flat is None:
        return None, None

    by_replicate = defaultdict(dict)
    for row in profile:
        key = (row["sample"], int(row["replicate"]))
        by_replicate[key][_float(row["distance_um"])] = _float(row["alignment_index"])

    by_sample = defaultdict(list)
    for (sample, _replicate), by_bin in sorted(by_replicate.items()):
        by_sample[sample].append([by_bin[c] for c in sorted(by_bin)])
    with np.errstate(invalid="ignore"):
        curves = np.array([np.nanmean(np.array(by_sample[s]), axis=0)
                           for s in sorted(by_sample)])

    flat_by_sample = defaultdict(list)
    for row in flat:
        flat_by_sample[row["sample"]].append(_float(row["alignment_index"]))
    baseline = np.array([np.mean(flat_by_sample[s]) for s in sorted(flat_by_sample)])

    return curves, baseline


def draw_dotplot(ax, data_by_group, ylabel, ylim):
    """One replicate per dot, thick line at the mean, whiskers at +/- 1 SD."""
    data = [np.asarray(data_by_group[g], dtype=float) for g in GROUP_KEYS]
    means = [np.nanmean(d) for d in data]
    sds = [np.nanstd(d, ddof=1) for d in data]
    x_pos = np.arange(len(DISPLAY_GROUPS))

    t_stat, p_value = stats.ttest_ind(
        np.asarray(data_by_group["Grooves"], dtype=float),
        np.asarray(data_by_group["Flat"], dtype=float),
        equal_var=False, nan_policy="omit")

    rng = np.random.default_rng(0)
    for i, group in enumerate(GROUP_KEYS):
        d = data[i]
        ax.scatter(i + rng.uniform(-DOT_JITTER, DOT_JITTER, len(d)), d, s=DOT_SIZE,
                   facecolors=DOT_FACE[group], edgecolors=INK,
                   linewidths=BAR_LW, zorder=3)
        ax.plot([i - MEAN_HALF_WIDTH, i + MEAN_HALF_WIDTH], [means[i]] * 2,
                color=INK, lw=LW * 2, solid_capstyle="butt", zorder=4)
        ax.errorbar(i, means[i], yerr=sds[i], fmt="none", ecolor=INK,
                    elinewidth=ERR_LW, capsize=CAPSIZE * 1.5,
                    capthick=ERR_LW, zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xlim(-0.6, len(x_pos) - 0.4)
    ax.set_xticklabels(DISPLAY_GROUPS, fontsize=FONT_SIZE, color=INK)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, color=INK, labelpad=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONT_SIZE, colors=INK, width=LW, length=2, pad=1.5)
    plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE, color=INK)
    frame(ax, color=INK)
    ax.grid(False)
    ax.set_axisbelow(False)

    if p_value < 0.05:
        _significance_bracket(ax, means, sds, data, p_value)
    return t_stat, p_value


def _significance_bracket(ax, means, sds, data, p_value):
    """Flat bracket above the data with the usual star notation."""
    y_min, y_max = ax.get_ylim()
    data_max = max(max(m + s, np.nanmax(d)) for m, s, d in zip(means, sds, data))
    y_range = y_max - y_min
    y_pos = data_max + y_range * 0.06
    star_top = y_pos + y_range * 0.08
    if star_top > y_max:
        ax.set_ylim(y_min, star_top)
        y_max = star_top

    ax.plot([0, 1], [y_pos, y_pos], color=INK, linewidth=LW, solid_capstyle="butt")
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
    ax.text(0.5, y_pos + y_range * 0.02, stars, ha="center", va="bottom",
            fontsize=FONT_SIZE, fontweight="bold", color=INK)


def draw_orientation_histogram(ax, histograms):
    """Mean orientation-deviation distribution of each group."""
    centers = (HIST_BIN_EDGES[:-1] + HIST_BIN_EDGES[1:]) / 2
    width = HIST_BIN_EDGES[1] - HIST_BIN_EDGES[0]

    for group in ("Grooves", "Flat"):
        curves = histograms.get(group)
        if curves is None or len(curves) == 0:
            continue
        pct = np.mean(curves, axis=0)
        ax.bar(centers, pct, width=width, facecolor=DOT_FACE[group],
               edgecolor=INK, linewidth=BAR_LW,
               zorder=2 if group == "Grooves" else 3)

    ax.set_xlabel("Deviation from mean fiber angle (°)", fontsize=FONT_SIZE,
                  color=INK, labelpad=1)
    ax.set_ylabel("Frequency (%)", fontsize=FONT_SIZE, color=INK, labelpad=1)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, 0, 90])
    ax.tick_params(labelsize=FONT_SIZE, colors=INK, width=LW, length=2, pad=1.5)
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontsize=FONT_SIZE, color=INK)
    frame(ax, color=INK)
    ax.grid(False)


def figure_fusion_density(metrics, out_dir):
    """Fusion index and fiber density, side by side."""
    fig, (ax_fusion, ax_density) = plt.subplots(1, 2, figsize=PAIR_FIGSIZE_IN)

    draw_dotplot(ax_fusion, metrics["fusion_index"],
                 "Fusion index (%)", (0, 100))
    draw_dotplot(ax_density, metrics["fiber_density"],
                 "Fiber density (%)", (0, 60))

    fit_axes(fig, (ax_fusion, ax_density), PAIR_FIGSIZE_IN)
    _save(fig, out_dir, "fusion_density_combined")


def figure_alignment(metrics, histograms, out_dir):
    """Alignment index and the orientation-deviation distribution, side by side."""
    fig, (ax_dots, ax_hist) = plt.subplots(1, 2, figsize=PAIR_FIGSIZE_IN)

    draw_dotplot(ax_dots, metrics["alignment_index"],
                 "Alignment index", (0, 1))
    draw_orientation_histogram(ax_hist, histograms)

    fit_axes(fig, (ax_dots, ax_hist), PAIR_FIGSIZE_IN)
    _save(fig, out_dir, "alignment_index_combined")


def figure_wall_distance(grooved_curves, flat_values, out_dir):
    """Alignment to the groove axis against distance from the wall."""
    mean = np.nanmean(grooved_curves, axis=0)
    sd = np.nanstd(grooved_curves, axis=0, ddof=1)
    flat_mean, flat_sd = flat_values.mean(), flat_values.std(ddof=1)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN)

    ax.axhspan(flat_mean - flat_sd, flat_mean + flat_sd, color=FLAT_BAND,
               zorder=0, linewidth=0)
    ax.axhline(flat_mean, color=FLAT_LINE, lw=0.6, ls=(0, (2.2, 1.6)), zorder=1,
               dash_capstyle="butt")

    ax.fill_between(DIST_CENTERS, mean - sd, mean + sd, color=BAND_GREY,
                    alpha=0.9, linewidth=0, zorder=2)
    ax.plot(DIST_CENTERS, mean, "-", lw=0.6, color=INK, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    ax.plot(DIST_CENTERS, mean, "o", ms=1.0, color=INK, linestyle="none",
            markeredgewidth=0, zorder=4)

    ax.text(147, mean[-1] + sd[-1] + 0.035, "Grooved", fontsize=FONT_SIZE,
            color=INK, ha="right", va="bottom")
    ax.text(147, flat_mean - flat_sd - 0.035, "Flat", fontsize=FONT_SIZE,
            color=FLAT_LINE, ha="right", va="top")

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Distance from wall (µm)", fontsize=FONT_SIZE, color=INK, labelpad=1)
    ax.set_ylabel("Alignment index", fontsize=FONT_SIZE, color=INK, labelpad=1)
    ax.tick_params(labelsize=FONT_SIZE, colors=INK, width=LW, length=2, pad=1.5)
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontsize=FONT_SIZE, color=INK)
    frame(ax, color=INK)
    ax.grid(False)

    fit_axes(fig, (ax,), SINGLE_FIGSIZE_IN)
    _save(fig, out_dir, "alignment_vs_wall_distance")


def _save(fig, out_dir, basename):
    path = os.path.join(out_dir, f"{basename}.png")
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"  wrote {path}")


def report_statistics(metrics, curves, flat):
    """Print every number quoted in the figure captions and the results text."""
    print("\n=== GROUP COMPARISONS (Welch's t-test on sample means) ===")
    for stem, (_column, label) in METRICS.items():
        data = metrics[stem]
        g = np.asarray(data["Grooves"], dtype=float)
        f = np.asarray(data["Flat"], dtype=float)
        t_stat, p_value = stats.ttest_ind(g, f, equal_var=False)
        print(f"\n{label}")
        print(f"  Grooves: n={len(g)}  mean={g.mean():.3f}  SD={g.std(ddof=1):.3f}")
        print(f"  Flat:    n={len(f)}  mean={f.mean():.3f}  SD={f.std(ddof=1):.3f}")
        print(f"  t = {t_stat:+.4f}, p = {p_value:.4f}")

    if curves is None:
        return

    print("\n=== ALIGNMENT VS DISTANCE FROM WALL ===")
    mean = np.nanmean(curves, axis=0)
    sd = np.nanstd(curves, axis=0, ddof=1)
    print("  mean:", np.round(mean, 3))
    print("  SD:  ", np.round(sd, 3))
    slopes = [stats.linregress(DIST_CENTERS, c)[0] * 100 for c in curves]
    _, p_slope = stats.ttest_1samp(slopes, 0)
    print(f"  slope {np.mean(slopes):+.3f} +/- {np.std(slopes, ddof=1):.3f} "
          f"per 100 um, p = {p_slope:.2f} (n = {len(curves)} samples)")
    print(f"  flat baseline = {flat.mean():.3f} +/- {flat.std(ddof=1):.3f} "
          f"(n = {len(flat)} samples)")
    per_bin = [stats.ttest_ind(curves[:, i], flat, equal_var=False).pvalue
               for i in range(len(DIST_CENTERS))]
    print("  grooved vs flat, Welch p per bin:", np.round(per_bin, 4))
    print("  grooved mean-SD exceeds flat mean+SD at every distance: "
          f"{bool(((mean - sd) > (flat.mean() + flat.std(ddof=1))).all())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    plt.rcParams.update(base_rcparams(resolve_font_family()))
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {stem: load_metric(args.data_dir, stem, column)
               for stem, (column, _label) in METRICS.items()}
    histograms = load_orientation_distribution(args.data_dir)
    curves, flat = load_wall_distance(args.data_dir)

    print("Drawing figures:")
    figure_fusion_density(metrics, args.out_dir)
    if histograms is not None:
        figure_alignment(metrics, histograms, args.out_dir)
    else:
        print("  [SKIPPED] alignment_index_combined: "
              "orientation_distribution.csv not found")
    if curves is not None:
        figure_wall_distance(curves, flat, args.out_dir)
    else:
        print("  [SKIPPED] alignment_vs_wall_distance: "
              "groove_alignment_profile.csv / flat_alignment.csv not found")

    report_statistics(metrics, curves, flat)


if __name__ == "__main__":
    main()
