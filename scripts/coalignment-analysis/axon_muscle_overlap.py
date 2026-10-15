"""
Axon-muscle overlap fraction per sample, and a dot plot comparing conditions.

    overlap (%) = 100 x area(axon mask AND muscle mask) / area(muscle mask)

Input layout:
    <input>/<condition>/<sample>/axon.tif          single-channel axon image
    <input>/<condition>/<sample>/muscle_mask.tif   binary mask, nonzero = muscle

Output (in --output, default ./results):
    overlap.csv                  one row per sample (condition, sample, overlap_pct)
    overlap_plot.png             dot plot

Usage:
    python axon_muscle_overlap.py <input> [--pixel-size <um/px>] [--order A B] [-o results]
    (pixel size read from the axon.tif resolution tags unless --pixel-size is given)
    python axon_muscle_overlap.py --csv results/overlap.csv [--order A B]   # re-plot only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import ndimage, stats
from skimage import filters, morphology

# Segmentation
MEDIAN_SIZE = 5               # px
GAUSSIAN_SIGMA = 0.8          # px
FRANGI_SIGMAS = range(1, 5)   # px
VESSEL_PCT = 99.0             # percentile of the Frangi response kept as axon
MIN_FRAGMENT_UM2 = 25.0       # smaller objects are removed

# Plot
SIZE_IN = (1.0, 1.3)
FONT = 5
LW = 0.35
SD_LW = 0.45
SD_CAPSIZE = 1.5
DOT_GREY = "#8C8C8C"
DOT_FACES = ["white", DOT_GREY]   # first condition open, second filled
DOT_SIZE = 4
DOT_LW = 0.5
DOT_JITTER = 0.12
MEAN_HALF_WIDTH = 0.25
MEAN_LW = 0.6


# ---------------------------
# ANALYSIS
# ---------------------------
def segment_axon(img, pixel_size_um):
    img = ndimage.median_filter(img.astype(np.float32), size=MEDIAN_SIZE)
    img = ndimage.gaussian_filter(img, sigma=GAUSSIAN_SIGMA)
    vmin, vmax = np.percentile(img, [1, 99.5])
    img = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
    vesselness = filters.frangi(img, sigmas=FRANGI_SIGMAS, black_ridges=False)
    mask = vesselness > np.percentile(vesselness, VESSEL_PCT)
    min_size_px = max(1, int(MIN_FRAGMENT_UM2 / pixel_size_um ** 2))
    return morphology.remove_small_objects(mask, min_size=min_size_px)


def tiff_pixel_size_um(path):
    """Pixel size in um from the TIFF XResolution / ResolutionUnit tags."""
    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        if "XResolution" not in tags:
            raise ValueError(f"{path}: no resolution in the TIFF metadata; give --pixel-size")
        num, den = tags["XResolution"].value
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
    um_per_unit = {2: 25400.0, 3: 10000.0}.get(int(unit))
    if um_per_unit is None or num == 0:
        raise ValueError(f"{path}: no usable resolution in the TIFF metadata; give --pixel-size")
    return um_per_unit * den / num


def overlap_pct(sample_dir, pixel_size_um=None):
    if pixel_size_um is None:
        pixel_size_um = tiff_pixel_size_um(sample_dir / "axon.tif")
    axon = segment_axon(tifffile.imread(sample_dir / "axon.tif"), pixel_size_um)
    muscle = tifffile.imread(sample_dir / "muscle_mask.tif") > 0
    if axon.shape != muscle.shape:
        raise ValueError(f"{sample_dir}: axon image and muscle mask differ in shape")
    return 100 * (axon & muscle).sum() / muscle.sum()


def analyse(input_dir, pixel_size_um):
    rows = []
    for condition_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        samples = sorted(p for p in condition_dir.iterdir() if p.is_dir())
        for i, sample_dir in enumerate(samples, start=1):
            rows.append({"condition": condition_dir.name, "sample": f"S{i}",
                         "overlap_pct": overlap_pct(sample_dir, pixel_size_um)})
    return pd.DataFrame(rows)


# ---------------------------
# PLOT
# ---------------------------
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def fit_margins(fig, ax, pad=0.02):
    """Fit the margins to the label extents so the figure stays exactly SIZE_IN."""
    w, h = SIZE_IN
    fig.subplots_adjust(left=0.3 / w, right=1 - 0.03 / w, bottom=0.3 / h, top=1 - 0.04 / h)
    for _ in range(3):
        tb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        pos = ax.get_position()
        left = pos.x0 * w - tb.x0 + pad
        bottom = pos.y0 * h - tb.y0 + pad
        top = tb.y1 - pos.y1 * h + pad
        right = max(tb.x1 - pos.x1 * w + pad, 0.03)
        fig.subplots_adjust(left=left / w, right=1 - right / w, bottom=bottom / h, top=1 - top / h)


def plot(df, order, out_stem):
    data = [df.loc[df.condition == c, "overlap_pct"].values for c in order]
    means = [d.mean() for d in data]
    sds = [d.std(ddof=1) for d in data]

    p = None
    if len(order) == 2:
        t, p = stats.ttest_ind(*data, equal_var=False)
        print(f"\nWelch's t-test {order[0]} vs {order[1]}: t={t:.3f}, p={p:.4f} "
              f"(n={len(data[0])} vs {len(data[1])})")

    with plt.rc_context({"font.family": ["Arial", "DejaVu Sans"], "font.size": FONT,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=SIZE_IN)
        rng = np.random.default_rng(42)
        for i, (d, m, sd) in enumerate(zip(data, means, sds)):
            ax.plot([i - MEAN_HALF_WIDTH, i + MEAN_HALF_WIDTH], [m, m], color="black",
                    lw=MEAN_LW, solid_capstyle="butt", zorder=3)
            ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="black", elinewidth=SD_LW,
                        capsize=SD_CAPSIZE, capthick=SD_LW, zorder=3)
            ax.scatter(i + rng.uniform(-DOT_JITTER, DOT_JITTER, len(d)), d, s=DOT_SIZE,
                       facecolors=DOT_FACES[i % 2], edgecolors=DOT_GREY, linewidths=DOT_LW, zorder=4)

        top = max(max(m + sd, d.max()) for m, sd, d in zip(means, sds, data))
        y_max = top * 1.35
        ax.set_ylim(0, y_max)
        if p is not None and p < 0.05:
            y_bar = top + y_max * 0.05
            ax.plot([0, 1], [y_bar, y_bar], color="black", lw=LW, solid_capstyle="butt")
            ax.text(0.5, y_bar + y_max * 0.02, stars(p), ha="center", va="bottom",
                    fontsize=FONT, fontweight="bold")

        ax.set_xticks(range(len(order)), order)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylabel("Axon–muscle overlap\nfraction (%)", labelpad=1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.tick_params(labelsize=FONT, width=LW, length=2, pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(LW)

        fit_margins(fig, ax)
        fig.savefig(out_stem.with_suffix(".png"), dpi=600)
        plt.close(fig)
    print(f"Saved -> {out_stem}.png")


# ---------------------------
# RUN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?", help="input folder (see layout above)")
    parser.add_argument("--pixel-size", type=float, help="pixel size in um/px (default: read from axon.tif)")
    parser.add_argument("--csv", type=Path, help="re-plot from an existing overlap.csv")
    parser.add_argument("--order", nargs="+", help="conditions left -> right (default: alphabetical)")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        out_dir = args.csv.parent
    elif args.input:
        df = analyse(args.input, args.pixel_size)
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "overlap.csv", index=False)
    else:
        parser.error("give either <input> or --csv <overlap.csv>")

    print(df.to_string(index=False))
    plot(df, args.order or sorted(df.condition.unique()), out_dir / "overlap_plot")


if __name__ == "__main__":
    main()
