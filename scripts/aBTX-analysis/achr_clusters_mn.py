"""
AChR cluster number, -MN vs +MN.

Input layout (one .nd2 stack per field of view):
    <input>/minusMN/<replicate>/*.nd2
    <input>/plusMN/<replicate>/*.nd2

Usage:
    python achr_clusters_mn.py --input <input> --out <out>    # analyse + plot
    python achr_clusters_mn.py --csv <out>/achr_clusters_mn.csv --out <out>   # plot only

Outputs (<out>):
    achr_clusters_mn.csv        one row per replicate: source data of the plot
    achr_clusters_mn.png        dot plot

Requires: numpy, pandas, scipy, scikit-image, matplotlib, nd2
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import nd2
from scipy.stats import ttest_ind
from skimage import measure, morphology
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# PARAMETERS
# ---------------------------
PIXEL_UM = 1.33                # um per pixel
AChR_CH = 1                    # 0-based channel index of the AChR (alpha-BTX) stain
TOPHAT_R_PX = 5
THRESHOLD_PERCENTILE = 99
MIN_MAJOR_UM, MAX_MAJOR_UM = 5.0, 15.0
MIN_AREA_PX = int(np.pi * (MIN_MAJOR_UM / 2 / PIXEL_UM) ** 2)

CONDITIONS = {"minusMN": "-MN", "plusMN": "+MN"}   # input folder -> label
ORDER = ["-MN", "+MN"]
CSV_NAME = "achr_clusters_mn.csv"

# Plot: 1.24 x 1.7 in, Arial 5 pt
FIG_IN = (1.24, 1.7)
FS = 5
LW = 0.35
ERR_LW = 0.45
DOT_COLORS = {"-MN": "#8E959E", "+MN": "#0E7C7B"}
MARKER_S = ((2 * 0.62 - 0.28) * 72 / 25.4) ** 2   # 1.24 mm outer diameter, 0.28 mm stroke
MARKER_LW = 0.28 * 72 / 25.4
MEAN_COLOR = "#1A1A1A"


# ---------------------------
# ANALYSIS
# ---------------------------
def rescale_255(img):
    img = img.astype(float)
    return (img - img.min()) / (img.max() - img.min() + 1e-9) * 255.0


def count_clusters(path):
    with nd2.ND2File(path) as f:
        mip = f.asarray()[:, AChR_CH].max(axis=0)          # (z, c, y, x) -> (y, x)
    tophat = rescale_255(morphology.white_tophat(rescale_255(mip), morphology.disk(TOPHAT_R_PX)))
    binary = tophat > np.percentile(tophat, THRESHOLD_PERCENTILE)
    binary = morphology.remove_small_objects(binary, MIN_AREA_PX)
    return sum(MIN_MAJOR_UM <= p.major_axis_length * PIXEL_UM <= MAX_MAJOR_UM
               for p in measure.regionprops(measure.label(binary)))


def analyse(input_dir):
    rows = []
    for folder, condition in CONDITIONS.items():
        rep_dirs = sorted(d for d in glob.glob(os.path.join(input_dir, folder, "*")) if os.path.isdir(d))
        for i, rep_dir in enumerate(rep_dirs, 1):
            counts = [count_clusters(p) for p in sorted(glob.glob(os.path.join(rep_dir, "*.nd2")))]
            rows.append(dict(condition=condition, replicate=f"R{i}", n_fov=len(counts),
                             AChR_clusters=np.mean(counts)))
            print(f"{condition} R{i}: {len(counts)} FOV, {np.mean(counts):.1f} clusters/FOV")
    return pd.DataFrame(rows)


# ---------------------------
# PLOT
# ---------------------------
def plot(rep, out_dir):
    groups = [rep.loc[rep.condition == c, "AChR_clusters"].values for c in ORDER]
    means = [g.mean() for g in groups]
    sds = [g.std(ddof=1) for g in groups]
    t, p = ttest_ind(*groups, equal_var=False)
    for c, g, m, s in zip(ORDER, groups, means, sds):
        print(f"{c}: n = {len(g)}, mean = {m:.1f}, SD = {s:.1f}")
    print(f"Welch's t-test: t = {t:.3f}, p = {p:.4g}")

    rc = {"font.family": "Arial", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
          "axes.linewidth": LW, "xtick.major.width": LW, "ytick.major.width": LW,
          "xtick.major.size": 2, "ytick.major.size": 2,
          "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=FIG_IN)
        rng = np.random.default_rng(42)
        for i, (c, g) in enumerate(zip(ORDER, groups)):
            ax.plot([i - 0.25, i + 0.25], [means[i]] * 2, color=MEAN_COLOR, lw=0.6,
                    solid_capstyle="butt", zorder=3)
            ax.errorbar(i, means[i], yerr=sds[i], fmt="none", ecolor=MEAN_COLOR, elinewidth=ERR_LW,
                        capsize=1.5, capthick=ERR_LW, zorder=3)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=MARKER_S, linewidths=MARKER_LW,
                       facecolor=DOT_COLORS[c], edgecolor=DOT_COLORS[c], zorder=4)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([c.replace("-", "−") for c in ORDER])
        ax.set_ylabel("Number of AChR Clusters")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        y_lim = max(max(m + s for m, s in zip(means, sds)), rep.AChR_clusters.max()) * 1.35
        ax.set_ylim(0, y_lim)
        if p < 0.05:
            y_bar = y_lim * 0.82
            ax.plot([0, 1], [y_bar, y_bar], color="black", lw=LW, solid_capstyle="butt")
            ax.text(0.5, y_bar + y_lim * 0.015, "***" if p < 0.001 else "**" if p < 0.01 else "*",
                    ha="center", va="bottom", fontweight="bold")
        plt.tight_layout(pad=0.3)
        base = os.path.join(out_dir, "achr_clusters_mn")
        fig.savefig(base + ".png", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="input folder (layout in the docstring)")
    src.add_argument("--csv", help="re-plot from a saved " + CSV_NAME)
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.csv:
        rep = pd.read_csv(a.csv)
    else:
        rep = analyse(a.input)
        rep.to_csv(os.path.join(a.out, CSV_NAME), index=False, float_format="%.4f")
    plot(rep, a.out)
