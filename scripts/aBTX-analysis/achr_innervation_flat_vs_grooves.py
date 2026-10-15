"""
AChR clusters and innervated AChR clusters, Flat vs Grooves.

Input layout (one .nd2 stack per field of view):
    <input>/Flat/<replicate>/*.nd2
    <input>/Grooves/<replicate>/*.nd2

Usage:
    python achr_innervation_flat_vs_grooves.py --input <input> --out <out>   # analyse + plot
    python achr_innervation_flat_vs_grooves.py --csv <out>/achr_innervation_flat_vs_grooves.csv --out <out>

Outputs (<out>):
    achr_innervation_flat_vs_grooves.csv   one row per replicate: source data of the plots
    AChR_clusters.png, innervated_clusters.png

Requires: numpy, pandas, scipy, scikit-image, matplotlib, nd2
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import nd2
from scipy import ndimage as ndi
from scipy.stats import ttest_ind
from skimage import draw, filters, measure, morphology
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------
# PARAMETERS
# ---------------------------
CONDITIONS = ["Flat", "Grooves"]       # input folder names = labels, left -> right
CSV_NAME = "achr_innervation_flat_vs_grooves.csv"
N_WORKERS = 4

AChR_CH, AXON_CH = 1, 2                # 0-based channel indices (AChR, TUBB3)
PIXEL_UM = 1.33

# AChR
BG_RADIUS_UM = 20.0                    # background opening, larger than any cluster
DESTREAK_LEN_UM = 33.0                 # bright lines longer than this = muscle fibre
N_ANGLES = 12
SMOOTH_SIGMA_PX = 1.0
AChR_SMOOTH_Z_UM = 2.0
AChR_LOW_K, AChR_HIGH_K = 3.5, 10.0    # hysteresis, in robust noise SD
MIN_Z_SLICES = 2
MIN_MAJOR_UM, MAX_MAJOR_UM = 5.0, 15.0
MIN_AREA_UM2 = np.pi * (MIN_MAJOR_UM / 2) ** 2

# Axons
AXON_TOPHAT_UM = 5.0
AXON_LINE_UM = 12.0
AXON_LINE_LOW_K, AXON_LINE_HIGH_K = 2.5, 5.0
AXON_GATE_K = 1.0
AXON_LOW_K, AXON_HIGH_K = 3.0, 6.0
AXON_MIN_OBJECT_PX = 20

SLAB_UM = 15.0                         # counting slab thickness
MIN_OVERLAP_PX = 1                     # innervated: shared pixels at the in-focus slice

# Plot: 1.0 x 1.0 in, Arial 5 pt; Flat = open grey dot, Grooves = filled grey dot
PANEL_IN = (1.0, 1.0)
FS = 5
LW = 0.35
ERR_LW = 0.45
DOT_GREY = "#8C8C8C"
DOT_FACE = {"Flat": "white", "Grooves": DOT_GREY}


# ---------------------------
# IMAGE PROCESSING
# ---------------------------
def robust_z(x):
    med = np.median(x)
    return (x - med) / max(1.4826 * np.median(np.abs(x - med)), 1e-6)


def line_footprint(length, theta):
    h = length // 2
    dy, dx = np.sin(theta) * h, np.cos(theta) * h
    fp = np.zeros((2 * h + 1, 2 * h + 1), bool)
    rr, cc = draw.line(int(round(h - dy)), int(round(h - dx)), int(round(h + dy)), int(round(h + dx)))
    fp[rr, cc] = True
    return fp


def line_footprints(length_um):
    length = int(round(length_um / PIXEL_UM)) | 1
    return [line_footprint(length, t) for t in np.linspace(0, np.pi, N_ANGLES, endpoint=False)]


def destreak(stack):
    """Subtract elongated (fibre) structures, keeping compact clusters."""
    fps = line_footprints(DESTREAK_LEN_UM)
    out = np.empty(stack.shape, np.float32)
    for z, img in enumerate(stack.astype(np.float32)):
        smooth = ndi.gaussian_filter(img, SMOOTH_SIGMA_PX)
        out[z] = img - np.max([ndi.grey_opening(smooth, footprint=fp) for fp in fps], axis=0)
    return out


def zscore_stack(stack, sigma_z):
    """Per-slice background subtraction, 3D smoothing (sigma_z in slices),"""
    fp = morphology.disk(int(round(BG_RADIUS_UM / PIXEL_UM)))
    sub = np.empty(stack.shape, np.float32)
    for z, img in enumerate(stack.astype(np.float32)):
        bg = filters.gaussian(ndi.grey_opening(img[::2, ::2], footprint=fp[::2, ::2]), 4,
                              preserve_range=True)
        bg = ndi.zoom(bg, 2, order=1)[: img.shape[0], : img.shape[1]]
        sub[z] = img - bg
    sub = ndi.gaussian_filter(sub, (sigma_z, SMOOTH_SIGMA_PX, SMOOTH_SIGMA_PX))
    return np.stack([robust_z(s) for s in sub])


def segment_axons(axon_raw):
    tophat_fp = morphology.disk(int(round(AXON_TOPHAT_UM / PIXEL_UM)))
    kernels = [k / k.sum() for k in (fp.astype(np.float32) for fp in line_footprints(AXON_LINE_UM))]
    th = np.empty(axon_raw.shape, np.float32)
    lr = np.empty(axon_raw.shape, np.float32)
    for z, img in enumerate(axon_raw.astype(np.float32)):
        t = morphology.white_tophat(ndi.gaussian_filter(img, 0.7), tophat_fp)
        th[z] = robust_z(t)
        lr[z] = robust_z(np.max([ndi.correlate(t, k) for k in kernels], axis=0))
    lines = filters.apply_hysteresis_threshold(lr, AXON_LINE_LOW_K, AXON_LINE_HIGH_K) & (th > AXON_GATE_K)
    mask = lines | filters.apply_hysteresis_threshold(th, AXON_LOW_K, AXON_HIGH_K)
    for z in range(mask.shape[0]):
        mask[z] = morphology.remove_small_objects(mask[z], AXON_MIN_OBJECT_PX)
    return mask


def kept_clusters(labels, achr_raw, axon_mask):
    """In-focus slice, footprint size and innervation of every kept cluster."""
    rows = []
    h, w = labels.shape[1:]
    for p in measure.regionprops(labels):
        zz, yy, xx = p.coords.T
        zs = np.unique(zz)
        if len(zs) < MIN_Z_SLICES or yy.min() == 0 or xx.min() == 0 or yy.max() == h - 1 or xx.max() == w - 1:
            continue
        vals = achr_raw[zz, yy, xx].astype(float)
        zf = int(zs[np.argmax([vals[zz == z].sum() for z in zs])])
        at = zz == zf
        fp = np.zeros((h, w), np.uint8)
        fp[yy[at], xx[at]] = 1
        rp = measure.regionprops(fp)[0]
        if MIN_MAJOR_UM <= rp.axis_major_length * PIXEL_UM <= MAX_MAJOR_UM and rp.area * PIXEL_UM ** 2 >= MIN_AREA_UM2:
            rows.append((zf, axon_mask[zf, yy[at], xx[at]].sum() >= MIN_OVERLAP_PX))
    return pd.DataFrame(rows, columns=["z_focus", "innervated"])


def analyse_stack(path):
    """(counted AChR clusters, innervated clusters) of one stack."""
    with nd2.ND2File(path) as f:
        data = f.asarray()                                   # (z, c, y, x)
        z_step = float(f.voxel_size().z)
    achr_raw, axon_raw = data[:, AChR_CH], data[:, AXON_CH]

    zachr = zscore_stack(destreak(achr_raw), sigma_z=AChR_SMOOTH_Z_UM / z_step)
    labels = measure.label(filters.apply_hysteresis_threshold(zachr, AChR_LOW_K, AChR_HIGH_K), connectivity=3)
    cdf = kept_clusters(labels, achr_raw, segment_axons(axon_raw))

    nz = labels.shape[0]
    k = min(nz, max(1, int(round(SLAB_UM / z_step))))
    n_focus = np.bincount(cdf.z_focus.astype(int), minlength=nz)
    z0 = int(np.argmax(np.convolve(n_focus, np.ones(k, int), "valid")))
    slab = cdf[cdf.z_focus.between(z0, z0 + k - 1)]
    return len(slab), int(slab.innervated.sum())


def analyse(input_dir):
    tasks = []
    for condition in CONDITIONS:
        rep_dirs = sorted(d for d in glob.glob(os.path.join(input_dir, condition, "*")) if os.path.isdir(d))
        for i, rep_dir in enumerate(rep_dirs, 1):
            tasks += [(condition, f"R{i}", p) for p in sorted(glob.glob(os.path.join(rep_dir, "*.nd2")))]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        counts = list(ex.map(analyse_stack, [t[2] for t in tasks]))
    fov = pd.DataFrame([(c, r, n, ni) for (c, r, _), (n, ni) in zip(tasks, counts)],
                       columns=["condition", "replicate", "AChR_clusters", "innervated_clusters"])
    rep = fov.groupby(["condition", "replicate"], sort=False).agg(
        n_fov=("AChR_clusters", "size"),
        AChR_clusters=("AChR_clusters", "mean"),
        innervated_clusters=("innervated_clusters", "mean")).reset_index()
    return rep


# ---------------------------
# PLOTS
# ---------------------------
def dot_plot(rep, metric, ylabel, out_base):
    groups = [rep.loc[rep.condition == c, metric].values for c in CONDITIONS]
    means = [g.mean() for g in groups]
    sds = [g.std(ddof=1) for g in groups]
    t, p = ttest_ind(*groups, equal_var=False)
    print(f"\n{metric}")
    for c, g, m, s in zip(CONDITIONS, groups, means, sds):
        print(f"  {c}: n = {len(g)}, mean = {m:.2f}, SD = {s:.2f}")
    print(f"  Welch's t-test: t = {t:.3f}, p = {p:.4g}")

    rc = {"font.family": "Arial", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
          "axes.linewidth": LW, "xtick.major.width": LW, "ytick.major.width": LW,
          "xtick.major.size": 2, "ytick.major.size": 2,
          "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=PANEL_IN)
        rng = np.random.default_rng(42)
        for i, (c, g) in enumerate(zip(CONDITIONS, groups)):
            ax.plot([i - 0.25, i + 0.25], [means[i]] * 2, color="black", lw=0.6,
                    solid_capstyle="butt", zorder=3)
            ax.errorbar(i, means[i], yerr=sds[i], fmt="none", ecolor="black", elinewidth=ERR_LW,
                        capsize=1.5, capthick=ERR_LW, zorder=3)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=4, linewidths=0.5,
                       facecolor=DOT_FACE[c], edgecolor=DOT_GREY, zorder=4)
        top = max(max(m + s, g.max()) for m, s, g in zip(means, sds, groups))
        y_lim = top * 1.35
        ax.set_ylim(0, y_lim)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if p < 0.05:
            y_pos = top + y_lim * 0.05
            ax.plot([0, 1], [y_pos, y_pos], color="black", lw=LW, solid_capstyle="butt", zorder=5)
            ax.text(0.5, y_pos + y_lim * 0.02, "***" if p < 0.001 else "**" if p < 0.01 else "*",
                    ha="center", va="bottom", fontweight="bold")

        # fixed margins (inches): the figure is exactly PANEL_IN
        w, h = PANEL_IN
        fig.subplots_adjust(left=0.34 / w, right=1 - 0.05 / w, bottom=0.12 / h, top=1 - 0.04 / h)
        fig.savefig(out_base + ".png", dpi=600)
        plt.close(fig)


def plot_all(rep, out_dir):
    dot_plot(rep, "AChR_clusters", "Average Number of\nAChR Clusters", os.path.join(out_dir, "AChR_clusters"))
    dot_plot(rep, "innervated_clusters", "Average Number of\nInnervated Clusters",
             os.path.join(out_dir, "innervated_clusters"))


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
    plot_all(rep, a.out)
