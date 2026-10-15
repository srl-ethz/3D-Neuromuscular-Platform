#!/usr/bin/env python3
"""
Mean ± SD of the absolute muscle displacement across constructs, glutamate
(left) and curare (right), from the CSV written by extract_displacement.py.

Usage
-----
    python plot_displacement.py --fps FPS --csv contraction_traces.csv --out displacement_abs
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

CONDITIONS = ("Glutamate", "Curare")
SMOOTH_FRAMES = 7
LABELS = {"Glutamate": "+ 600 µM Glutamate", "Curare": "+ 100 µM curare"}
BG_COLORS = {"Glutamate": "#fa9f3a", "Curare": "#3d5a80"}
GAP_LABEL = "≈10 s\nequilibration time"
FIG_SIZE_IN = (3.6, 1.6)
MARGIN_IN = dict(left=0.32, right=0.05, bottom=0.28, top=0.12)
GAP_IN = 0.25
AXIS_LW, TRACE_LW = 0.35, 0.4
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 5, "axes.labelsize": 5, "xtick.labelsize": 5, "ytick.labelsize": 5,
    "axes.linewidth": AXIS_LW, "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "axes.labelpad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
}


def load(path):
    """{condition: {construct: displacement array}} from columns <condition>_<construct>."""
    cols = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            for name, v in r.items():
                if v:
                    cols[name].append(float(v))
    return {c: {name: np.array(v) for name, v in cols.items() if name.startswith(c.lower() + "_")}
            for c in CONDITIONS}


def mean_sd_abs(traces, fps):
    stack = np.abs(np.stack([savgol_filter(x, SMOOTH_FRAMES, 3) for x in traces.values()]))
    t = np.arange(stack.shape[1]) / fps
    return t, stack.mean(axis=0), stack.std(axis=0, ddof=1), len(traces)


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot mean ± SD |displacement| per condition.")
    p.add_argument("--fps", type=float, required=True, help="Frame rate (frames/s).")
    p.add_argument("--csv", default="contraction_traces.csv")
    p.add_argument("--out", default="displacement_abs", help="Output path without extension.")
    args = p.parse_args(argv)
    matplotlib.rcParams.update(RC)

    data = load(args.csv)
    curves = {c: mean_sd_abs(data[c], args.fps) for c in CONDITIONS}
    durations = [curves[c][0][-1] + 1 / args.fps for c in CONDITIONS]
    axes_w = FIG_SIZE_IN[0] - MARGIN_IN["left"] - MARGIN_IN["right"] - GAP_IN
    wspace = GAP_IN / (axes_w / 2)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_IN, sharey=True,
                             gridspec_kw={"width_ratios": durations, "wspace": wspace})
    for ax, cond, dur in zip(axes, CONDITIONS, durations):
        t, m, s, _ = curves[cond]
        ax.axvspan(0, dur, color=BG_COLORS[cond], alpha=0.14, lw=0, zorder=0)
        ax.axhline(0, color="0.88", lw=0.4, zorder=0)
        ax.fill_between(t, m - s, m + s, color="0.55", alpha=0.45, lw=0, zorder=1)
        ax.plot(t, m, lw=TRACE_LW, color="black", zorder=2, solid_capstyle="round")
        ax.set_xlim(0, dur)
        ax.annotate(LABELS[cond], xy=(0, 1.02), xycoords="axes fraction", xytext=(3, 0),
                    textcoords="offset points", va="bottom", ha="left", color="0.25")

    left, right = axes
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(left=False, labelleft=False)
    d, kw = 0.02, dict(color="0.25", clip_on=False, lw=AXIS_LW)
    for y0 in (0, 1):
        left.plot((1 - d, 1 + d), (y0 - d, y0 + d), transform=left.transAxes, **kw)
        right.plot((-d, d), (y0 - d, y0 + d), transform=right.transAxes, **kw)
    right.annotate(GAP_LABEL, xy=(0, 0.5), xycoords="axes fraction", xytext=(-GAP_IN * 36, 0),
                   textcoords="offset points", clip_on=False, ha="center", va="center",
                   rotation=90, color="0.25")
    left.annotate(f"mean ± SD, n = {curves['Glutamate'][3]}", xy=(1, 1.02),
                  xycoords="axes fraction", va="bottom", ha="right", color="0.25")
    left.set_ylabel("|Muscle displacement| (µm)")
    fig.supxlabel("Time (s)", y=0.01, fontsize=5)
    w, h = FIG_SIZE_IN
    fig.subplots_adjust(left=MARGIN_IN["left"] / w, right=1 - MARGIN_IN["right"] / w,
                        bottom=MARGIN_IN["bottom"] / h, top=1 - MARGIN_IN["top"] / h, wspace=wspace)
    fig.savefig(args.out + ".png", dpi=300)


if __name__ == "__main__":
    main()
