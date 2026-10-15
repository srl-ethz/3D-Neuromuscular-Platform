#!/usr/bin/env python3
"""
Bending-angle figure from bending_traces.csv (made by extract_bending.py):
mean bending angle per group and peak bending amplitude per construct.

Usage
-----
    python plot_bending.py bending_traces.csv --fps FPS --out figure_bending
"""

import argparse
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

FIG_SIZE_IN = (2.6, 1.63)
FONT_PT = 5
LW = 0.35
EDGE = "0.2"
LINE = {"Control": "#4D4D4D", "Injury Grooves": "#a33a0b", "Injury Flat": "#e08a3c"}
DOT = {"Control": "#BDBDBD", "Injury Grooves": "#EBA966", "Injury Flat": "#EBA966"}
FILLED = {"Injury Flat": False}  # open dots; all other groups filled
LABEL = {"Control": "Ctr"}
RC = {
    "font.family": ["Arial", "Arial Unicode MS"], "font.size": FONT_PT, "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "text.color": "black",
    "axes.labelcolor": "black", "axes.edgecolor": EDGE, "axes.linewidth": LW,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": EDGE, "ytick.color": EDGE,
    "xtick.labelcolor": "black", "ytick.labelcolor": "black", "xtick.major.width": LW,
    "ytick.major.width": LW, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def peak_amplitude(y, stim, fps):
    """Rest-to-peak height (deg) of the average beat; noise floor if no beat."""
    if not stim == stim:  # no stimulation frequency: not beating
        return float(2 * np.sqrt(2) * np.std(y))
    period = fps / stim                     # frames per beat (not rounded)
    n_cyc = int(y.size // period)           # complete beats only
    frames = np.arange(y.size)
    phase = np.arange(100) / 100 * period
    cycle = np.mean([np.interp(c * period + phase, frames, y) for c in range(n_cyc)], axis=0)
    return float(np.ptp(cycle))


def holm(p):
    """Holm-adjusted p-values for a dict of raw p-values."""
    keys = sorted(p, key=p.get)
    adj, running = {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * p[k]))
        adj[k] = running
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def label(g):
    return LABEL.get(g, g)


def main():
    ap = argparse.ArgumentParser(description="Bending-angle figure from the traces CSV.")
    ap.add_argument("csv")
    ap.add_argument("--fps", type=float, required=True, help="Acquisition frame rate (frames/s).")
    ap.add_argument("--out", default="figure_bending", help="Output path without extension.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    groups = list(dict.fromkeys(df["group"]))
    traces = {g: [] for g in groups}
    amps = {g: [] for g in groups}
    for (g, _), d in df.groupby(["group", "construct"], sort=False):
        y = d["bending_deg"].to_numpy()
        traces[g].append(savgol_filter(y, 5, 3))
        amps[g].append(peak_amplitude(y, d["stim_hz"].iloc[0], args.fps))
    samples = [np.array(amps[g]) for g in groups]
    pairs = list(combinations(range(len(groups)), 2))
    tests = {(i, j): stats.ttest_ind(samples[i], samples[j], equal_var=False) for i, j in pairs}
    p_holm = holm({k: t.pvalue for k, t in tests.items()})

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=FIG_SIZE_IN)
    gs = fig.add_gridspec(len(groups), 2, width_ratios=[2.1, 1], wspace=0.34, hspace=0.14,
                          left=0.085, right=0.99, top=0.975, bottom=0.165)
    t = np.arange(len(traces[groups[0]][0])) / args.fps
    stacks = {g: np.vstack(traces[g]) for g in groups}
    ylim = max(np.max(np.abs(s.mean(0)) + s.std(0, ddof=1)) for s in stacks.values()) * 1.1
    axes = []
    for gi, g in enumerate(groups):
        ax = fig.add_subplot(gs[gi, 0], sharex=axes[0] if axes else None)
        mean, sd = stacks[g].mean(0), stacks[g].std(0, ddof=1)
        ax.fill_between(t, mean - sd, mean + sd, color=LINE[g], alpha=0.22, lw=0)
        ax.plot(t, mean, lw=0.6, color=LINE[g])
        ax.axhline(0, color="0.85", lw=0.35, zorder=0)
        ax.set_ylim(-ylim, 1.9 * ylim)  # headroom for the group name
        ax.set_yticks([v for v in (-2, 0, 2) if -ylim <= v <= ylim])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        ax.set_xlim(0, t[-1])
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.text(0.01, 0.93, label(g), transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PT,
                fontweight="bold")
        if gi < len(groups) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        axes.append(ax)
    fig.text(0.005, 0.555, "Bending angle (°)", rotation=90, ha="left", va="center", fontsize=FONT_PT)
    axes[-1].set_xlabel("Time (s)", labelpad=1)

    axb = fig.add_subplot(gs[:, 1])
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(groups, samples)):
        mean, sd = vals.mean(), vals.std(ddof=1)
        axb.plot([i - 0.25, i + 0.25], [mean, mean], color="black", lw=0.6, solid_capstyle="butt", zorder=3)
        axb.errorbar(i, mean, yerr=sd, fmt="none", ecolor="black", elinewidth=0.45, capsize=1.5,
                     capthick=0.45, zorder=3)
        axb.scatter(i + rng.uniform(-0.12, 0.12, vals.size), vals, s=4,
                    facecolor=DOT[g] if FILLED.get(g, True) else "white", edgecolor=DOT[g],
                    linewidths=0.5, zorder=4)
    axb.set_xticks(range(len(groups)))
    axb.set_xlim(-0.45, len(groups) - 0.55)
    axb.set_xticklabels([label(g).replace(" ", "\n") for g in groups], fontsize=FONT_PT)
    axb.set_ylabel("Peak bending amplitude (°)")
    top = max(max(v.max(), v.mean() + v.std(ddof=1)) for v in samples)
    step = 0.13 * top
    for level, (i, j) in enumerate(sorted(pairs, key=lambda ij: (ij[1] - ij[0], ij[0]))):
        y = top + step * (0.6 + level)
        mark = stars(p_holm[(i, j)])
        axb.plot([i, j], [y, y], color=EDGE, lw=LW, solid_capstyle="butt")
        axb.text((i + j) / 2, y + (0.0 if mark != "ns" else 0.01 * top), mark, ha="center", va="bottom",
                 fontsize=FONT_PT)
    axb.set_ylim(0, top + step * (len(pairs) + 0.4))
    for side, spine in axb.spines.items():
        spine.set_visible(side in ("left", "bottom"))

    fig.savefig(args.out + ".png", dpi=600)
    plt.close(fig)

    for g, v in zip(groups, samples):
        print(f"{g}: {v.mean():.2f} ± {v.std(ddof=1):.2f}° (mean ± SD, n={v.size}); "
              + ", ".join(f"{x:.2f}" for x in v))
    print("Welch's t-tests (two-tailed), Holm-corrected:")
    for (i, j), tt in tests.items():
        print(f"  {groups[i]} vs {groups[j]}: t({tt.df:.2f}) = {tt.statistic:.2f}, p = {tt.pvalue:.4f}, "
              f"p_Holm = {p_holm[(i, j)]:.4f} ({stars(p_holm[(i, j)])})")
    print(f"Saved {args.out}.png")


if __name__ == "__main__":
    main()
