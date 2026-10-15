#!/usr/bin/env python3
"""
Active Ca2+ area, -MN vs +MN, before and after glutamate: statistics and plot.

Input: CSV with columns condition (-MN | +MN), construct, timepoint (Pre | Post),
active_area_pct (from detect_active_area.py).

Usage
-----
    python plot_active_area.py --csv active_area.csv --out active_area
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORDER = ["-MN", "+MN"]
TIMEPOINTS = ["Pre", "Post"]
LABELS = {"Pre": "Pre-glutamate", "Post": "Post-glutamate"}
COLORS = {"-MN": "#8E959E", "+MN": "#0E7C7B"}
MEAN_COLOR = "#1A1A1A"
FS = 7                                   # Arial 7 pt
FIGSIZE = (2.8, 2.4)                     # inches
LW, ERR_LW = 0.35, 0.3                   # axis / error-bar line widths (pt)
MM_TO_PT = 72 / 25.4
MARKER_S = ((2 * 0.62 - 0.28) * MM_TO_PT) ** 2     # dots r = 0.62 mm, 0.28 mm stroke
MARKER_LW = 0.28 * MM_TO_PT
X = {("-MN", "Pre"): 0.0, ("-MN", "Post"): 1.0, ("+MN", "Pre"): 2.4, ("+MN", "Post"): 3.4}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def stats(d):
    lg = np.log(d.pivot_table(index=["condition", "construct"], columns="timepoint", values="active_area_pct"))
    rows = []
    for c in ORDER:
        r = ttest_rel(lg.loc[c, "Post"], lg.loc[c, "Pre"])
        rows.append(dict(comparison=f"{c} Pre vs Post", test="paired t", statistic=r.statistic,
                         p_value=r.pvalue, cells=((c, "Pre"), (c, "Post"))))
    for tp in TIMEPOINTS:
        r = ttest_ind(lg.loc[ORDER[0], tp], lg.loc[ORDER[1], tp], equal_var=False)
        rows.append(dict(comparison=f"{ORDER[0]} vs {ORDER[1]} {tp}", test="Welch t", statistic=r.statistic,
                         p_value=r.pvalue, cells=((ORDER[0], tp), (ORDER[1], tp))))
    st = pd.DataFrame(rows)
    st["p_holm"] = holm(st.p_value)
    fc = lg["Post"] - lg["Pre"]
    r = ttest_ind(fc.loc[ORDER[0]], fc.loc[ORDER[1]], equal_var=False)
    st.loc[len(st)] = dict(comparison=f"{ORDER[0]} vs {ORDER[1]} fold change (Post / Pre)",
                           test="Welch t on log fold change", statistic=r.statistic, p_value=r.pvalue,
                           cells=None, p_holm=np.nan)
    return st


def plot(d, st, out_base):
    L = np.log10
    rc = {"font.family": "Arial", "font.size": FS, "axes.linewidth": LW, "xtick.major.width": LW,
          "ytick.major.width": LW, "ytick.minor.width": LW * 0.8, "xtick.major.size": 2, "ytick.major.size": 2,
          "ytick.minor.size": 1.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2,
          "pdf.fonttype": 42}
    rng = np.random.default_rng(42)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        for c in ORDER:
            pos = {}
            for tp in TIMEPOINTS:
                sub = d[(d.condition == c) & (d.timepoint == tp)].sort_values("construct")
                v = L(sub.active_area_pct.values)
                x = X[(c, tp)]
                jit = x + rng.uniform(-0.12, 0.12, v.size)
                pos[tp] = dict(zip(sub.construct, zip(jit, v)))
                m, sd = v.mean(), v.std(ddof=1)
                ax.plot([x - 0.25, x + 0.25], [m, m], color=MEAN_COLOR, lw=0.6, solid_capstyle="butt", zorder=3)
                ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=MEAN_COLOR, elinewidth=ERR_LW * 1.5,
                            capsize=1.5, capthick=ERR_LW * 1.5, zorder=3)
                ax.scatter(jit, v, s=MARKER_S, linewidths=MARKER_LW, edgecolor=COLORS[c],
                           facecolor="white" if tp == "Pre" else COLORS[c], zorder=4)
        ax.set_xticks([X[(c, tp)] for c in ORDER for tp in TIMEPOINTS])
        ax.set_xticklabels(TIMEPOINTS * len(ORDER))
        for c in ORDER:
            ax.text((X[(c, "Pre")] + X[(c, "Post")]) / 2, -0.13, c.replace("-", "−"),
                    transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.set_xlim(-0.55, X[(ORDER[1], "Post")] + 0.55)
        lo, hi = L(d.active_area_pct.min()), L(d.active_area_pct.max())
        dec = range(int(np.floor(lo)), int(np.ceil(hi)) + 1)
        ticks = [v for v in (m * 10.0 ** k for k in dec for m in (1, 2, 5)) if lo - 0.1 <= L(v) <= hi + 0.35]
        if len(ticks) > 6:
            ticks = [v for v in ticks if float(L(v)).is_integer()]
        ax.set_yticks([L(v) for v in ticks])
        ax.set_yticklabels([f"{v:g}" for v in ticks])
        ax.set_yticks([k + L(m) for k in dec for m in range(2, 10)], minor=True)
        ax.set_ylabel("Active area (% of field)")
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
        step, level, bars = 0.22, -np.inf, []
        for cells, p in st.dropna(subset=["cells"])[["cells", "p_holm"]].itertuples(index=False):
            if p < 0.05:
                top = d[d.set_index(["condition", "timepoint"]).index.isin(cells)].active_area_pct.max()
                bars.append((X[cells[0]], X[cells[1]], p, L(top)))
        for x1, x2, p, top in sorted(bars, key=lambda b: abs(b[1] - b[0])):
            level = max(top + step * 0.6, level + step)
            ax.plot([x1, x2], [level, level], color="black", lw=LW, solid_capstyle="butt", zorder=5)
            ax.annotate("***" if p < 0.001 else "**" if p < 0.01 else "*", ((x1 + x2) / 2, level),
                        xytext=(0, -0.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS, fontweight="bold")
        ax.set_ylim(np.floor(lo) if lo - np.floor(lo) < 0.1 else lo - 0.1, max(hi, level) + step * 0.55)
        mk = dict(marker="o", linestyle="none", markersize=MARKER_S ** 0.5, markeredgewidth=MARKER_LW,
                  markeredgecolor=MEAN_COLOR)
        handles = [Line2D([], [], markerfacecolor="white", label=LABELS["Pre"], **mk),
                   Line2D([], [], markerfacecolor=MEAN_COLOR, label=LABELS["Post"], **mk),
                   Line2D([], [], color=MEAN_COLOR, lw=0.6, marker="|", markersize=3,
                          markeredgewidth=ERR_LW * 1.5, label="Mean ± SD")]
        fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                   handlelength=1.0, handletextpad=0.3, columnspacing=0.8, borderpad=0.2, borderaxespad=0.1)
        plt.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
        plt.savefig(out_base + ".png", dpi=600)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="active_area.csv")
    ap.add_argument("--out", default="active_area")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    st = stats(d)
    print(st.drop(columns="cells").round(4).to_string(index=False))
    plot(d, st, args.out)
    print(f"Saved {args.out}.png")


if __name__ == "__main__":
    main()
