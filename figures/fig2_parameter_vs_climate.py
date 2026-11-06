"""Figure 2: calibrated capacity parameter against the climate-derived benchmark (Results 3.1).

Per model, log-log density of the optimum capacity (GR6J X1, HBV FC, SAC-SMA UZTWM + LZTWM) against the
climate-derived S_R,20 (satellite E, Gumbel 20-year; eligible catchments only), with the 1:1 line, the median
parameter in benchmark deciles, the limits of the sampling range, Spearman rho and the median parameter/benchmark ratio.
Pairs passing the KGE screen.

Input: data/rq2/parameter_benchmarks_screened_catchments.csv (src/rq2_parameter.py).
Usage: python figures/fig2_parameter_vs_climate.py [--benchmark stocker]
       (--benchmark stocker draws the SI version against Stocker et al. 2023, figS_parameter_vs_stocker)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.stats import spearmanr

from style import DATA, FIG_W, INK, INK_2, MODEL_LABEL, MODELS, panel_label, save

PRIOR = {"GR6J": (1, 2000), "HBV": (1, 2000), "SACSMA": (2, 1600)}  # capacity at the prior's corners
CAPACITY_NAME = {"GR6J": "X1", "HBV": "FC", "SACSMA": "UZTWM + LZTWM"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
X_LIM, Y_LIM = (20, 2000), (0.7, 3000)
X_LABEL = {"climate": "Climate-derived benchmark (mm)", "stocker": "Stocker et al. (2023) benchmark (mm)"}
OUT_NAME = {"climate": "fig2_parameter_vs_climate", "stocker": "figS_parameter_vs_stocker"}

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", choices=list(X_LABEL), default="climate")
BENCH = ap.parse_args().benchmark
if BENCH == "stocker":  # 76 Stocker values lie below 20 mm (6 climate values do; those stay off the axis)
    X_LIM = (2, 2000)
d = pd.read_csv(DATA / "rq2/parameter_benchmarks_screened_catchments.csv").dropna(subset=[BENCH, "capacity_opt"])

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.75), sharey=True)
fig.subplots_adjust(left=0.09, right=0.9, top=0.88, bottom=0.17, wspace=0.08)
norm = LogNorm(1, 60)
for i, (ax, model) in enumerate(zip(axes, MODELS)):
    g = d[d.model == model]
    x, y = g[BENCH].to_numpy(), g.capacity_opt.to_numpy()
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=(22, 26), extent=np.log10([*X_LIM, *Y_LIM]),
                   mincnt=1, cmap=SEQ, norm=norm, linewidths=0.2, edgecolors="face")
    ax.plot(X_LIM, X_LIM, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    for b in PRIOR[model]:
        ax.axhline(b, color=INK_2, lw=0.6, ls=":")
    edges = np.quantile(x, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    med = pd.DataFrame({"k": k, "x": x, "y": y}).groupby("k").median()
    ax.plot(med.x, med.y, color=INK, lw=1.4, marker="o", ms=3.2, mec="white", mew=0.6, zorder=4)
    rho = spearmanr(x, y).statistic
    ax.text(0.97, 0.16, f"ρ = {rho:.2f}".replace("-", "−") + f"\nmedian ratio {np.median(y / x):.1f}×\nn = {len(g):,}", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000] if BENCH == "climate" else [3, 10, 30, 100, 300, 1000]))
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(plain); ax.yaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"{MODEL_LABEL[model]} ({CAPACITY_NAME[model]})", fontweight="bold")
    if i == 1:
        ax.set_xlabel(X_LABEL[BENCH])
    if i == 0:
        ax.set_ylabel("Calibrated capacity (mm)")
    ax.text(1250, 800, "1:1", ha="left", va="top", fontsize=8, color=INK_2)
    if i == 0:
        ax.text(X_LIM[0] * 1.25, 1.12, "sampling-range limits", ha="left", va="bottom", fontsize=7.5, color=INK_2)
cax = fig.add_axes([0.915, 0.17, 0.014, 0.71])
cb = fig.colorbar(hb, cax=cax)
cb.set_label("Catchments per cell")
cb.set_ticks([1, 3, 10, 30, 60], labels=["1", "3", "10", "30", "60"])
cb.ax.yaxis.set_minor_formatter(NullFormatter())
cb.outline.set_linewidth(0.4)
save(fig, OUT_NAME[BENCH])
