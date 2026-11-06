"""Supporting-information figures.

figS_spread_vs_envelope: per model, the behavioural spread of the capacity parameter, log10(P95/P5), against the
    width of the benchmark envelope, log10(max/min), for classifiable pairs. Above the 1:1 line the pair is
    unidentifiable under the main rule; the dotted line marks the fixed 10-fold rule and, for SAC-SMA, the
    dashed line the spread of its own prior (log10 0.72), which no behavioural spread can exceed by much.
figS_strata: number of screened catchments per aridity x snow-fraction cell (the analysis strata), with the
    share passing the KGE screen for at least one model.

Inputs: data/classification/classification_screened_catchments.csv, data/catalog/catchments.csv,
        data/catalog/screened_catchments.csv.
Usage: python figures/figS_si.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullFormatter

from style import DATA, FIG_W, INK, INK_2, MODEL_LABEL, MODELS, save

SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SACSMA_PRIOR_SPREAD = 0.72  # log10(P95/P5) of UZTWM + LZTWM under the prior

# spread vs envelope width
c = pd.read_csv(DATA / "classification/classification_screened_catchments.csv")
c = c[c["class"].isin(["unidentifiable", "consistent", "inconsistent"])]
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.6), sharey=True)
fig.subplots_adjust(left=0.115, right=0.9, top=0.88, bottom=0.18, wspace=0.08)
for i, (ax, m) in enumerate(zip(axes, MODELS)):
    g = c[c.model == m]
    hb = ax.hexbin(g.envelope_width, g.spread_abs, gridsize=(24, 24), extent=(0, 2.4, 0, 3.3), mincnt=1, cmap=SEQ,
                   norm=LogNorm(1, 100), linewidths=0.2, edgecolors="face")
    ax.plot([0, 2.4], [0, 2.4], color=INK_2, lw=0.8, ls=(0, (4, 3)))
    ax.axhline(1, color=INK_2, lw=0.6, ls=":")
    if m == "SACSMA":
        ax.axhline(SACSMA_PRIOR_SPREAD, color=INK, lw=0.8, ls="--")
        ax.text(2.35, SACSMA_PRIOR_SPREAD - 0.06, "width of sampling range", ha="right", va="top", fontsize=7.5, color=INK)
    share = (g["class"] == "unidentifiable").mean()
    ax.text(0.04, 0.96, f"above 1:1: {100 * share:.0f}%\nn = {len(g):,}", transform=ax.transAxes, va="top",
            fontsize=8, linespacing=1.3)
    ax.set_xlim(0, 2.4); ax.set_ylim(0, 3.3)
    ax.set_title(MODEL_LABEL[m], fontweight="bold")
    if i == 1:
        ax.set_xlabel("Width of benchmark range, log$_{10}$(max/min)")
    if i == 0:
        ax.set_ylabel("Width of behavioural range,\nlog$_{10}$($q_{95}$/$q_{5}$)")
        ax.text(2.35, 1.05, "10-fold", ha="right", va="bottom", fontsize=7.5, color=INK_2)
cax = fig.add_axes([0.915, 0.18, 0.014, 0.7])
cb = fig.colorbar(hb, cax=cax)
cb.set_ticks([1, 3, 10, 30, 100], labels=["1", "3", "10", "30", "100"])
cb.ax.yaxis.set_minor_formatter(NullFormatter())
cb.set_label("Pairs per cell")
cb.outline.set_linewidth(0.4)
save(fig, "figS_spread_vs_envelope")

# strata
cat = pd.read_csv(DATA / "catalog/catchments.csv", usecols=["gauge_id", "aridity_FAO_PM", "frac_snow"])
s = pd.read_csv(DATA / "catalog/screened_catchments.csv", usecols=["gauge_id"]).merge(cat, on="gauge_id")
calib = pd.read_csv(DATA / "calibration/calibration_summary.csv")
calib["passes"] = (calib.kge_cal >= 0.5) & (calib.kge_val >= 0.5)
s["passes_any"] = s.gauge_id.map(calib.groupby("gauge_id").passes.any()).fillna(False)
a_bins, s_bins = [0, 0.5, 1, 2, np.inf], [0, 0.05, 0.2, 0.4, 1.0001]
s["a"] = pd.cut(s.aridity_FAO_PM, a_bins, right=False, labels=False)
s["s"] = pd.cut(s.frac_snow, s_bins, right=False, labels=False)
n = s.pivot_table(index="s", columns="a", values="gauge_id", aggfunc="count").reindex(index=range(4), columns=range(4))
p = s.pivot_table(index="s", columns="a", values="passes_any", aggfunc="mean").reindex(index=range(4), columns=range(4))
fig, ax = plt.subplots(figsize=(4.2, 3.0))
fig.subplots_adjust(left=0.2, right=0.84, top=0.95, bottom=0.17)
im = ax.imshow(n.to_numpy(float), origin="lower", cmap=SEQ, norm=LogNorm(5, 3000), aspect="auto")
for (j, k), v in np.ndenumerate(n.to_numpy(float)):
    if np.isfinite(v):
        dark = v > 300
        ax.text(k, j, f"{int(v):,}\n{100 * p.iat[j, k]:.0f}%", ha="center", va="center", fontsize=7.5,
                color="white" if dark else INK, linespacing=1.15)
ax.set_xticks(range(4), ["<0.5", "0.5–1", "1–2", "≥2"])
ax.set_yticks(range(4), ["<0.05", "0.05–0.2", "0.2–0.4", "≥0.4"])
ax.set_xlabel("Aridity ($E_\\mathrm{p}/P$)")
ax.set_ylabel("Snow fraction")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
cb.set_ticks([10, 100, 1000], labels=["10", "100", "1,000"])
cb.ax.yaxis.set_minor_formatter(NullFormatter())
cb.set_label("Selected catchments")
cb.outline.set_linewidth(0.4)
save(fig, "figS_strata")
