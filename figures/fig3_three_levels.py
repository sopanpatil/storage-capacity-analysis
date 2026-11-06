"""Figure 3: agreement with the benchmarks at three levels, and what the model state is made of (Results 3.2).

(a) Spearman rank correlation across catchments between each benchmark and the calibrated capacity parameter, the
    model state (S_R,20 from the model's own fluxes) and the forcing-only control, per model, with bootstrap 90%
    intervals. Optimum parameter sets, pairs passing the KGE screen.
(b) Rank regression of the state on the parameter and the control: the state's explained rank variance (R^2) split
    into the part only the parameter explains, the part only the forcing (control) explains, and the shared part.

Inputs: data/rq3/state_vs_benchmarks_summary_screened_catchments.csv (src/rq3_benchmarks.py),
        data/rq3/coupling_screened_catchments.csv (src/rq3_coupling.py).
Usage: python figures/fig3_three_levels.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import (BENCHMARK_LABEL, BENCHMARKS, DATA, FIG_W, GRID, INK_2, LEVEL_COLOR, LEVEL_LABEL, LEVELS,
                   MODEL_LABEL, MODELS, SHARED_COLOR, panel_label, save)

summary = pd.read_csv(DATA / "rq3/state_vs_benchmarks_summary_screened_catchments.csv")
summary = summary[(summary.stratum == "all") & summary.benchmark.isin(BENCHMARKS)]
coupling = pd.read_csv(DATA / "rq3/coupling_screened_catchments.csv").set_index("model")

fig = plt.figure(figsize=(FIG_W, 4.0))
gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 0.55], hspace=0.62, wspace=0.08,
                      left=0.085, right=0.985, top=0.88, bottom=0.11)

# (a) three levels
offset = {"parameter": -0.22, "state": 0.0, "control": 0.22}
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for i, (ax, model) in enumerate(zip(axes, MODELS)):
    ax.axhline(0, color=INK_2, lw=0.6, zorder=1)
    for x in range(1, len(BENCHMARKS)):
        ax.axvline(x - 0.5, color=GRID, lw=0.6, zorder=0)
    for level in LEVELS:
        d = summary[(summary.level == level) & (summary.model == model)].set_index("benchmark").loc[BENCHMARKS]
        x = [j + offset[level] for j in range(len(BENCHMARKS))]
        ax.vlines(x, d.spearman_lo, d.spearman_hi, color=LEVEL_COLOR[level], lw=1.6, zorder=2)
        ax.scatter(x, d.spearman, s=26, color=LEVEL_COLOR[level], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(BENCHMARKS)), [BENCHMARK_LABEL[b] for b in BENCHMARKS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(BENCHMARKS) - 0.5)
    ax.set_ylim(-0.42, 0.82)
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.set_title(MODEL_LABEL[model], fontweight="bold")
    if i:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("Spearman ρ with benchmark")
panel_label(axes[0], "a", x=-0.2)
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=LEVEL_COLOR[l], mec="white",
                           label=LEVEL_LABEL[l]) for l in LEVELS],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.535, 1.0), handlelength=0.8, handletextpad=0.4,
           columnspacing=1.6)

# (b) decomposition of the state
parts = [("unique_r2_param", "Capacity only", LEVEL_COLOR["parameter"]),
         ("shared", "Shared", SHARED_COLOR),
         ("unique_r2_control", "Storage-free deficit only", LEVEL_COLOR["control"])]
coupling["shared"] = coupling.r2 - coupling.unique_r2_param - coupling.unique_r2_control
axb = [fig.add_subplot(gs[1, i]) for i in range(3)]
for i, (ax, model) in enumerate(zip(axb, MODELS)):
    left = 0.0
    for col, _, color in parts:
        w = coupling.loc[model, col]
        ax.barh(0, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0)
        if w >= 0.08:
            ax.text(left + w / 2, 0, f"{w:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if col == "unique_r2_control" else "#1a1a1a")
        left += w
    ax.text(0, 0.42, f"R² = {left:.2f}", va="bottom", ha="left", fontsize=8, color=INK_2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    if i == 1:
        ax.set_xlabel("Share of the rank variance of the simulated deficit explained")
panel_label(axb[0], "b", x=-0.2, y=1.3)
fig.legend(handles=[Patch(color=c, label=l) for _, l, c in parts], loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.535, 0.245), handlelength=1.2, handletextpad=0.4, columnspacing=1.6)

save(fig, "fig3_three_levels")
