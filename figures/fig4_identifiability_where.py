"""Figure 4: identifiability under each rule, and where agreement and identifiability change (Results 3.3-3.4).

(a) Share of classified catchment-model pairs that are unidentifiable, identifiable and consistent, or identifiable
    and inconsistent, under the main rule and the four alternatives (src/classify.py).
(b) Spearman rho with the Stocker et al. benchmark (the one deficit-based benchmark covering every aridity class) of the calibrated
    parameter, the model state and the forcing-only control, by aridity class, with bootstrap 90% intervals.
(c) By aridity class: share of catchments whose optimum passes the KGE screen, and the shares of classified pairs
    that are unidentifiable and consistent (main rule).

Inputs: data/classification/classification_summary_screened_catchments.csv,
        data/rq4/profiles_screened_catchments.csv (src/rq4_where.py).
Usage: python figures/fig4_identifiability_where.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import (DATA, FIG_W, GRID, INK, INK_2, LEVEL_COLOR, LEVEL_LABEL, LEVELS, MODEL_LABEL, MODELS,
                   panel_label, save)

RULES = ["class", "class_ci", "class_nosoil", "class_fixed10", "class_halfprior"]
RULE_LABEL = {"class": "Main criterion", "class_ci": "Lower CI of width", "class_nosoil": "Range without soil",
              "class_fixed10": "Fixed 10-fold", "class_halfprior": "Half sampling range"}
CLASSES = ["unidentifiable", "consistent", "inconsistent"]
CLASS_LABEL = {"unidentifiable": "Unidentifiable", "consistent": "Identifiable, consistent",
               "inconsistent": "Identifiable, inconsistent"}
CLASS_COLOR = {"unidentifiable": "#c9c8c3", "consistent": "#1baf7a", "inconsistent": "#eb6834"}
ARIDITY_LABEL = ["<0.5", "0.5–1", "1–2", "≥2"]

summary = pd.read_csv(DATA / "classification/classification_summary_screened_catchments.csv")
summary = summary[summary.group == "all"].set_index(["model", "variant"])
profiles = pd.read_csv(DATA / "rq4/profiles_screened_catchments.csv")
profiles = profiles[profiles.covariate == "aridity"]

fig = plt.figure(figsize=(FIG_W, 7.0))
gs_a = fig.add_gridspec(1, 3, wspace=0.14, left=0.205, right=0.975, top=0.905, bottom=0.685)
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.08, left=0.105, right=0.975, top=0.585, bottom=0.075)

# (a) classification under each rule
axa = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
for i, (ax, model) in enumerate(zip(axa, MODELS)):
    for k, rule in enumerate(RULES):
        r = summary.loc[(model, rule)]
        total = r[CLASSES].sum()
        left = 0.0
        for c in CLASSES:
            w = r[c] / total
            ax.barh(k, w, left=left, height=0.7, color=CLASS_COLOR[c], edgecolor="white", linewidth=0.8)
            if c == "unidentifiable":
                # a sliver too narrow for its label gets it just left of the bar, so it is not read as the next class
                inside = w >= 0.1
                ax.text(0.02 if inside else -0.015, k, f"{100 * w:.0f}%", va="center", ha="left" if inside else "right",
                        fontsize=7.5, color=INK if inside else INK_2, clip_on=False)
            left += w
    ax.set_ylim(len(RULES) - 0.5, -0.5)
    ax.set_yticks(range(len(RULES)), [RULE_LABEL[r] for r in RULES] if i == 0 else [])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_title(MODEL_LABEL[model], fontweight="bold")
    if i == 1:
        ax.set_xlabel("Share of assessed pairs (%)")
panel_label(axa[0], "a", x=-0.66, y=1.06)
fig.legend(handles=[Patch(color=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASSES], loc="upper center",
           ncol=3, frameon=False, bbox_to_anchor=(0.59, 1.0), handlelength=1.2, handletextpad=0.4,
           columnspacing=1.4)

# (b) rank agreement along aridity
x = np.arange(4)
offset = {"parameter": -0.12, "state": 0.0, "control": 0.12}
axb = [fig.add_subplot(gs[0, i]) for i in range(3)]
for i, (ax, model) in enumerate(zip(axb, MODELS)):
    p = profiles[profiles.model == model]
    ax.axhline(0, color=INK_2, lw=0.6)
    for level in LEVELS:
        rho, lo, hi = (p[f"rho_{level}_stocker{s}"].to_numpy() for s in ("", "_lo", "_hi"))
        ax.vlines(x + offset[level], lo, hi, color=LEVEL_COLOR[level], lw=1.3, zorder=2)
        ax.plot(x + offset[level], rho, color=LEVEL_COLOR[level], lw=1.2, marker="o", ms=4, mec="white",
                mew=0.6, zorder=3)
    ax.set_xticks(x, ARIDITY_LABEL)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.5, 0.85)
    ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.grid(axis="y", color=GRID, lw=0.4)
    if i:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("Spearman ρ with\nStocker et al.")
panel_label(axb[0], "b", x=-0.25, y=1.1)
fig.legend(handles=[Line2D([], [], color=LEVEL_COLOR[l], marker="o", ms=4, mec="white", lw=1.2,
                           label=LEVEL_LABEL[l]) for l in LEVELS],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.588), handlelength=1.6,
           handletextpad=0.4, columnspacing=1.4)

# (c) screen survival and classes along aridity
axc = [fig.add_subplot(gs[1, i]) for i in range(3)]
# the pass share (of all catchments in the class) is context for the two class shares (of assessed pairs): thin, dashed,
# no markers, drawn underneath
lines = [("passes", "Passes KGE threshold", INK_2, (0, (3, 2)), 0.9, 0, 1),
         ("share_unidentifiable", "Unidentifiable", "#9b9a94", "-", 1.4, 4, 2),
         ("share_consistent", "Identifiable, consistent", CLASS_COLOR["consistent"], "-", 1.4, 4, 2)]
for i, (ax, model) in enumerate(zip(axc, MODELS)):
    p = profiles[profiles.model == model]
    for col, label, color, ls, lw, ms, z in lines:
        ax.plot(x, p[col], color=color, ls=ls, lw=lw, marker="o" if ms else None, ms=ms, mec="white", mew=0.6,
                label=label, zorder=z)
    for xi, n in zip(x, p.n_classified):
        ax.text(xi, -0.08, f"{n:,}", ha="center", va="top", fontsize=7, color=INK_2)
    if i == 0:
        ax.text(-0.62, -0.08, "n", ha="center", va="top", fontsize=7, color=INK_2, style="italic")
    ax.set_xticks(x, ARIDITY_LABEL)
    ax.tick_params(axis="x", pad=11)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    ax.grid(axis="y", color=GRID, lw=0.4)
    if i:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("Share (%)")
    if i == 1:
        ax.set_xlabel("Aridity ($E_\\mathrm{p}/P$)")
panel_label(axc[0], "c", x=-0.25, y=1.1)
handles, labels = axc[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.292),
           handlelength=1.8, handletextpad=0.4, columnspacing=1.4)

save(fig, "fig4_identifiability_where")
