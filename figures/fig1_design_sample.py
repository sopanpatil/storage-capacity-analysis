"""Figure 1: the sample, and the comparison design in one catchment (Data; Methods 3.4).

(a) The 6,144 screened catchments, coloured by aridity class (FAO-PM PET / P, Caravan attributes).
(b) Holland Creek at Kelfeera (camelsaus_404207; aridity 0.98, no snow), chosen because the three models land in
    the three classes. Per model: the behavioural P5-P95 range of the capacity parameter with the optimum, the
    model state (S_R,20 from the optimum's own fluxes) and the forcing-only control, on one capacity axis against
    the four benchmarks and their envelope. The class is from the main rule (src/classify.py).

Inputs: data/rq2/parameter_benchmarks_screened_catchments.csv, data/rq3/state_capacity_screened_catchments_summary.csv,
        data/rq3/control_screened_catchments.csv, data/classification/classification_screened_catchments.csv,
        data/catalog/catchments.csv, data/catalog/screened_catchments.csv; Natural Earth 1:110m land (cartopy).
Usage: python figures/fig1_design_sample.py
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import blended_transform_factory

from style import DATA, FIG_W, GRID, INK, INK_2, LEVEL_COLOR, MODEL_LABEL, MODELS, panel_label, save

EXAMPLE = "camelsaus_404207"
BENCH = {"climate": "Climate", "soil": "Soil", "stocker": "Stocker", "zhao": "Zhao"}
CLASS_LABEL = {"unidentifiable": "unidentifiable", "consistent": "identifiable, consistent",
               "inconsistent": "identifiable, inconsistent"}
CLASS_COLOR = {"unidentifiable": "#c9c8c3", "consistent": "#1baf7a", "inconsistent": "#eb6834"}
ENVELOPE = "#e6e5e0"
ARIDITY_BINS = [0, 0.5, 1, 2, np.inf]
ARIDITY_LABEL = ["< 0.5", "0.5–1", "1–2", "≥ 2"]
# Diverging about aridity 1 (energy- vs water-limited): teal arm for the wet classes, brown arm for the dry ones, light
# next to 1 and dark at the extremes (OKLCH L 0.42/0.74 teal, 0.78/0.46 brown). Each arm passes the colour-vision
# validator's ordinal checks on the land colour; the teal is kept away from the blues of the three quantities, and the
# worst separation from any panel (b) colour, under normal, protan or deutan vision, is dE 5.3 (tan vs the green class).
ARIDITY_COLOR = ["#085955", "#5fbdb7", "#d9af75", "#7a4c18"]

bench = pd.read_csv(DATA / "rq2/parameter_benchmarks_screened_catchments.csv").query("gauge_id == @EXAMPLE")
bench = bench.set_index("model")
state = pd.read_csv(DATA / "rq3/state_capacity_screened_catchments_summary.csv").query("gauge_id == @EXAMPLE")
control = pd.read_csv(DATA / "rq3/control_screened_catchments.csv").query("gauge_id == @EXAMPLE")
state, control = state.set_index("model"), control.set_index("model")
classes = pd.read_csv(DATA / "classification/classification_screened_catchments.csv").query("gauge_id == @EXAMPLE")
classes = classes.set_index("model")["class"]

fig = plt.figure(figsize=(FIG_W, 5.6))
ax = fig.add_axes([0.13, 0.085, 0.56, 0.38])

# (b) one catchment
b = bench.iloc[0]
lo, hi = b[list(BENCH)].min(), b[list(BENCH)].max()
rows = {"bench": 0}
y = 1.2
for m in MODELS:
    rows[m] = y
    y += 1.15
ax.axvspan(lo, hi, color=ENVELOPE, zorder=0, lw=0)
ax.text(np.sqrt(lo * hi), rows[MODELS[-1]] + 0.5, "benchmark\nrange", ha="center", va="bottom", fontsize=7.5,
        color=INK_2, linespacing=1.0)
order = sorted(BENCH, key=lambda k: b[k])
groups = [[order[0]]]
for k in order[1:]:  # one label for benchmarks within 15% of each other
    (groups[-1].append(k) if b[k] / b[groups[-1][-1]] < 1.15 else groups.append([k]))
for j, g in enumerate(groups):  # outer labels beside the diamonds, inner ones above
    for k in g:
        ax.plot(b[k], 0, marker="D", ms=5, color=INK_2, mec="white", mew=0.6, zorder=3)
    label = ", ".join(BENCH[k] for k in g)
    if j == 0:
        ax.text(min(b[k] for k in g) / 1.12, 0, label, ha="right", va="center", fontsize=7, color=INK_2)
    elif j == len(groups) - 1:
        ax.text(max(b[k] for k in g) * 1.12, 0, label, ha="left", va="center", fontsize=7, color=INK_2)
    else:
        ax.text(np.exp(np.mean(np.log([b[k] for k in g]))), -0.3, label, ha="center", va="bottom", fontsize=7,
                color=INK_2)
dy = {"parameter": -0.32, "state": 0.0, "control": 0.32}
for m in MODELS:
    r = bench.loc[m]
    yp = rows[m]
    ax.hlines(yp + dy["parameter"], r.beh_p5, r.beh_p95, color=LEVEL_COLOR["parameter"], lw=4, zorder=2,
              capstyle="butt")
    ax.plot(r.capacity_opt, yp + dy["parameter"], marker="|", ms=9, mew=2, color=INK, zorder=3)
    ax.plot(state.loc[m, "opt_sr_20"], yp + dy["state"], marker="o", ms=5.5, color=LEVEL_COLOR["state"],
            mec="white", mew=0.6, zorder=3)
    ax.plot(control.loc[m, "ctrl_sr_20"], yp + dy["control"], marker="s", ms=5, color=LEVEL_COLOR["control"],
            mec="white", mew=0.6, zorder=3)
    c = classes[m]
    ax.plot(1.04, yp, marker="s", ms=6, color=CLASS_COLOR[c], transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.065, yp, CLASS_LABEL[c], transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8)
ax.text(1.03, rows[MODELS[0]] - 0.75, "Class (main criterion)", transform=ax.get_yaxis_transform(), va="center",
        ha="left", fontsize=8, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(10, 3000)
ax.set_ylim(rows[MODELS[-1]] + 0.6, -0.7)
ax.set_yticks([0] + [rows[m] for m in MODELS], ["Benchmarks"] + [MODEL_LABEL[m] for m in MODELS])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Storage capacity (mm)")
ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
ax.set_title("Holland Creek at Kelfeera, Australia (aridity 0.98)", fontsize=9, loc="left")
panel_label(ax, "b", x=-0.16, y=1.04)
ax.legend(handles=[Line2D([], [], color=LEVEL_COLOR["parameter"], lw=4, label="Behavioural range (5–95%)"),
                   Line2D([], [], color=INK, marker="|", ms=9, mew=2, ls="", label="Calibrated optimum"),
                   Line2D([], [], color=LEVEL_COLOR["state"], marker="o", ms=5.5, mec="white", ls="",
                          label="Simulated deficit"),
                   Line2D([], [], color=LEVEL_COLOR["control"], marker="s", ms=5, mec="white", ls="",
                          label="Storage-free deficit")],
          loc="upper left", bbox_to_anchor=(1.02, 1.08), frameon=False, fontsize=7.5, handlelength=1.6,
          borderaxespad=0, labelspacing=0.35)

# (a) the sample
cat = pd.read_csv(DATA / "catalog/catchments.csv", usecols=["gauge_id", "gauge_lat", "gauge_lon", "aridity_FAO_PM"])
screened = pd.read_csv(DATA / "catalog/screened_catchments.csv", usecols=["gauge_id"])
s = screened.merge(cat, on="gauge_id", validate="1:1")
s["k"] = pd.cut(s.aridity_FAO_PM, ARIDITY_BINS, right=False, labels=False)
axm = fig.add_axes([0.0, 0.53, 1.0, 0.43], projection=ccrs.Robinson())
axm.set_extent([-180, 180, -56, 72], crs=ccrs.PlateCarree())
axm.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#ecebe7", edgecolor="none")
axm.spines["geo"].set_linewidth(0.4)
axm.spines["geo"].set_edgecolor("#bbbbbb")
for k in range(4):
    g = s[s.k == k]
    axm.scatter(g.gauge_lon, g.gauge_lat, s=2.2, color=ARIDITY_COLOR[k], lw=0, transform=ccrs.PlateCarree(),
                zorder=2 + k)
ex = cat.set_index("gauge_id").loc[EXAMPLE]
# the example: its own dot redrawn on top, inside a ring just larger than one dot
k_ex = int(pd.cut([ex.aridity_FAO_PM], ARIDITY_BINS, right=False, labels=False)[0])
axm.scatter(ex.gauge_lon, ex.gauge_lat, s=2.2, color=ARIDITY_COLOR[k_ex], lw=0, transform=ccrs.PlateCarree(), zorder=9)
axm.plot(ex.gauge_lon, ex.gauge_lat, marker="o", ms=4.2, mfc="none", mec=INK, mew=0.9,
         transform=ccrs.PlateCarree(), zorder=10)
axm.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=ARIDITY_COLOR[k],
                           label=f"{ARIDITY_LABEL[k]} ({(s.k == k).sum():,})") for k in range(4)]
                   + [Line2D([], [], marker="o", ls="", ms=4.2, mfc="none", mec=INK, mew=0.9, label="Example in (b)")],
           title="Aridity ($E_\\mathrm{p}/P$)", loc="lower left", bbox_to_anchor=(0.085, 0.05), frameon=False, fontsize=7.5,
           title_fontsize=7.5, handletextpad=0.2, alignment="left")
# "(a)" at the same figure x as "(b)" (whose axes start at 0.13, label at -0.16 of its 0.56 width)
axm.text(0.13 - 0.16 * 0.56, 1.0, "(a)", transform=blended_transform_factory(fig.transFigure, axm.transAxes),
         fontweight="bold", ha="right", va="bottom")

save(fig, "fig1_design_sample")
