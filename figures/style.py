"""Shared style for the paper's figures.

Figures are drawn at the full-page width (6.5 in) with a 9 pt base, so text is at least 8 pt at the printed
size, and saved without Type 3 fonts. Models are columns in a fixed order; the three
comparison levels use one blue ordinal ramp (checked with a colour-vision validator: monotone lightness,
visible steps, light end >= 2:1 on white).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
FIG_W = 6.5

MODELS = ["GR6J", "HBV", "SACSMA"]
MODEL_LABEL = {"GR6J": "GR6J", "HBV": "HBV", "SACSMA": "SAC-SMA"}
LEVELS = ["parameter", "state", "control"]
LEVEL_LABEL = {"parameter": "Calibrated capacity", "state": "Simulated deficit", "control": "Storage-free deficit"}
LEVEL_COLOR = {"parameter": "#86b6ef", "state": "#2a78d6", "control": "#104281"}
SHARED_COLOR = "#c9c8c3"
BENCHMARKS = ["climate", "stocker", "zhao", "soil"]
BENCHMARK_LABEL = {"climate": "Climate", "stocker": "Stocker", "zhao": "Zhao", "soil": "Soil"}
INK, INK_2, GRID = "#1a1a1a", "#555555", "#e3e3e3"

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "mathtext.fontset": "custom", "mathtext.default": "it",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    """PNG at 300 dpi and PDF, both at FIG_W; prints the PDF's actual width."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    with open(OUT / f"{name}.pdf", "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[([^\]]*)\]", fh.read())
    print(f"{name}: {float(m.group(1).split()[2]) / 72:.2f} in wide")
    plt.close(fig)
