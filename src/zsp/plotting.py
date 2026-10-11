"""Shared figure style, following Nature's figure specifications.

Arial at 5-7 pt, bold lowercase panel labels at 8 pt, no background grid, axis lines and
ticks only, a colour-blind-safe palette (Wong, Nat. Methods 8, 441; 2011), and widths of one
(89 mm) or two (183 mm) journal columns. Messages belong in the caption, not on the figure,
so panels carry letters instead of titles. Every figure is saved as an editable vector PDF
and a 300 dpi PNG for the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4  # inches per millimetre
SINGLE = 89 * MM  # one Nature column
ONE_HALF = 120 * MM
DOUBLE = 183 * MM  # two Nature columns

INK, INK_2, INK_3 = "#000000", "#333333", "#777777"
# Wong palette: distinct under every common colour-vision deficiency
BLUE, VERMILLION, GREEN, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"
GREY_LIGHT, GREY_MID = "#e6e6e6", "#bfbfbf"
SERIES = [BLUE, VERMILLION, GREEN, ORANGE, SKY]
LINE_LABELS = {"hepg2": "HepG2", "jurkat": "Jurkat", "hct116": "HCT116", "h1": "H1"}
MODEL_LABELS = {
    "mean_transfer": "Mean transfer",
    "weighted": "Weighted transfer",
    "norm_restored": "Norm-restored transfer",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderaxespad": 0.2,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "errorbar.capsize": 1.5,
            "patch.linewidth": 0.4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays editable
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    """Bold lowercase panel letter at the top left, outside the axes (Nature style)."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig, path: Path) -> None:
    """Write the vector PDF (for print) and a 300 dpi PNG (for the README)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
