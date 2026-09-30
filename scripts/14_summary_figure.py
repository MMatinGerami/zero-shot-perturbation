"""README summary figure: the best model against the two ceilings, per held-out line.

  noise ceiling     best centred gene-wise r any prediction could reach (scripts/10)
  transfer ceiling  noise ceiling x the noise-corrected HepG2/Jurkat agreement of the same
                    knockdown (scripts/12): what a perfect copy of another line's response
                    would reach. Approximate: a product of medians, from the one well-powered
                    pair, and on knockdowns reliable in both lines.
  best model        centred gene-wise r of norm-restored transfer (scripts/10)

HepG2 and Jurkat only; the HCT116 pairs rest on 64 or fewer knockdowns.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from zsp.config import load_config

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
NOISE, TRANSFER, MODEL = "#e4e3de", "#c3c2b7", "#2a78d6"
LINES = {"hepg2": "HepG2", "jurkat": "Jurkat"}


def main() -> None:
    cfg = load_config()
    tables, figs = cfg.path("results") / "tables", cfg.path("results") / "figures"
    ceil = pd.read_csv(tables / "noise_ceiling.csv").set_index("held_out")
    models = pd.read_csv(tables / "noise_ceiling_models.csv")
    pair = pd.read_csv(tables / "transferability_pairs.csv")
    r_true = float(
        pair[(pair.line_1 == "hepg2") & (pair.line_2 == "jurkat")].r_true_centred_median.iloc[0]
    )
    rows = []
    for key, label in LINES.items():
        noise = float(ceil.loc[key, "ceiling_centred_median"])
        best = models[(models.held_out == key) & (models.model == "norm_restored")]
        rows.append((label, noise, noise * r_true, float(best.r_centred_median.iloc[0])))

    fig, ax = plt.subplots(figsize=(7.2, 2.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_label, noise, transfer, model) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, noise, height=0.62, color=NOISE, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, transfer, height=0.62, color=TRANSFER, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, model, height=0.22, color=MODEL)
        ax.text(model + 0.008, y - 0.2, f"model {model:.2f}", va="center", fontsize=8, color=INK)
        ax.text(
            transfer - 0.008,
            y + 0.2,
            f"transfer ceiling {transfer:.2f}",
            va="center",
            ha="right",
            fontsize=8,
            color=INK_2,
        )
        ax.text(
            noise - 0.008,
            y + 0.2,
            f"noise ceiling {noise:.2f}",
            va="center",
            ha="right",
            fontsize=8,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "gene-wise correlation with the measured response (centred, median)",
        fontsize=8.5,
        color=INK_2,
    )
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(
        "Transfer is near its own ceiling; the rest of the response is specific to the line",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "ceilings.png", dpi=200, facecolor=SURFACE)
    for r in rows:
        print(f"{r[0]:7s} noise {r[1]:.3f} transfer {r[2]:.3f} model {r[3]:.3f}")


if __name__ == "__main__":
    main()
