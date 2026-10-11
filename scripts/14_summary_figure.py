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
from zsp.plotting import BLUE, DOUBLE, GREY_LIGHT, GREY_MID, INK, INK_2, MM, apply_style, save

LINES = {"hepg2": "HepG2", "jurkat": "Jurkat"}


def plot(rows: list[tuple[str, float, float, float]], figs) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.7, 40 * MM))
    for i, (_label, noise, transfer, model) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, noise, height=0.62, color=GREY_LIGHT, lw=0)
        ax.barh(y, transfer, height=0.62, color=GREY_MID, lw=0)
        ax.barh(y, model, height=0.22, color=BLUE, lw=0)
        ax.text(model + 0.008, y - 0.2, f"Model {model:.2f}", va="center", fontsize=5.5, color=INK)
        ax.text(
            transfer - 0.008,
            y + 0.2,
            f"Transfer ceiling {transfer:.2f}",
            va="center",
            ha="right",
            fontsize=5.5,
            color=INK_2,
        )
        ax.text(
            noise - 0.008,
            y + 0.2,
            f"Noise ceiling {noise:.2f}",
            va="center",
            ha="right",
            fontsize=5.5,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Gene-wise correlation with the measured response (centred, median)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, figs / "ceilings")


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

    plot(rows, figs)
    for r in rows:
        print(f"{r[0]:7s} noise {r[1]:.3f} transfer {r[2]:.3f} model {r[3]:.3f}")


if __name__ == "__main__":
    main()
