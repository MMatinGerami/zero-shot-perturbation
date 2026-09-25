"""Does the model's per-perturbation uncertainty predict which predictions are wrong?

For every (held-out context, model) run that emitted an uncertainty score, join it with the
scorer's per-perturbation metrics and ask two questions:
  1. rank correlation: are less-certain perturbations scored worse?
  2. selective prediction: if the least-certain fraction were withheld, how much does the
     mean score of the rest improve?

Writes results/tables/uncertainty_{correlation,selective}.csv and figures/uncertainty.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from zsp.config import load_config

# per-perturbation metrics where higher = better prediction of that perturbation
QUALITY = {
    "pds_cosine": "pds",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_direction_reach_raw": "reach",
}
COVERAGE = np.linspace(0.3, 1.0, 15)


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    corr_rows, sel_rows = [], []
    for unc_path in sorted((res / "local_eval").glob("*/uncertainty.csv")):
        held_out, _, model = unc_path.parent.name.partition("_")
        unc = pd.read_csv(unc_path, index_col=0)["uncertainty"]
        per = pd.read_csv(unc_path.parent / "results.csv")
        for internal, name in QUALITY.items():
            q = per[per.metric == internal].set_index("perturbation")["value"].dropna()
            common = q.index.intersection(unc.index)
            if len(common) < 10:
                continue
            rho, p = spearmanr(unc.loc[common], q.loc[common])
            corr_rows.append(
                {
                    "held_out": held_out,
                    "model": model,
                    "metric": name,
                    "n": len(common),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )
            order = np.argsort(unc.loc[common].to_numpy())  # most certain first
            sorted_q = q.loc[common].to_numpy()[order]
            for cov in COVERAGE:
                k = max(1, round(cov * len(sorted_q)))
                sel_rows.append(
                    {
                        "held_out": held_out,
                        "model": model,
                        "metric": name,
                        "coverage": cov,
                        "mean_score": sorted_q[:k].mean(),
                    }
                )
    corr = pd.DataFrame(corr_rows)
    sel = pd.DataFrame(sel_rows)
    corr.to_csv(res / "tables" / "uncertainty_correlation.csv", index=False)
    sel.to_csv(res / "tables" / "uncertainty_selective.csv", index=False)
    print(corr.round(3).to_string(index=False))

    if len(sel):
        (res / "figures").mkdir(exist_ok=True)
        metrics = list(QUALITY.values())
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), squeeze=False)
        for ax, m in zip(axes[0], metrics, strict=True):
            for (held, model), d in sel[sel.metric == m].groupby(["held_out", "model"]):
                ax.plot(
                    d.coverage,
                    d.mean_score,
                    marker="o",
                    ms=3,
                    label=f"{held} / {model}",
                )
            ax.set(
                xlabel="coverage (most certain fraction kept)",
                ylabel=f"mean {m}",
                title=m,
            )
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(res / "figures" / "uncertainty.png", dpi=200)


if __name__ == "__main__":
    main()
