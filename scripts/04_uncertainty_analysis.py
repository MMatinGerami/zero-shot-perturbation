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

# per-perturbation metrics; lower-is-better ones are negated so every score reads
# "higher = better prediction of that perturbation"
QUALITY = {
    "pds_cosine": ("pds", 1),
    "expr_mse_unbiased_capped": ("mse", -1),
    "de_wilcoxon_lfc_nmae": ("nmae", -1),
    "de_wilcoxon_direction_fidelity_yield_raw": ("fid", 1),
    "de_wilcoxon_direction_reach_raw": ("reach", 1),
    "de_wilcoxon_sig_jaccard": ("jac", 1),
}
COVERAGE = np.linspace(0.3, 1.0, 15)


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    corr_rows, sel_rows = [], []
    for unc_path in sorted((res / "local_eval").glob("*/uncertainty.csv")):
        held_out, _, model = unc_path.parent.name.partition("_")
        signals = pd.read_csv(unc_path, index_col=0)
        per = pd.read_csv(unc_path.parent / "results.csv")
        for signal in signals.columns:
            unc = signals[signal].dropna()
            for internal, (name, sign) in QUALITY.items():
                q = sign * per[per.metric == internal].set_index("perturbation")["value"].dropna()
                common = q.index.intersection(unc.index)
                if len(common) < 10:
                    continue
                rho, p = spearmanr(unc.loc[common], q.loc[common])
                corr_rows.append(
                    {
                        "held_out": held_out,
                        "model": model,
                        "signal": signal,
                        "metric": name,
                        "n": len(common),
                        "spearman_rho": rho,
                        "p_value": p,
                    }
                )
                order = np.argsort(unc.loc[common].to_numpy(), kind="stable")  # most certain first
                sorted_q = q.loc[common].to_numpy()[order]
                for cov in COVERAGE:
                    k = max(1, round(cov * len(sorted_q)))
                    sel_rows.append(
                        {
                            "held_out": held_out,
                            "model": model,
                            "signal": signal,
                            "metric": name,
                            "coverage": cov,
                            "mean_score": sorted_q[:k].mean(),
                            "gain_vs_full": sorted_q[:k].mean() - sorted_q.mean(),
                        }
                    )
    corr = pd.DataFrame(corr_rows)
    sel = pd.DataFrame(sel_rows)
    corr.to_csv(res / "tables" / "uncertainty_correlation.csv", index=False)
    sel.to_csv(res / "tables" / "uncertainty_selective.csv", index=False)
    print(corr.round(3).to_string(index=False))

    if len(sel):
        # does keeping the 50 % most-certain targets raise the score? one number per
        # (signal, metric), averaged over held-out lines, for the reference model
        ref = sel[(sel.model == "mean_transfer") & (np.isclose(sel.coverage, 0.5))]
        gain = ref.pivot_table(
            index="signal", columns="metric", values="gain_vs_full", aggfunc="mean"
        )
        gain.to_csv(res / "tables" / "uncertainty_gain_at_half_coverage.csv")
        print("\nmean gain at 50 % coverage (mean_transfer, averaged over held-out lines):")
        print(gain.round(3).to_string())
        (res / "figures").mkdir(exist_ok=True)
        metrics = [name for name, _ in QUALITY.values()]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, m in zip(axes.ravel(), metrics, strict=True):
            d = ref[ref.metric == m]
            for signal, dd in d.groupby("signal"):
                curve = dd.groupby("coverage")["mean_score"].mean()
                ax.plot(curve.index, curve.to_numpy(), marker="o", ms=3, label=signal)
            ax.set(xlabel="coverage (most certain fraction kept)", ylabel=f"mean {m}", title=m)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7)
        fig.suptitle("Selective prediction, mean transfer, averaged over held-out lines")
        fig.tight_layout()
        fig.savefig(res / "figures" / "uncertainty.png", dpi=200)


if __name__ == "__main__":
    main()
