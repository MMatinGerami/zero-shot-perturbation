"""Collect cell-eval2 outputs from every local run into one table with challenge metric names.

Writes results/tables/local_benchmark.csv (held_out x model x metric) and prints a pivot with
the unweighted mean over the six competition metrics, mirroring the challenge aggregate
(before its reference normalisation, which needs the organisers' baseline artefacts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zsp.config import load_config

# cell-eval2 internal name -> challenge name (vcc2026 preset); higher is better except nmae/mse
METRICS = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
}
LOWER_IS_BETTER = {"mse", "nmae"}
# per-perturbation column the scorer writes for each aggregate (the normalised mse has no
# per-perturbation form; its capped version is bootstrapped instead)
PER_PERT = {**METRICS, "expr_mse_unbiased_capped": "mse"}
N_BOOT = 1000


def bootstrap_ci(values: np.ndarray, rng, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile 95 % CI of the mean over *perturbations* - the unit of replication here;
    cells within a perturbation are not independent transfer experiments."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    # discover finished runs from their scorer output rather than trusting a runs table
    runs = []
    for agg_path in sorted((res / "local_eval").glob("*/agg_results.csv")):
        held_out, _, model = agg_path.parent.name.partition("_")
        if held_out.startswith("smoke"):
            continue
        runs.append((held_out, model, agg_path))
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for held_out, model, agg_path in runs:
        agg = pd.read_csv(agg_path).set_index("statistic")
        per = pd.read_csv(agg_path.parent / "results.csv")
        for internal, name in METRICS.items():
            per_col = internal if internal in set(per.metric) else "expr_mse_unbiased_capped"
            vals = per[per.metric == per_col]["value"].dropna().to_numpy()
            lo, hi = bootstrap_ci(vals, rng) if len(vals) > 1 else (np.nan, np.nan)
            point = float(agg.at["mean", internal])
            # the normalised mse is a ratio of means, so shift its CI by the same factor
            scale = point / vals.mean() if name == "mse" and vals.mean() > 0 else 1.0
            rows.append(
                {
                    "held_out": held_out,
                    "model": model,
                    "metric": name,
                    "value": point,
                    "n_perturbations": len(vals),
                    "ci_low": lo * scale,
                    "ci_high": hi * scale,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(res / "tables" / "local_benchmark.csv", index=False)
    wide = df.pivot_table(index=["held_out", "model"], columns="metric", values="value")
    # direction-free summary: flip lower-is-better metrics so every column reads "higher = better"
    oriented = wide.copy()
    for m in LOWER_IS_BETTER:
        if m in oriented:
            oriented[m] = -oriented[m]
    wide["mean_oriented"] = oriented.mean(axis=1)
    print(wide.round(4).to_string())
    ci = df.assign(
        cell=lambda d: d.apply(lambda r: f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
    )
    print(
        ci.pivot_table(
            index=["held_out", "model"], columns="metric", values="cell", aggfunc="first"
        ).to_string()
    )


if __name__ == "__main__":
    main()
