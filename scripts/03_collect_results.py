"""Collect cell-eval2 outputs from every local run into one table with challenge metric names.

Writes results/tables/local_benchmark.csv (held_out x model x metric) and prints a pivot with
the unweighted mean over the six competition metrics, mirroring the challenge aggregate
(before its reference normalisation, which needs the organisers' baseline artefacts).
"""

from __future__ import annotations

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
    for held_out, model, agg_path in runs:
        agg = pd.read_csv(agg_path).set_index("statistic")
        for internal, name in METRICS.items():
            rows.append(
                {
                    "held_out": held_out,
                    "model": model,
                    "metric": name,
                    "value": float(agg.at["mean", internal]),
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


if __name__ == "__main__":
    main()
