"""Evaluate the pre-registered biological examples (results/prereg/biological_examples.md).

Nothing here is tuned: groups, model, metrics and tests were fixed in the markdown file
before any per-target score was read. Writes results/tables/prereg_examples.csv and
results/tables/prereg_rule_c.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from zsp.config import load_config

GROUP_A = ["AARS", "POP7", "GTF3C6", "ACTR2"]
GROUP_B = ["HSD17B12", "GABPA", "E4F1", "GLB1"]
METRICS = {
    "pds_cosine": "pds",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
}
MODEL = "mean_transfer"
N_DECLINE = 20


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    rows, rule_rows = [], []
    pooled = {m: {"A": [], "B": []} for m in METRICS.values()}
    for held in cfg["evaluation"]["held_out"]:
        run = res / "local_eval" / f"{held}_{MODEL}"
        per = pd.read_csv(run / "results.csv")
        wide = per.pivot_table(index="perturbation", columns="metric", values="value")
        wide = wide.rename(columns=METRICS)[list(METRICS.values())]
        median = wide.median()
        for group, names in (("A", GROUP_A), ("B", GROUP_B)):
            for t in names:
                if t not in wide.index:
                    continue
                for m in METRICS.values():
                    v = wide.at[t, m]
                    pooled[m][group].append(v)
                    rows.append(
                        {
                            "held_out": held,
                            "group": group,
                            "target": t,
                            "metric": m,
                            "value": v,
                            "above_median": bool(v > median[m]),
                        }
                    )
        # rule C: highest normalised disagreement -> lower reach / jac
        # the signals are model-independent, so any run of this line carries them
        unc_path = next(p for p in sorted((res / "local_eval").glob(f"{held}_*/uncertainty.csv")))
        unc = pd.read_csv(unc_path, index_col=0)["disagreement_norm"]
        unc = unc.reindex(wide.index).dropna()
        decline = unc.sort_values(ascending=False).index[:N_DECLINE]
        keep = unc.index.difference(decline)
        for m in ("reach", "jac"):
            a, b = wide.loc[decline, m].dropna(), wide.loc[keep, m].dropna()
            _, p = mannwhitneyu(a, b, alternative="less")
            rule_rows.append(
                {
                    "held_out": held,
                    "metric": m,
                    "declined_mean": a.mean(),
                    "kept_mean": b.mean(),
                    "n_declined": len(a),
                    "n_kept": len(b),
                    "p_one_sided": p,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(res / "tables" / "prereg_examples.csv", index=False)
    print(
        df.pivot_table(index=["group", "target"], columns=["metric", "held_out"], values="value")
        .round(3)
        .to_string()
    )
    print("\ngroup A vs B (pooled over lines), one-sided A > B:")
    for m in METRICS.values():
        a, b = np.array(pooled[m]["A"], dtype=float), np.array(pooled[m]["B"], dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]  # the scorer leaves reach undefined for some
        _, p = mannwhitneyu(a, b, alternative="greater")
        print(f"  {m:5s} A={a.mean():.3f} B={b.mean():.3f} p={p:.3f}")
    rc = pd.DataFrame(rule_rows)
    rc.to_csv(res / "tables" / "prereg_rule_c.csv", index=False)
    print("\nrule C (highest disagreement_norm should score lower):")
    print(rc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
