"""Decompose the DE members of scored runs: how many genes were called, how often right.

Reads the unthresholded DE tables a `cell-eval2 run --write-degenes` leaves (de_real.parquet,
de_pred.parquet) and reports, per run, the median over perturbations of: genes the reference
calls significant (n_real), genes the prediction calls (n_pred), the precision of the
prediction's calls (same sign as the reference), its coverage min(1, n_pred / n_real), the
Jaccard of the two sets, and the mean |log2 FC| of called and uncalled genes. The target gene
is excluded as the scorer does. Usage: scripts/21_de_calls.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05


def sets(de: pd.DataFrame) -> dict[str, pd.DataFrame]:
    de = de[de.target != de.feature]
    return {t: g.set_index("feature") for t, g in de.groupby("target", sort=True)}


def main() -> None:
    rows = []
    real_cache = {}
    for run in map(Path, sys.argv[1:]):
        real_path = run / "de_real.parquet"
        key = str(real_path.resolve())
        if key not in real_cache:
            real_cache[key] = sets(pd.read_parquet(real_path))
        real = real_cache[key]
        pred = sets(pd.read_parquet(run / "de_pred.parquet"))
        per = []
        for t, r in real.items():
            if t not in pred:
                continue
            p = pred[t]
            R = set(r.index[r.p_adj < ALPHA])
            P = set(p.index[p.p_adj < ALPHA])
            adjud = r.log2_fold_change.replace(0, np.nan).dropna()
            called = [g for g in P if g in adjud.index]
            k = sum(np.sign(p.log2_fold_change.get(g, 0)) == np.sign(adjud[g]) for g in called)
            n_pred, n_real = len(called), len(R)
            per.append(
                {
                    "n_real": n_real,
                    "n_pred": n_pred,
                    "precision": k / n_pred if n_pred else np.nan,
                    "coverage": min(1.0, n_pred / n_real) if n_real else np.nan,
                    "fid": k / max(n_pred, n_real) if max(n_pred, n_real) else np.nan,
                    "jac": len(R & P) / len(R | P) if R | P else 1.0,
                    "abs_lfc_called": p.loc[list(P), "abs_log2_fold_change"].mean()
                    if P
                    else np.nan,
                    "abs_lfc_uncalled": p.loc[~p.index.isin(P), "abs_log2_fold_change"].mean(),
                }
            )
        d = pd.DataFrame(per)
        rows.append(
            {
                "run": run.name,
                **d.median().round(3).to_dict(),
                "n_perts": len(d),
                "fid_mean": round(d.fid.mean(), 3),
                "jac_mean": round(d.jac.mean(), 3),
            }
        )
    print(pd.DataFrame(rows).set_index("run").T.to_string())


if __name__ == "__main__":
    main()
