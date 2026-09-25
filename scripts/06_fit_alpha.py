"""Pseudobulk-level diagnostics and leave-one-source-out selection of the transfer hyperparameters.

Two things the scorer cannot tell us cheaply:
  1. how much of a knockdown's response is generic (shared by all knockdowns in that line) -
     the quantity naive transfer gets wrong;
  2. which shrinkage `alpha` and basal gate transfer best *between the sources themselves*,
     chosen without ever touching the held-out line (leave-one-source-out, LOSO).

Writes results/tables/generic_share.csv and results/tables/loso_alpha.csv.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from zsp.config import load_config
from zsp.data import align, cp10k
from zsp.models import CalibratedTransfer, decompose
from zsp.store import load_context

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
GATES = [0.0, 0.05, 0.2]


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1) + 1e-12)


def main() -> None:
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    contexts = align([load_context(n, proc) for n in names])
    effects = {c.name: decompose(c) for c in contexts}
    genes = contexts[0].genes
    symbols = contexts[0].symbols.loc[genes].to_numpy()

    # 1. generic share: fraction of total LFC variance explained by the generic component
    rows = []
    for n, e in effects.items():
        total = (e.lfc.to_numpy() ** 2).sum()
        resid = (e.specific.to_numpy() ** 2).sum()
        rows.append(
            {
                "context": n,
                "n_perturbations": len(e.lfc),
                "generic_share": 1 - resid / total,
                "median_own_target_lfc": float(np.nanmedian(e.own_lfc)),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "generic_share.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 2. LOSO: for each held-out line, pick (alpha, gate) that transfers best among its sources
    loso = []
    for held in cfg["evaluation"]["held_out"]:
        source_names = [n for n in names if n != held]
        for alpha, gate in itertools.product(ALPHAS, GATES):
            scores = []
            for pseudo_target in source_names:
                srcs = [effects[n] for n in source_names if n != pseudo_target]
                tgt = effects[pseudo_target]
                perts = [p for p in tgt.lfc.index if any(p in s.lfc.index for s in srcs)]
                basal = cp10k(next(c for c in contexts if c.name == pseudo_target).control)[0]
                pred = CalibratedTransfer(alpha=alpha, gate_cp10k=gate)(
                    srcs, basal, perts, genes, symbols
                )
                truth = tgt.lfc.loc[perts].to_numpy()
                scores.append(pearson_rows(pred.lfc.to_numpy(), truth).mean())
            loso.append(
                {
                    "held_out": held,
                    "alpha": alpha,
                    "gate_cp10k": gate,
                    "mean_pearson_loso": float(np.mean(scores)),
                }
            )
    df = pd.DataFrame(loso)
    df.to_csv(tables / "loso_alpha.csv", index=False)
    best = df.loc[df.groupby("held_out")["mean_pearson_loso"].idxmax()]
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
