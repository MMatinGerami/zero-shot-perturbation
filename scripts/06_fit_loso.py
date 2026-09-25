"""Leave-one-source-out selection of every model's hyper-parameters (see zsp/loso.py).

For each held-out line, only the *other* contexts take part: each of them in turn is the
pseudo-target, the rest are sources. The selection rule (fixed in advance) is the mean of
the six oriented metric proxies. Writes

  results/tables/loso_grid.csv       every (held_out, model, params) with all proxies
  results/tables/loso_selection.csv  the chosen params per (held_out, model)
  results/tables/generic_share.csv   how much of each context's response is generic
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from zsp.config import load_config
from zsp.data import align_union
from zsp.loso import loso_scores
from zsp.models import (
    CalibratedTransfer,
    GeneScaledTransfer,
    ScaledTransfer,
    WeightedTransfer,
    decompose,
    predict_mean_transfer,
)
from zsp.store import load_context

GRIDS = {
    "mean_transfer": [{}],
    "calibrated": [
        {"alpha": a, "gate_cp10k": g}
        for a, g in itertools.product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.05, 0.2])
    ],
    "weighted": [{"temperature": t} for t in [0.02, 0.05, 0.1, 0.2, 0.5, 1e6]],
    "scaled": [{"scale": s} for s in [0.4, 0.6, 0.8, 1.0, 1.2]],
    "gene_scaled": [{"lam": lam} for lam in [0.1, 0.3, 1.0, 3.0, 10.0]],
}


def make(name: str, params: dict):
    if name == "mean_transfer":
        return lambda: predict_mean_transfer
    cls = {
        "calibrated": CalibratedTransfer,
        "weighted": WeightedTransfer,
        "scaled": ScaledTransfer,
        "gene_scaled": GeneScaledTransfer,
    }[name]
    return lambda: cls(**params)


def main() -> None:
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    contexts = {c.name: c for c in align_union([load_context(n, proc) for n in names])}
    effects = {n: decompose(c) for n, c in contexts.items()}

    rows = []
    for n, e in effects.items():
        total, resid = np.nansum(e.lfc.to_numpy() ** 2), np.nansum(e.specific.to_numpy() ** 2)
        rows.append(
            {
                "context": n,
                "n_perturbations": len(e.lfc),
                "generic_share": 1 - resid / total,
                "median_own_target_lfc": float(np.nanmedian(e.own_lfc)),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "generic_share.csv", index=False)

    grid = []
    for held in cfg["evaluation"]["held_out"]:
        source_names = [n for n in names if n != held]
        for model, params_list in GRIDS.items():
            for params in params_list:
                scores = loso_scores(make(model, params), effects, contexts, source_names)
                grid.append(
                    {"held_out": held, "model": model, "params": json.dumps(params), **scores}
                )
                print(
                    f"{held} {model} {params}: mean_oriented={scores['mean_oriented']:.4f}",
                    flush=True,
                )
    df = pd.DataFrame(grid)
    df.to_csv(tables / "loso_grid.csv", index=False)
    best = df.loc[df.groupby(["held_out", "model"])["mean_oriented"].idxmax()]
    best.to_csv(tables / "loso_selection.csv", index=False)
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
