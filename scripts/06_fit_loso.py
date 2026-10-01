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
    AgreementTransfer,
    BasalModulatedTransfer,
    CalibratedTransfer,
    DepthAwareTransfer,
    GeneScaledTransfer,
    MedianTransfer,
    NormRestoredTransfer,
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
    "norm_restored": [{"temperature": t} for t in [0.05, 0.1, 0.2, 1e6]],
    "median": [{"temperature": t} for t in [0.05, 0.1, 0.2, 1e6]],
    "depth_aware": [
        {"temperature": 0.1, "depth_weight": k, "normalise": n}
        for k, n in itertools.product([0.0, 1.0, 2.0], [False, True])
    ],
    "basal_modulated": [
        {"temperature": 0.1, "restore": r} for r in ["none", "consensus", "source"]
    ],
    "agreement": [
        {"temperature": 0.1, "threshold": t, "restore": r}
        for t, r in itertools.product([0.0, 0.5, 0.75, 1.0], ["none", "source"])
    ],
}


def make(name: str, params: dict):
    if name == "mean_transfer":
        return lambda: predict_mean_transfer
    cls = {
        "calibrated": CalibratedTransfer,
        "weighted": WeightedTransfer,
        "scaled": ScaledTransfer,
        "gene_scaled": GeneScaledTransfer,
        "norm_restored": NormRestoredTransfer,
        "median": MedianTransfer,
        "depth_aware": DepthAwareTransfer,
        "basal_modulated": BasalModulatedTransfer,
        "agreement": AgreementTransfer,
    }[name]
    return lambda: cls(**params)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of the grids to run (default: all)")
    ap.add_argument("--extra-sources", nargs="*", default=[], help="round 9 sources, e.g. h1 cd4")
    ap.add_argument(
        "--challenge",
        action="store_true",
        help="the submission setting: no local line is held out and every context is a "
        "pseudo-target (rows tagged held_out='challenge')",
    )
    args = ap.parse_args()
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = cfg.context_names(args.extra_sources)
    suffix = "".join(f"_{n}" for n in args.extra_sources)  # never overwrite the core tables
    suffix += "_challenge" if args.challenge else ""
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
    pd.DataFrame(rows).to_csv(tables / f"generic_share{suffix}.csv", index=False)

    grid = []
    for held in ["challenge"] if args.challenge else cfg["evaluation"]["held_out"]:
        source_names = [n for n in names if n != held]
        for model, params_list in GRIDS.items():
            if args.models and model not in args.models:
                continue
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
    path = tables / f"loso_grid{suffix}.csv"
    if args.models and path.exists():  # partial run: replace only the models that were rerun
        old = pd.read_csv(path)
        df = pd.concat([old[~old.model.isin(args.models)], df], ignore_index=True)
    df.to_csv(path, index=False)
    best = df.loc[df.groupby(["held_out", "model"])["mean_oriented"].idxmax()]
    best.to_csv(tables / f"loso_selection{suffix}.csv", index=False)
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
