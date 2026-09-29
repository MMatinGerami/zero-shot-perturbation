"""Response-scale audit: how large is each screen's knockdown response, and by how much does
transfer over- or under-shoot a held-out line?

Two tables:

  results/tables/response_scale_contexts.csv   one row per context: median RMS log2 fold
      change over expressed genes (control CP10k >= 1), median knockdown depth (the target's
      own LFC), cells per perturbation, UMI per cell. A screen whose knockdowns are shallower
      or whose cells are fewer will show a smaller response for reasons that have nothing to
      do with biology.
  results/tables/response_scale_heldout.csv    one row per held-out line and model, from the
      benchmark's own prediction and real files: median over perturbations of the
      predicted/real RMS ratio, of the least-squares slope of real on predicted LFC (the
      factor the prediction should have been multiplied by), and of the gene-wise Pearson r.

Nothing here is used by any model; it is a diagnostic for the open problem named in the
README (the per-context response scale is inherited from the sources, not predicted).
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.data import cp10k, log_fold_change
from zsp.models import decompose
from zsp.store import load_context

STUDY = {
    "k562": "Replogle 2022",
    "rpe1": "Replogle 2022",
    "hepg2": "Nadig 2025",
    "jurkat": "Nadig 2025",
    "hct116": "X-Atlas 2025",
    "hek293t": "X-Atlas 2025",
}
MIN_CP10K = 1.0


def rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(x**2, axis=axis))


def context_rows(cfg) -> pd.DataFrame:
    proc = cfg.path("processed")
    rows = []
    for name in cfg["contexts"]:
        if not (proc / f"{name}_pert.parquet").exists():
            continue
        ctx = load_context(name, proc)
        eff = decompose(ctx)
        expressed = cp10k(ctx.control)[0] >= MIN_CP10K
        lfc = eff.lfc.to_numpy()[:, expressed]
        per_pert = rms(lfc, axis=1)
        umi = float(np.nansum(ctx.control))
        rows.append(
            {
                "context": name,
                "study": STUDY.get(name, ""),
                "n_perturbations": len(eff.lfc),
                "cells_per_pert_median": float(ctx.n_cells.reindex(eff.lfc.index).median()),
                "umi_per_cell": umi,
                "genes_expressed": int(expressed.sum()),
                "rms_lfc_median": float(np.nanmedian(per_pert)),
                "rms_lfc_q25": float(np.nanquantile(per_pert, 0.25)),
                "rms_lfc_q75": float(np.nanquantile(per_pert, 0.75)),
                "generic_rms": float(rms(eff.generic[expressed])),
                "knockdown_depth_median": float(eff.own_lfc.median()),
                "knockdown_depth_q25": float(eff.own_lfc.quantile(0.25)),
                "knockdown_depth_q75": float(eff.own_lfc.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def pseudobulk(path) -> tuple[np.ndarray, pd.DataFrame]:
    """Control mean and per-target mean counts (targets x genes) from a benchmark h5ad."""
    a = ad.read_h5ad(path)
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    target = a.obs["target"].astype(str).to_numpy()
    control = np.asarray(X[target == "non-targeting"].mean(0)).ravel()
    perts = sorted(set(target) - {"non-targeting"})
    means = np.vstack([np.asarray(X[target == p].mean(0)).ravel() for p in perts])
    return control, pd.DataFrame(means, index=perts, columns=a.var_names)


def heldout_rows(cfg) -> pd.DataFrame:
    work = cfg.path("results") / "local_eval"
    rows = []
    for held in cfg["evaluation"]["held_out"]:
        real_path = work / f"{held}_real.h5ad"
        if not real_path.exists():
            continue
        ctrl_real, pert_real = pseudobulk(real_path)
        expressed = cp10k(ctrl_real)[0] >= MIN_CP10K
        lfc_real = log_fold_change(pert_real.to_numpy(), ctrl_real)[:, expressed]
        for pred_path in sorted(work.glob(f"{held}_*_pred.h5ad")):
            model = pred_path.name[len(held) + 1 : -len("_pred.h5ad")]
            if model == "control":
                continue
            ctrl_pred, pert_pred = pseudobulk(pred_path)
            pert_pred = pert_pred.loc[pert_real.index, pert_real.columns]
            lfc_pred = log_fold_change(pert_pred.to_numpy(), ctrl_pred)[:, expressed]
            ratio = rms(lfc_pred, axis=1) / rms(lfc_real, axis=1)
            slope = (lfc_real * lfc_pred).sum(1) / np.maximum((lfc_pred**2).sum(1), 1e-12)
            r = np.array(
                [np.corrcoef(lfc_real[i], lfc_pred[i])[0, 1] for i in range(len(lfc_real))]
            )
            rows.append(
                {
                    "held_out": held,
                    "model": model,
                    "n_perturbations": len(ratio),
                    "genes_expressed": int(expressed.sum()),
                    "rms_real_median": float(np.median(rms(lfc_real, axis=1))),
                    "rms_pred_median": float(np.median(rms(lfc_pred, axis=1))),
                    "ratio_pred_over_real_median": float(np.median(ratio)),
                    "ratio_q25": float(np.quantile(ratio, 0.25)),
                    "ratio_q75": float(np.quantile(ratio, 0.75)),
                    "slope_real_on_pred_median": float(np.median(slope)),
                    "slope_q25": float(np.quantile(slope, 0.25)),
                    "slope_q75": float(np.quantile(slope, 0.75)),
                    "pearson_r_median": float(np.nanmedian(r)),
                }
            )
            msg = (
                f"{held} / {model}: ratio {np.median(ratio):.2f}, "
                f"slope {np.median(slope):.2f}, r {np.nanmedian(r):.2f}"
            )
            print(msg, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heldout", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    tables = cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    ctx = context_rows(cfg)
    ctx.to_csv(tables / "response_scale_contexts.csv", index=False)
    print(ctx.round(3).to_string(index=False), flush=True)
    if not args.skip_heldout:
        held = heldout_rows(cfg)
        held.to_csv(tables / "response_scale_heldout.csv", index=False)
        print(held.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
