"""Build a challenge submission from the organisers' control cells and target list.

Expected inputs (from virtualcellchallenge.org after registration; verify column names against
the real files before the first upload):
  --controls   h5ad of one context's non-targeting cells, raw counts, var indexed by symbol
  --targets    text file, one target gene symbol per line
  --cells      cells to emit per perturbation
Output: one h5ad with obs["target"] (controls kept as "non-targeting") and integer raw counts.

The model is the same calibrated transfer used in the local benchmark, fitted on every public
context that has been built (data/processed). Uncertainty per target is written beside the
submission for the write-up; the challenge itself scores only the cells.
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.data import align, cp10k
from zsp.emit import emit_cells
from zsp.models import CalibratedTransfer, decompose, lfc_to_counts
from zsp.store import load_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=1000)
    ap.add_argument("--pert-col", default="target")
    args = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)

    ctrl = ad.read_h5ad(args.controls)
    if args.pert_col in ctrl.obs:
        ctrl = ctrl[ctrl.obs[args.pert_col].astype(str) == "non-targeting"].copy()
    targets = [t.strip() for t in open(args.targets) if t.strip()]
    ctrl_cells = sp.csr_matrix(ctrl.X)
    control_mean = np.asarray(ctrl_cells.mean(0)).ravel()

    # sources: every built public context, aligned to each other, then mapped onto the
    # challenge gene panel by symbol (panel genes no source measured get the generic response 0)
    names = [n for n in cfg["contexts"] if (cfg.path("processed") / f"{n}_pert.parquet").exists()]
    contexts = align([load_context(n, cfg.path("processed")) for n in names])
    src_symbols = contexts[0].symbols.loc[contexts[0].genes].to_numpy()
    sources = [decompose(c) for c in contexts]
    panel = pd.Index(ctrl.var_names)
    sym_pos = pd.Series(np.arange(len(src_symbols)), index=src_symbols)
    sym_pos = sym_pos[~sym_pos.index.duplicated()]
    hit = panel.isin(sym_pos.index)
    src_idx = sym_pos.loc[panel[hit]].to_numpy()

    target_basal_src = cp10k(control_mean)[0][hit]
    pred = CalibratedTransfer()(
        sources, target_basal_src, targets, contexts[0].genes[src_idx], src_symbols[src_idx]
    )
    lfc_panel = np.zeros((len(targets), len(panel)))
    lfc_panel[:, hit] = pred.lfc.to_numpy()
    means = lfc_to_counts(lfc_panel, control_mean)

    blocks, labels = [ctrl_cells], ["non-targeting"] * ctrl_cells.shape[0]
    for i, t in enumerate(targets):
        blocks.append(emit_cells(ctrl_cells, control_mean, means[i], args.cells, rng))
        labels += [t] * args.cells
    sub = ad.AnnData(
        sp.vstack(blocks).tocsr(),
        obs=pd.DataFrame({args.pert_col: labels}),
        var=pd.DataFrame(index=panel),
    )
    sub.obs_names = [f"cell{i}" for i in range(sub.n_obs)]
    sub.write_h5ad(args.out)
    pred.uncertainty.rename("uncertainty").to_frame().to_csv(args.out + ".uncertainty.csv")
    print(
        f"wrote {sub.n_obs:,} cells x {sub.n_vars:,} genes; {len(targets)} targets; "
        f"{hit.sum():,}/{len(panel):,} panel genes covered by sources",
        flush=True,
    )


if __name__ == "__main__":
    main()
