"""How many genes would a variant call significant on the challenge contexts?

The leaderboard's direction fidelity is precision times coverage, and coverage is the number
of genes the submission calls over the number the reference calls, which is hidden. This
script emits cells for a sample of panel targets exactly as the builder (scripts/20) would,
runs the scorer's test (two-sided Wilcoxon rank-sum on CPM against the context's control
cells, genes with control mean above 5 CPM, Benjamini-Hochberg within the target, alpha 0.05)
and reports the median number of calls per target and context. With one leaderboard reading
of fid for a variant whose precision is known, the reference's count follows.

Variants are JSON dicts as in scripts/18, plus the builder's post-processing keys
`rho_target` and `generic_scale` and the emitter choice `exact` (default true).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from zsp.config import load_config
from zsp.emit import emit_cells, emit_cells_exact
from zsp.profile import calibrate_amplitude, emitted_group_noise

sys.path.insert(0, str(Path(__file__).resolve().parent))
fast = importlib.import_module("18_fast_eval")
build = importlib.import_module("20_make_submission_cached")
sub = importlib.import_module("05_make_submission")


def cpm(X: sp.csr_matrix) -> np.ndarray:
    X = X.toarray().astype(np.float64)
    return X / np.maximum(X.sum(1, keepdims=True), 1) * 1e6


def n_calls(pert_cpm: np.ndarray, ctrl_cpm: np.ndarray, tested: np.ndarray) -> int:
    p = mannwhitneyu(pert_cpm[:, tested], ctrl_cpm[:, tested], axis=0, method="asymptotic").pvalue
    p = np.nan_to_num(p, nan=1.0)
    return int(multipletests(p, alpha=0.05, method="fdr_bh")[0].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/raw/vcc")
    ap.add_argument("--contexts", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-targets", type=int, default=24)
    ap.add_argument("--n-controls", type=int, default=6000, help="control cells used in the test")
    args = ap.parse_args()
    cfg = load_config()
    res = cfg.path("results")
    names = cfg.context_names(cfg["evaluation"].get("extra_sources", []))
    cache = fast.Cache(res / "cache", names)
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    n_cells = int(manifest["cells_per_pert"])
    panel = sub.read_genes(bundle / "gene_names.csv")
    targets = sub.read_perts(bundle / "pert_counts.csv")["target_gene"].astype(str).tolist()
    sym_pos = pd.Series(np.arange(len(cache.genes)), index=cache.symbols.to_numpy())
    sym_pos = sym_pos[~sym_pos.index.duplicated()]
    hit = panel.isin(sym_pos.index)
    src_idx = sym_pos.loc[panel[hit]].to_numpy()
    L, generic, basal = cache.stack(names, targets)
    rng = np.random.default_rng(cfg.seed)
    sample = sorted(rng.choice(len(targets), args.n_targets, replace=False))
    rows = []
    for ctx in args.contexts:
        a = ad.read_h5ad(bundle / f"context_{ctx}.h5ad")
        a = a[a.obs[manifest["pert_col"]].astype(str) == manifest["control_label"]]
        cells = sp.csr_matrix(a.X)
        control_mean = np.asarray(cells.mean(0)).ravel()
        ctrl_idx = rng.choice(cells.shape[0], min(args.n_controls, cells.shape[0]), replace=False)
        ctrl_cpm = cpm(cells[ctrl_idx])
        tested = ctrl_cpm.mean(0) > 5
        noise = emitted_group_noise(cells, n_cells, rng)
        for spec in args.variants:
            v = json.loads(spec)
            means = build.predict_means(
                cache, names, L, generic, basal, v, ctx, control_mean, hit, src_idx, len(panel)
            )
            amp = 1.0
            if v.get("deflate_k"):
                means = build.deflate(means, cells, control_mean, int(v["deflate_k"]), rng)
            if v.get("rho_target") or v.get("generic_scale") is not None:
                credit = noise * len(targets) / (len(panel) - 1)
                means, amp = calibrate_amplitude(
                    means, control_mean, credit, v.get("rho_target") or 1.0, v.get("generic_scale")
                )
            emit = emit_cells_exact if v.get("exact", True) else emit_cells
            counts = []
            for i in sample:
                block = (
                    emit_cells_exact(
                        cells,
                        control_mean,
                        means[i],
                        n_cells,
                        rng,
                        v.get("responder_fraction", 1.0),
                    )
                    if v.get("exact", True)
                    else emit(cells, control_mean, means[i], n_cells, rng)
                )
                counts.append(n_calls(cpm(block), ctrl_cpm, tested))
            row = {
                "context": ctx,
                "variant": spec,
                "amplitude": round(amp, 3),
                "calls_median": int(np.median(counts)),
                "calls_q1": int(np.percentile(counts, 25)),
                "calls_q3": int(np.percentile(counts, 75)),
                "genes_tested": int(tested.sum()),
            }
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(res / "tables" / "call_counts.csv", index=False)


if __name__ == "__main__":
    main()
