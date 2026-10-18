"""Build a challenge submission from the source cache, with the round-10 post-processing.

Same output as scripts/05 (one .h5ad over all contexts, `cells_per_pert` cells per target,
raw integer counts, official gene order), but the sources come from the cache of scripts/17
(a few hundred MB instead of the tens of GB the aligned contexts take), and three steps
follow the transfer model:

  * the generic (mean) response may be scaled (`--generic-scale`);
  * the amplitude of the perturbation-specific part is set so that the across-perturbation
    spread of the predicted profiles equals `--rho-target` times the sampling credit the
    emitted groups can claim (zsp.profile.calibrate_amplitude);
  * cells are emitted with exact pooled means (`--exact`, zsp.emit.emit_cells_exact).

The cache must hold every target of the bundle's panel; rebuild it with scripts/17 for a new
panel. `--diagnose-only` prints the per-context numbers (noise, credit, spread, amplitude)
without emitting cells.
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

from zsp.config import load_config
from zsp.data import cp10k
from zsp.emit import emit_cells, emit_cells_exact
from zsp.models import lfc_to_counts
from zsp.profile import calibrate_amplitude, emitted_group_noise, profile
from zsp.submission_io import StreamingH5ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
fast = importlib.import_module("18_fast_eval")
sub = importlib.import_module("05_make_submission")


def predict_means(cache, names, L, generic, basal, v, ctx, control_mean, hit, src_idx, n_panel):
    """Mean counts per cell (targets x panel genes) for one context under variant `v`."""
    targets = L.shape[1]
    target_basal = np.full(len(cache.genes), np.nan)
    target_basal[src_idx] = cp10k(control_mean)[0][hit]
    if v.get("context_weights"):
        # per-context source weights by name, e.g. {"A": {"cd4": 3, "k562": 1}};
        # a context absent from the dict, or a source absent from its entry, gets 1
        cw = v["context_weights"].get(ctx, {})
        w = np.array([float(cw.get(n, 1.0)) for n in names])
    elif v.get("temperature") is None:
        w = np.ones(len(names))
    else:
        sim = fast.basal_similarity(target_basal, basal)
        w = np.exp((sim - sim.max()) / v["temperature"])
    pred, _ = fast.transfer(L, generic, w, v.get("restore_norm", False))
    if v.get("gate"):
        gate = fast.agreement_gate(L, w, v["gate"], v["gate_threshold"])
        pred = pred * np.maximum(gate, v.get("gate_floor", 0.0))
    pred = pred * v.get("shrink", 1.0)
    if v.get("lfc_floor"):
        pred = np.where(np.abs(pred) < v["lfc_floor"], 0.0, pred)
    lfc_panel = np.zeros((targets, n_panel))
    lfc_panel[:, hit] = pred[:, src_idx]
    means = lfc_to_counts(lfc_panel, control_mean)
    return means


def deflate(means, cells, control_mean, k, rng, n_fit=6000):
    """Remove the predicted responses' projection on the top-k control-cell axes (profile
    space); the axes are fitted on a subsample of the context's control cells."""
    idx = rng.choice(cells.shape[0], min(n_fit, cells.shape[0]), replace=False)
    defl = fast.Deflator(sp.csr_matrix(cells[idx]), k)
    b_basal = profile(control_mean)[0]
    delta = defl.apply(profile(means) - b_basal, 1.0)
    out = np.expm1(np.clip(b_basal + delta, 0.0, None))
    return out / out.sum(1, keepdims=True) * control_mean.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True, help="JSON dict of model options (scripts/18)")
    ap.add_argument("--rho-target", type=float, default=None)
    ap.add_argument("--generic-scale", type=float, default=None)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--responder-fraction", type=float, default=1.0)
    ap.add_argument(
        "--deflate-k",
        type=int,
        default=0,
        help="project the predicted response off the top-k principal axes of the context's "
        "own control cells (scripts/18 Deflator), before the amplitude is set",
    )
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--diagnose-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    res = cfg.path("results")
    cache_dir = res / "cache"
    names = args.sources or cfg.context_names(cfg["evaluation"].get("extra_sources", []))
    cache = fast.Cache(cache_dir, names)
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    pert_col, ctx_col = manifest["pert_col"], manifest["context_col"]
    n_cells = args.cells or int(manifest["cells_per_pert"])
    panel = sub.read_genes(bundle / "gene_names.csv")
    perts = sub.read_perts(bundle / "pert_counts.csv")
    targets = perts["target_gene"].astype(str).tolist()
    missing = [t for t in targets if t not in cache.perts]
    if missing:
        raise SystemExit(
            f"{len(missing)} panel targets not in the cache (rebuild scripts/17): {missing[:5]}"
        )
    v = json.loads(args.variant)
    sym_pos = pd.Series(np.arange(len(cache.genes)), index=cache.symbols.to_numpy())
    sym_pos = sym_pos[~sym_pos.index.duplicated()]
    hit = panel.isin(sym_pos.index)
    src_idx = sym_pos.loc[panel[hit]].to_numpy()
    L, generic, basal = cache.stack(names, targets)
    coverage = pd.DataFrame(
        {n: np.isfinite(L[i]).any(1) for i, n in enumerate(names)}, index=targets
    )
    rng = np.random.default_rng(cfg.seed)
    n_total = len(manifest["contexts"]) * len(targets) * n_cells
    writer = (
        None if args.diagnose_only else StreamingH5ad(args.out, pd.DataFrame(index=panel), n_total)
    )
    diag = {}
    for ctx in manifest["contexts"]:
        a = ad.read_h5ad(bundle / f"context_{ctx}.h5ad")
        a = a[a.obs[pert_col].astype(str) == manifest["control_label"]]
        if a.var_names.tolist() != panel.tolist():
            raise ValueError(f"context {ctx} genes are not in the official order")
        cells = sp.csr_matrix(a.X)
        control_mean = np.asarray(cells.mean(0)).ravel()
        means = predict_means(
            cache, names, L, generic, basal, v, ctx, control_mean, hit, src_idx, len(panel)
        )
        if args.deflate_k:
            means = deflate(means, cells, control_mean, args.deflate_k, rng)
        noise = emitted_group_noise(cells, n_cells, rng)
        G = len(panel) - 1
        credit_total = noise * len(targets) / G
        delta = profile(means) - profile(control_mean)[0]
        gen = delta.mean(0, keepdims=True)
        B = ((delta - gen) ** 2).sum() / G
        d = {
            "noise_per_group": noise,
            "credit_total": credit_total,
            "spread_B_at_amplitude_1": B,
            "rho_at_amplitude_1": min(1.0, B / credit_total),
            "median_effect_norm": float(np.median(np.linalg.norm(delta, axis=1))),
            "generic_norm": float(np.linalg.norm(gen)),
            "amplitude": 1.0,
        }
        if args.rho_target or args.generic_scale is not None:
            means, amp = calibrate_amplitude(
                means, control_mean, credit_total, args.rho_target or 1.0, args.generic_scale
            )
            d["amplitude"] = amp
        diag[ctx] = d
        print(f"context {ctx}: " + ", ".join(f"{k} {x:.4g}" for k, x in d.items()), flush=True)
        if writer is None:
            continue
        emit = emit_cells_exact if args.exact else emit_cells
        for i, t in enumerate(targets):
            block = (
                emit_cells_exact(
                    cells, control_mean, means[i], n_cells, rng, args.responder_fraction
                )
                if args.exact
                else emit(cells, control_mean, means[i], n_cells, rng)
            )
            writer.append(block, pd.DataFrame({pert_col: [t] * n_cells, ctx_col: [ctx] * n_cells}))
        print(
            f"context {ctx}: emitted {len(targets)} x {n_cells} cells; nnz so far {writer.nnz:,}",
            flush=True,
        )
        del a, cells
    meta = {
        "variant": v,
        "rho_target": args.rho_target,
        "generic_scale": args.generic_scale,
        "exact": args.exact,
        "responder_fraction": args.responder_fraction,
        "deflate_k": args.deflate_k,
        "sources": names,
        "diagnostics": diag,
        "panel_id": manifest.get("panel_id"),
    }
    if writer is not None:
        obs = writer.close()
        coverage.to_csv(args.out + ".coverage.csv")
        json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
        print(
            f"wrote {len(obs):,} cells x {len(panel):,} genes, {writer.nnz:,} nonzeros; "
            f"{int(hit.sum()):,}/{len(panel):,} panel genes covered; targets with a specific "
            f"effect from >=1 source: {int(coverage.any(axis=1).sum())}/{len(targets)}",
            flush=True,
        )
    else:
        print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
