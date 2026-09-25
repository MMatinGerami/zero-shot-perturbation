"""Build a challenge submission from the organisers' control bundle.

Format (manifest.json in the bundle): one .h5ad covering contexts A, B, C with
obs["target_gene"] (no control cells), obs["context"], exactly `cells_per_pert` cells per
(context, target), raw integer counts, var indexed by the official gene list
(gene_names.csv, 18,533 symbols, in order). `vcc prep` validates all of this locally.

Model: calibrated transfer fitted on every built public context (data/processed); alpha
and the basal gate default to the leave-one-source-out choice pooled over the local
held-out lines (results/tables/loso_alpha.csv). The prediction is streamed to disk block
by block (~2 billion nonzeros do not fit in memory as one matrix). Per-target uncertainty
and source coverage are written beside the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.data import align_union, cp10k
from zsp.emit import emit_cells
from zsp.models import CalibratedTransfer, decompose, lfc_to_counts, predict_mean_transfer
from zsp.store import load_context
from zsp.submission_io import StreamingH5ad


def load_sources(cfg):
    proc = cfg.path("processed")
    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    contexts = align_union([load_context(n, proc) for n in names])
    symbols = contexts[0].symbols.loc[contexts[0].genes].to_numpy()
    return contexts, [decompose(c) for c in contexts], symbols


def loso_pooled(res: Path) -> CalibratedTransfer:
    """(alpha, gate) maximising the leave-one-source-out score averaged over held-out lines."""
    path = res / "tables" / "loso_alpha.csv"
    if not path.exists():
        return CalibratedTransfer()
    t = pd.read_csv(path).groupby(["alpha", "gate_cp10k"])["mean_pearson_loso"].mean()
    alpha, gate = t.idxmax()
    return CalibratedTransfer(alpha=float(alpha), gate_cp10k=float(gate))


def read_genes(path) -> pd.Index:
    """gene_names.csv, with or without a header line."""
    col = pd.read_csv(path, header=None)[0].astype(str)
    if col.iloc[0].lower() in ("gene_name", "gene", "symbol"):
        col = col.iloc[1:]
    return pd.Index(col.to_numpy(), name="gene")


def read_perts(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    target_col = next(
        c for c in df.columns if c.lower() in ("target_gene", "target", "perturbation", "gene")
    )
    return df.rename(columns={target_col: "target_gene"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="unpacked controls bundle directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=None, help="override cells per perturbation")
    ap.add_argument("--n-perts", type=int, default=None, help="debug: only the first N targets")
    ap.add_argument("--model", choices=["mean_transfer", "calibrated"], default="calibrated")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gate", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    bundle = Path(args.bundle)

    manifest = json.loads((bundle / "manifest.json").read_text())
    pert_col, ctx_col = manifest["pert_col"], manifest["context_col"]
    n_cells = args.cells or int(manifest["cells_per_pert"])
    panel = read_genes(bundle / "gene_names.csv")
    perts = read_perts(bundle / "pert_counts.csv")
    if args.n_perts:
        perts = perts.head(args.n_perts)
    if len(panel) != manifest["n_genes"]:
        raise ValueError(f"gene list has {len(panel)} genes, manifest says {manifest['n_genes']}")

    contexts, sources, src_symbols = load_sources(cfg)
    if args.model == "mean_transfer":
        model = predict_mean_transfer
        print(f"sources: {[c.name for c in contexts]}; model=mean_transfer")
    else:
        model = loso_pooled(cfg.path("results"))
        if args.alpha is not None:
            model.alpha = args.alpha
        if args.gate is not None:
            model.gate_cp10k = args.gate
        print(
            f"sources: {[c.name for c in contexts]}; model=calibrated "
            f"alpha={model.alpha} gate={model.gate_cp10k}"
        )

    sym_pos = pd.Series(np.arange(len(src_symbols)), index=src_symbols)
    sym_pos = sym_pos[~sym_pos.index.duplicated()]
    hit = panel.isin(sym_pos.index)
    src_idx = sym_pos.loc[panel[hit]].to_numpy()
    coverage = pd.DataFrame(
        {s.name: perts["target_gene"].isin(s.specific.index).to_numpy() for s in sources},
        index=perts["target_gene"],
    )

    n_total = len(manifest["contexts"]) * len(perts) * n_cells
    writer = StreamingH5ad(args.out, pd.DataFrame(index=panel), n_total)
    unc_rows = []
    for ctx in manifest["contexts"]:
        a = ad.read_h5ad(bundle / f"context_{ctx}.h5ad")
        a = a[a.obs[pert_col].astype(str) == manifest["control_label"]]
        if a.var_names.tolist() != panel.tolist():
            raise ValueError(f"context {ctx} genes are not in the official order")
        cells = sp.csr_matrix(a.X)
        control_mean = np.asarray(cells.mean(0)).ravel()
        ctx_perts = perts[perts[ctx_col] == ctx] if ctx_col in perts else perts
        targets = ctx_perts["target_gene"].astype(str).tolist()
        counts = (
            ctx_perts["n_cells"].astype(int).tolist()
            if "n_cells" in ctx_perts
            else [n_cells] * len(targets)
        )

        # the model works on the full source gene set; basal state for source genes the
        # panel lacks is 0, so they are gated and then dropped when mapping back to the panel
        basal_src = np.zeros(len(src_symbols))
        basal_src[src_idx] = cp10k(control_mean)[0][hit]
        pred = model(sources, basal_src, targets, contexts[0].genes, src_symbols)
        lfc_panel = np.zeros((len(targets), len(panel)))
        lfc_panel[:, hit] = pred.lfc.to_numpy()[:, src_idx]
        means = lfc_to_counts(lfc_panel, control_mean)
        for i, (t, n) in enumerate(zip(targets, counts, strict=True)):
            block = emit_cells(cells, control_mean, means[i], n, rng)
            writer.append(block, pd.DataFrame({pert_col: [t] * n, ctx_col: [ctx] * n}))
        if len(pred.uncertainty):
            unc_rows.append(pred.uncertainty.rename("uncertainty").to_frame().assign(context=ctx))
        print(
            f"context {ctx}: {len(targets)} targets x {n_cells} cells "
            f"from {cells.shape[0]} controls; nnz so far {writer.nnz:,}",
            flush=True,
        )
        del a, cells

    obs = writer.close()
    if unc_rows:
        pd.concat(unc_rows).to_csv(args.out + ".uncertainty.csv")
    coverage.to_csv(args.out + ".coverage.csv")
    print(
        f"wrote {len(obs):,} cells x {len(panel):,} genes, {writer.nnz:,} nonzeros; "
        f"{hit.sum():,}/{len(panel):,} panel genes covered by sources; "
        f"targets with a specific effect from >=1 source: "
        f"{int(coverage.any(axis=1).sum())}/{len(perts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
