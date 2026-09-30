"""Round 9: two more source contexts, built into the same pseudobulk representation.

  h1    VCC 2025 H1 hESC CRISPRi (Roohani et al., Cell 2025), all three released splits pooled;
        genes are symbols, mapped to Ensembl through the other contexts' own annotations.
  cd4   genome-scale CRISPRi in primary human CD4+ T cells (Zhu et al., Cell 2026,
        doi:10.1016/j.cell.2026.08.002). Only the publisher's differential-expression statistics
        are used; see `cd4` below for how they enter.

Usage: uv run python scripts/13_new_contexts.py h1 [cd4]
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from zsp.config import load_config
from zsp.data import Context, context_from_symbol_cells, cp10k, ensembl_by_symbol
from zsp.store import load_context, save_context

EXISTING = ["k562", "rpe1", "hepg2", "jurkat", "hct116", "hek293t"]


def h1(cfg) -> None:
    raw, proc = cfg.path("raw"), cfg.path("processed")
    ens = ensembl_by_symbol([load_context(n, proc) for n in EXISTING])
    paths = [raw / "h1" / f"adata_{s}.h5ad" for s in ("Training", "Validation", "Test")]
    ctx = context_from_symbol_cells(paths, "h1", ens)
    save_context(ctx, proc)
    print(
        f"h1: {ctx.pert.shape[0]} perturbations x {len(ctx.genes)} genes, "
        f"median {ctx.n_cells.median():.0f} cells per perturbation",
        flush=True,
    )


def h1_eval(cfg, n_control: int = 12_000, per_target: int = 150) -> None:
    """A fixed single-cell subset of H1 for holding it out in the local benchmark, in the same
    layout as the X-Atlas evaluation files (obs `gene`, Ensembl var index, `gene_name`): all
    300 targets with up to `per_target` cells each, plus `n_control` non-targeting cells, drawn
    at random over the three splits. H1 is the only public screen on the challenge's own
    platform (10x Flex, Arc), so it is the closest local stand-in for the hidden lines."""
    import anndata as ad
    import scipy.sparse as sp

    raw, proc = cfg.path("raw"), cfg.path("processed")
    ctx = load_context("h1", proc)
    sym_of = pd.Series(ctx.symbols.loc[ctx.genes].to_numpy(), index=ctx.genes)
    files = [raw / "h1" / f"adata_{s}.h5ad" for s in ("Training", "Validation", "Test")]
    obs = []
    for i, f in enumerate(files):
        a = ad.read_h5ad(f, backed="r")
        obs.append(
            pd.DataFrame(
                {
                    "file": i,
                    "row": np.arange(a.n_obs),
                    "gene": a.obs.target_gene.astype(str).to_numpy(),
                }
            )
        )
    obs = pd.concat(obs, ignore_index=True)
    ctrl = obs[obs.gene == "non-targeting"].sample(n_control, random_state=cfg.seed)
    pert = (
        obs[obs.gene != "non-targeting"]
        .sample(frac=1.0, random_state=cfg.seed)  # shuffle, then keep the first cells per target
        .groupby("gene")
        .head(per_target)
    )
    take = pd.concat([ctrl, pert]).sort_values(["file", "row"])
    blocks = []
    for i, f in enumerate(files):
        a = ad.read_h5ad(f, backed="r")
        cols = pd.Index(a.var_names.astype(str)).get_indexer(sym_of.to_numpy())
        rows = take[take.file == i].row.to_numpy()
        X = a.X[rows]
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        blocks.append(sp.csr_matrix(X)[:, cols])
    out = ad.AnnData(
        sp.vstack(blocks).tocsr().astype(np.float32),
        obs=pd.DataFrame({"gene": take.gene.to_numpy()}),
        var=pd.DataFrame({"gene_name": sym_of.to_numpy()}, index=ctx.genes),
    )
    out.obs_names = [f"h1cell{i}" for i in range(out.n_obs)]
    out.write_h5ad(raw / "h1" / "h1_eval_cells.h5ad")
    print(
        f"h1 eval cells: {out.n_obs} cells, {len(pert.gene.unique())} targets, "
        f"{len(ctrl)} controls",
        flush=True,
    )


def cd4_condition(path, condition: str) -> Context:
    """One culture condition of the CD4+ T-cell screen as a Context.

    The release has no raw counts, only per-target DESeq2 statistics. The basal profile is the
    per-gene median of `baseMean` over that condition's targets (normalised counts of the
    pooled non-targeting and targeting pseudobulks, so close to the control mean for almost every
    gene), and each target's profile is basal x 2**log_fc. `decompose` then recovers the
    publisher's log2 fold change up to the pseudo-count and library renormalisation that every
    other source goes through. Targets flagged as distal off-target are dropped; a target with
    more than one row is averaged."""
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        rows = np.flatnonzero(
            (obs.culture_condition.astype(str) == condition).to_numpy()
            & ~obs.distal_offtarget_flag.astype(bool).to_numpy()
        )
        lfc = np.asarray(f["layers"]["log_fc"][rows, :], dtype=np.float64)
        base = np.asarray(f["layers"]["baseMean"][rows, :], dtype=np.float64)
    sub = obs.iloc[rows]
    genes = pd.Index(var.index.astype(str), name="gene_id")
    basal = np.nanmedian(base, axis=0)
    target = sub.target_contrast_gene_name.astype(str).to_numpy()
    df = pd.DataFrame(lfc, index=target, columns=genes).groupby(level=0).mean()
    pert = pd.DataFrame(basal[None, :] * np.exp2(df.to_numpy()), index=df.index, columns=genes)
    n = sub.groupby(target).n_cells_target.sum().reindex(df.index)
    symbols = pd.Series(var.gene_name.astype(str).to_numpy(), index=genes)
    return Context(f"cd4_{condition.lower()}", genes, symbols, basal, pert, n)


def cd4(cfg) -> None:
    """Build every culture condition, then keep ONE as the `cd4` source: the condition whose
    basal profile is most similar (mean Pearson of log1p CP10k over shared genes) to the
    challenge's own validation contexts A to C. Fixed before any benchmark with CD4 was run;
    three T-cell contexts would otherwise count one cell type three times."""
    import anndata as ad

    raw, proc = cfg.path("raw"), cfg.path("processed")
    conds = {
        c: cd4_condition(raw / "cd4" / "GWCD4i.DE_stats.h5ad", c)
        for c in ["Rest", "Stim8hr", "Stim48hr"]
    }
    ens = ensembl_by_symbol([load_context(n, proc) for n in EXISTING])
    targets = {}
    for ctx_name in "ABC":
        a = ad.read_h5ad(raw / "vcc" / f"context_{ctx_name}.h5ad", backed="r")
        X = a.X[:5000]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        prof = pd.Series(cp10k(X.mean(0))[0], index=a.var_names.astype(str))
        prof = prof[prof.index.isin(ens.index)]
        prof.index = ens.loc[prof.index].to_numpy()
        targets[ctx_name] = prof[~prof.index.duplicated()]
    rows = []
    for c, ctx in conds.items():
        b = pd.Series(cp10k(ctx.control)[0], index=ctx.genes)
        sims = []
        for t in targets.values():
            g = b.index.intersection(t.index)
            sims.append(np.corrcoef(np.log1p(b.loc[g]), np.log1p(t.loc[g]))[0, 1])
        rows.append(
            {
                "condition": c,
                **{f"sim_{k}": s for k, s in zip("ABC", sims, strict=True)},
                "sim_mean": float(np.mean(sims)),
                "n_targets": len(ctx.pert),
            }
        )
    sel = pd.DataFrame(rows).sort_values("sim_mean", ascending=False)
    sel.to_csv(cfg.path("results") / "tables" / "cd4_condition_selection.csv", index=False)
    print(sel.round(3).to_string(index=False), flush=True)
    best = conds[sel.iloc[0].condition]
    chosen = Context("cd4", best.genes, best.symbols, best.control, best.pert, best.n_cells)
    save_context(chosen, proc)
    print(f"cd4 = {sel.iloc[0].condition}: {len(chosen.pert)} targets x {len(chosen.genes)} genes")


def main(which: list[str]) -> None:
    cfg = load_config()
    for name in which or ["h1"]:
        {"h1": h1, "h1_eval": h1_eval, "cd4": cd4}[name](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
