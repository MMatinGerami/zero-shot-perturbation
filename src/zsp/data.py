"""Turning public Perturb-seq screens into per-perturbation pseudobulk profiles.

Every context is reduced to the same representation, a `Context`:
  * `control`  - mean raw UMI counts per cell of non-targeting cells (the basal state);
  * `pert`     - mean raw UMI counts per cell for each perturbation target;
  * `n_cells`  - cells contributing to each perturbation mean.
Genes are Ensembl IDs. Contexts are aligned on the *union* of their genes: a gene a
context did not measure is NaN in that context (not zero - "not measured" and "measured at
zero" must never be confused), and every downstream step is NaN-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CONTROL = "non-targeting"


@dataclass
class Context:
    name: str
    genes: pd.Index  # Ensembl IDs
    symbols: pd.Series  # Ensembl ID -> gene symbol
    control: np.ndarray  # (n_genes,)
    pert: pd.DataFrame  # (n_perts, n_genes) mean counts, index = target symbol
    n_cells: pd.Series  # target symbol -> cells

    def subset_genes(self, genes: pd.Index) -> Context:
        pos = self.genes.get_indexer(genes)
        if (pos < 0).any():
            raise KeyError("requested genes missing from context")
        return Context(
            self.name,
            genes,
            self.symbols.loc[genes],
            self.control[pos],
            self.pert.iloc[:, pos].set_axis(genes, axis=1),
            self.n_cells,
        )

    def reindex_genes(self, genes: pd.Index, symbols: pd.Series) -> Context:
        """Like subset_genes, but genes this context lacks become NaN."""
        pos = self.genes.get_indexer(genes)
        control = np.full(len(genes), np.nan)
        control[pos >= 0] = self.control[pos[pos >= 0]]
        pert = self.pert.reindex(columns=genes)
        return Context(self.name, genes, symbols.loc[genes], control, pert, self.n_cells)

    @property
    def measured(self) -> np.ndarray:
        return np.isfinite(self.control)


def _weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (X * w[:, None]).sum(0) / w.sum()


def context_from_bulk(path: str | Path, name: str) -> Context:
    """Replogle-style pseudobulk h5ad: one row per guide pair, obs.num_cells_filtered."""
    a = ad.read_h5ad(path)
    X = np.asarray(a.X, dtype=np.float64)
    target = a.obs.index.to_series().str.split("_").str[1].to_numpy()
    w = a.obs["num_cells_filtered"].fillna(0).to_numpy(dtype=np.float64)
    ctrl = target == CONTROL
    control = _weighted_mean(X[ctrl], w[ctrl])
    rows, counts = {}, {}
    for t in np.unique(target[~ctrl]):
        m = (target == t) & (w > 0)
        if m.any():
            rows[t] = _weighted_mean(X[m], w[m])
            counts[t] = w[m].sum()
    genes = pd.Index(a.var_names, name="gene_id")
    pert = pd.DataFrame(np.vstack(list(rows.values())), index=list(rows), columns=genes)
    return Context(name, genes, a.var["gene_name"].astype(str), control, pert, pd.Series(counts))


def context_from_singlecell(path: str | Path, name: str, chunk: int = 20_000) -> Context:
    """Nadig-style single-cell h5ad with obs.gene; pseudobulked by streaming row chunks."""
    a = ad.read_h5ad(path, backed="r")
    target = a.obs["gene"].astype(str).to_numpy()
    codes, uniq = pd.factorize(target)
    sums = np.zeros((len(uniq), a.n_vars))
    for start in range(0, a.n_obs, chunk):
        X = np.asarray(a.X[start : start + chunk], dtype=np.float64)
        np.add.at(sums, codes[start : start + chunk], X)
    n = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
    means = sums / n[:, None]
    genes = pd.Index(a.var_names, name="gene_id")
    df = pd.DataFrame(means, index=uniq, columns=genes)
    control = df.loc[CONTROL].to_numpy()
    pert = df.drop(index=CONTROL)
    counts = pd.Series(n, index=uniq).drop(CONTROL)
    return Context(name, genes, a.var["gene_name"].astype(str), control, pert, counts)


def context_from_symbol_cells(
    paths: list[str | Path],
    name: str,
    ensembl_of: pd.Series,
    target_col: str = "target_gene",
    chunk: int = 20_000,
) -> Context:
    """Single-cell h5ads whose genes are symbols (the VCC 2025 H1 release), pooled over files
    that share one gene axis (train / validation / test splits, each with its own controls).
    Symbols are mapped to Ensembl IDs with `ensembl_of` (symbol -> Ensembl, built from the other
    contexts); unmapped or ambiguous symbols are dropped, never guessed."""
    sums: dict[str, np.ndarray] = {}
    n: dict[str, float] = {}
    var_names = None
    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        if var_names is None:
            var_names = pd.Index(a.var_names.astype(str))
        elif not var_names.equals(pd.Index(a.var_names.astype(str))):
            raise ValueError(f"{path}: gene axis differs from the first file")
        target = a.obs[target_col].astype(str).to_numpy()
        for start in range(0, a.n_obs, chunk):
            block = a.X[start : start + chunk]
            X = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            t = target[start : start + chunk]
            codes, uniq = pd.factorize(t)
            part = np.zeros((len(uniq), X.shape[1]))
            np.add.at(part, codes, X.astype(np.float64))
            for i, u in enumerate(uniq):
                sums[u] = sums.get(u, 0.0) + part[i]
                n[u] = n.get(u, 0.0) + float((codes == i).sum())
    ens = ensembl_of.reindex(var_names)
    keep = ens.notna().to_numpy() & ~ens.duplicated(keep=False).to_numpy()
    genes = pd.Index(ens[keep].to_numpy(), name="gene_id")
    symbols = pd.Series(var_names[keep].to_numpy(), index=genes)
    means = {u: sums[u][keep] / n[u] for u in sums}
    control = means.pop(CONTROL)
    counts = pd.Series(n).drop(CONTROL)
    pert = pd.DataFrame(np.vstack(list(means.values())), index=list(means), columns=genes)
    return Context(name, genes, symbols, control, pert, counts.reindex(pert.index))


def ensembl_by_symbol(contexts: list[Context]) -> pd.Series:
    """symbol -> Ensembl ID from contexts' own annotations; symbols that map to more than one
    ID are left out."""
    pairs = pd.concat(
        [pd.Series(c.genes.to_numpy(), index=c.symbols.loc[c.genes].to_numpy()) for c in contexts]
    )
    pairs = pairs[~pairs.index.isna()]
    uniq = pairs.groupby(level=0).nunique()
    good = uniq[uniq == 1].index
    return pairs[pairs.index.isin(good)].groupby(level=0).first()


def cp10k(x: np.ndarray) -> np.ndarray:
    """Library-size normalise mean counts (rows or a single vector) to counts per 10k.

    NaN entries (genes a context did not measure) are left out of the library size and
    stay NaN."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x / np.nansum(x, axis=1, keepdims=True) * 1e4


def log_fold_change(pert: np.ndarray, control: np.ndarray, pseudo: float = 0.1) -> np.ndarray:
    """log2 fold change of CP10k-normalised pseudobulks against the control."""
    return np.log2(cp10k(pert) + pseudo) - np.log2(cp10k(control) + pseudo)


def align(contexts: list[Context]) -> list[Context]:
    """Restrict all contexts to their shared genes (order of the first context)."""
    shared = contexts[0].genes
    for c in contexts[1:]:
        shared = shared.intersection(c.genes, sort=False)
    return [c.subset_genes(shared) for c in contexts]


def align_union(contexts: list[Context]) -> list[Context]:
    """Put all contexts on the union of their genes (first context's order, then new
    genes in order of appearance); unmeasured genes are NaN."""
    genes = contexts[0].genes
    symbols = contexts[0].symbols.copy()
    for c in contexts[1:]:
        genes = genes.append(c.genes.difference(genes, sort=False))
        symbols = pd.concat([symbols, c.symbols[~c.symbols.index.isin(symbols.index)]])
    genes = pd.Index(genes, name="gene_id")
    return [c.reindex_genes(genes, symbols) for c in contexts]
