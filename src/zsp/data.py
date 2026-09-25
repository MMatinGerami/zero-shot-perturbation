"""Turning public Perturb-seq screens into per-perturbation pseudobulk profiles.

Every context is reduced to the same representation, a `Context`:
  * `control`  - mean raw UMI counts per cell of non-targeting cells (the basal state);
  * `pert`     - mean raw UMI counts per cell for each perturbation target;
  * `n_cells`  - cells contributing to each perturbation mean.
Genes are Ensembl IDs, so contexts can be aligned by intersection.
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


def cp10k(x: np.ndarray) -> np.ndarray:
    """Library-size normalise mean counts (rows or a single vector) to counts per 10k."""
    x = np.atleast_2d(x)
    return x / x.sum(1, keepdims=True) * 1e4


def log_fold_change(pert: np.ndarray, control: np.ndarray, pseudo: float = 0.1) -> np.ndarray:
    """log2 fold change of CP10k-normalised pseudobulks against the control."""
    return np.log2(cp10k(pert) + pseudo) - np.log2(cp10k(control) + pseudo)


def align(contexts: list[Context]) -> list[Context]:
    """Restrict all contexts to their shared genes (order of the first context)."""
    shared = contexts[0].genes
    for c in contexts[1:]:
        shared = shared.intersection(c.genes, sort=False)
    return [c.subset_genes(shared) for c in contexts]
