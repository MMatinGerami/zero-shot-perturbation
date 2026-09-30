"""Round 8: predicting a knockdown's response from the target line's own unperturbed variation.

Fluctuation-response relation, first order: in a linear stochastic system near steady state,
the response of gene g to a small push on gene p is proportional to Cov(g, p) / Var(p) in the
unperturbed state. Here the push is a CRISPRi knockdown of p (depth d, log2 units), so

    fr(g, p) = beta_gp * d,    beta_gp = Cov(g, p) / Var(p)

estimated on the held-out line's control cells. Pre-registration, including the reasons it may
fail: results/prereg/fluctuation_response.md.

Estimators (chosen on source lines only, never on the held-out line):
  "raw"      regression on log1p(CP10k)
  "libsize"  the same after regressing log library size out of every gene (cell size is the
             largest shared axis of single-cell variation and says nothing about p)
  "knn"      "libsize" on cells averaged with their k nearest neighbours in PCA space
             (metacell-style smoothing; dropout noise in single cells biases covariances to 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.models import NormRestoredTransfer, Prediction, WeightedTransfer

ESTIMATORS = ("raw", "libsize", "knn")


def normalised_cells(
    counts, estimator: str = "libsize", k: int = 10, n_pcs: int = 50, seed: int = 0
) -> np.ndarray:
    """Centred cells x genes matrix on which covariances are taken."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}")
    X = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib = np.maximum(lib, 1.0)
    Z = np.log1p(np.asarray(X.multiply(1e4 / lib[:, None]).todense(), dtype=np.float32))
    Z -= Z.mean(axis=0)
    if estimator == "raw":
        return Z
    lz = (np.log(lib) - np.log(lib).mean()).astype(np.float32)
    Z -= np.outer(lz, (lz @ Z) / (lz @ lz))
    if estimator == "libsize":
        return Z
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    n_pcs = min(n_pcs, min(Z.shape) - 1)
    pcs = PCA(n_components=n_pcs, random_state=seed).fit_transform(Z)
    nn = NearestNeighbors(n_neighbors=min(k, Z.shape[0])).fit(pcs)
    idx = nn.kneighbors(pcs, return_distance=False)
    n, kk = idx.shape
    W = sp.csr_matrix(
        (np.full(n * kk, 1.0 / kk, np.float32), (np.repeat(np.arange(n), kk), idx.ravel())),
        shape=(n, n),
    )
    S = np.asarray(W @ Z, dtype=np.float32)  # each cell replaced by its neighbourhood mean
    return S - S.mean(axis=0)


def regression_columns(Z: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """beta[i, g] = Cov(g, cols[i]) / Var(cols[i]); rows of zeros where the column is constant."""
    C = Z[:, cols]
    var = (C * C).sum(axis=0)
    cov = C.T @ Z  # (len(cols), genes)
    with np.errstate(all="ignore"):
        beta = np.where(var[:, None] > 0, cov / var[:, None], 0.0)
    return beta


def fr_prediction(
    Z: np.ndarray,
    symbols: np.ndarray,
    perts: list[str],
    depth: float,
    usable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """fr LFC for every perturbation (perts x genes) and a mask of the perturbations it applies
    to: the target must be on the gene axis and `usable` (expressed enough to have a variance).
    The target's own gene is set to the knockdown depth."""
    pos = {s: i for i, s in enumerate(symbols)}
    out = np.zeros((len(perts), Z.shape[1]))
    ok = np.array([p in pos and usable[pos[p]] for p in perts])
    if ok.any():
        cols = np.array([pos[p] for p, o in zip(perts, ok, strict=True) if o])
        beta = regression_columns(Z, cols)
        out[ok] = beta * depth
        out[np.flatnonzero(ok), cols] = depth
    return out, ok


@dataclass
class FluctuationBlend:
    """Transfer prediction + lambda * fluctuation-response prediction, for targets the held-out
    line expresses at `min_cp10k` or more; other targets keep the transfer prediction.

    The benchmark hands the model the basal half of the held-out control cells through
    `set_basal_cells(counts, gene_ids)`; without it the model is its base transfer."""

    base: str = "norm_restored"
    lam: float = 0.5
    estimator: str = "libsize"
    temperature: float = 0.1
    min_cp10k: float = 1.0
    _cells: tuple | None = field(default=None, repr=False)

    def set_basal_cells(self, counts, gene_ids) -> None:
        self._cells = (counts, pd.Index(gene_ids))

    def _base(self):
        cls = {"norm_restored": NormRestoredTransfer, "weighted": WeightedTransfer}[self.base]
        return cls(temperature=self.temperature)

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        from zsp.models import knockdown_depth

        pred = self._base()(sources, target_basal, perts, genes, symbols)
        if self._cells is None or self.lam == 0:
            return pred
        counts, cell_genes = self._cells
        Z = normalised_cells(counts, self.estimator)
        cp = np.asarray(counts.mean(axis=0)).ravel()
        cp = cp / cp.sum() * 1e4
        sym = np.asarray(
            pd.Series(np.asarray(symbols), index=pd.Index(genes)).reindex(cell_genes)
        ).astype(str)
        depth = -float(np.median([knockdown_depth(s) for s in sources]))
        fr, ok = fr_prediction(Z, sym, list(perts), depth, cp >= self.min_cp10k)
        cols = pd.Index(genes).get_indexer(cell_genes)
        lfc = pred.lfc.to_numpy().copy()
        rows = np.flatnonzero(ok)
        lfc[np.ix_(rows, cols[cols >= 0])] += self.lam * fr[np.ix_(rows, np.flatnonzero(cols >= 0))]
        return Prediction(pd.DataFrame(lfc, index=pred.lfc.index, columns=pred.lfc.columns))
