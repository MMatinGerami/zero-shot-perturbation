"""Turning a predicted mean profile into single cells, as the challenge requires.

The challenge scores cells, not pseudobulks, so every prediction is emitted by resampling the
target context's own control cells and rescaling each gene by the predicted fold change
(stochastic rounding keeps counts integer). Cell-to-cell heterogeneity therefore comes from
real basal cells of that context, not from a model assumption.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def emit_cells(
    control_cells: sp.csr_matrix,
    control_mean: np.ndarray,
    pred_mean: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    ratio = np.divide(pred_mean, control_mean, out=np.ones_like(pred_mean), where=control_mean > 0)
    idx = rng.integers(0, control_cells.shape[0], n_cells)
    X = control_cells[idx].toarray().astype(np.float64) * ratio[None, :]
    floor = np.floor(X)
    X = floor + (rng.random(X.shape) < (X - floor))
    return sp.csr_matrix(X.astype(np.float32))


def emit_cells_exact(
    control_cells: sp.csr_matrix,
    control_mean: np.ndarray,
    pred_mean: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Like `emit_cells`, but the emitted group's pooled profile equals the prediction.

    Resampling control cells adds sampling noise to the pooled profile of the group, which
    is not part of the prediction: a mean prediction is best rendered by cells whose sum is
    that mean. Cells are drawn and rescaled as in `emit_cells`, every gene's column is then
    rescaled to the predicted total, and counts are made integer gene by gene so that each
    column sums to the rounded predicted total exactly (the extra units go to the cells with
    the largest fractional parts). Cell-to-cell heterogeneity is still the context's own.
    """
    ratio = np.divide(pred_mean, control_mean, out=np.ones_like(pred_mean), where=control_mean > 0)
    idx = rng.integers(0, control_cells.shape[0], n_cells)
    X = control_cells[idx].toarray().astype(np.float64) * ratio[None, :]
    target = np.rint(n_cells * np.asarray(pred_mean, dtype=np.float64))
    col = X.sum(0)
    scale = np.divide(target, col, out=np.zeros_like(target), where=col > 0)
    X *= scale[None, :]
    # genes the prediction expects but no sampled cell carries: spread the units at random
    empty = (col == 0) & (target > 0)
    for g in np.flatnonzero(empty):
        X[rng.integers(0, n_cells, int(target[g])), g] += 1.0
    floor = np.floor(X)
    frac = X - floor
    short = (target - floor.sum(0)).astype(np.int64)
    short = np.clip(short, 0, n_cells)
    # rank fractional parts per gene; the `short[g]` largest get one more unit
    order = np.argsort(-frac, axis=0, kind="stable")
    rank = np.empty_like(order)
    rank[order, np.arange(X.shape[1])[None, :]] = np.arange(n_cells)[:, None]
    floor += rank < short[None, :]
    return sp.csr_matrix(floor.astype(np.float32))
