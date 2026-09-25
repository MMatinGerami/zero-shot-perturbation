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
