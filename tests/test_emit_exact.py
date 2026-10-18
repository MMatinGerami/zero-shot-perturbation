import numpy as np
import scipy.sparse as sp

from zsp.emit import emit_cells, emit_cells_exact


def _controls(rng, n=300, g=50):
    lam = rng.gamma(0.5, 4.0, size=g)
    return sp.csr_matrix(rng.poisson(lam[None, :] * rng.gamma(2.0, 0.5, size=(n, 1))))


def test_exact_emitter_matches_predicted_totals_and_keeps_scatter():
    rng = np.random.default_rng(0)
    ctrl = _controls(rng)
    cm = np.asarray(ctrl.mean(0)).ravel()
    pred = cm * rng.uniform(0.5, 1.5, size=cm.size)
    pred[cm == 0] = 0
    X = emit_cells_exact(ctrl, cm, pred, 120, rng)
    sums = np.asarray(X.sum(0)).ravel()
    assert np.array_equal(sums, np.rint(120 * pred))
    assert (X.data >= 0).all() and np.allclose(X.data, np.round(X.data))
    Y = emit_cells(ctrl, cm, pred, 120, rng)
    # within-group dispersion is preserved (same order of magnitude as the iid emitter)
    fano_x = (X.toarray().var(0) / np.maximum(X.toarray().mean(0), 1e-9)).mean()
    fano_y = (Y.toarray().var(0) / np.maximum(Y.toarray().mean(0), 1e-9)).mean()
    assert 0.7 < fano_x / fano_y < 1.3


def test_exact_emitter_fills_genes_absent_from_sampled_cells():
    rng = np.random.default_rng(1)
    ctrl = sp.csr_matrix(np.zeros((40, 3)))
    ctrl[0, 0] = 5  # only one control cell carries gene 0
    cm = np.asarray(ctrl.mean(0)).ravel()
    pred = np.array([0.5, 0.0, 0.0])
    X = emit_cells_exact(ctrl, cm, pred, 10, rng)
    assert X.sum() == 5
