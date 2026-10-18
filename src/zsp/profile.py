"""The scorer's pooled-profile space, and what a prediction can set in it before emission.

`cell-eval2` scores expression on pooled profiles: counts summed over a group's cells,
normalised to a fixed total and log-transformed. Its expression error subtracts each side's
sampling noise, estimated by a delete-one-cell jackknife, and caps the credit a submission
may claim in total by the across-perturbation spread of its own profiles (the `rho` term of
the vcc2026 specification). Two consequences are used here:

* a prediction is best rendered by cells whose pooled profile *is* the prediction
  (`zsp.emit.emit_cells_exact`), so that no sampling noise is added to it;
* the amplitude of the perturbation-specific part of the prediction can be set so that the
  spread just unlocks the full credit (`calibrate_amplitude`), a choice that reads nothing
  but the prediction and the context's control cells.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

TS = 5e4  # the scorer's profile normalisation total


def profile(counts: np.ndarray) -> np.ndarray:
    """log1p of total-normalised pooled profiles; rows are groups (or mean counts per cell,
    which pool to the same profile)."""
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    return np.log1p(TS * counts / counts.sum(1, keepdims=True))


def jackknife(X: sp.csr_matrix, chunk: int = 1000) -> float:
    """Delete-one-cell jackknife of a group's pooled profile (the specification's C)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    P = np.asarray(X.sum(0)).ravel().astype(np.float64)
    S = P.sum()
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    s1 = np.zeros_like(P)
    s2 = np.zeros_like(P)
    for start in range(0, n, chunk):
        Y = X[start : start + chunk].toarray().astype(np.float64)
        V = np.log1p(TS * (P[None, :] - Y) / (S - lib[start : start + chunk])[:, None])
        s1 += V.sum(0)
        s2 += (V**2).sum(0)
    return float((n - 1) / n * (s2 - s1**2 / n).sum())


def emitted_group_noise(
    control_cells: sp.csr_matrix, n_cells: int, rng: np.random.Generator, draws: int = 5
) -> float:
    """The sampling noise (jackknife C) of a group of `n_cells` cells resampled from the
    context's control cells: what an emitted group of that size claims, averaged over a
    few draws. The predicted fold changes barely move it, so it is estimated once per
    context at ratio 1."""
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, control_cells.shape[0], n_cells)
        vals.append(jackknife(sp.csr_matrix(control_cells[idx])))
    return float(np.mean(vals))


def calibrate_amplitude(
    means: np.ndarray,
    control_mean: np.ndarray,
    credit_total: float,
    rho_target: float = 1.0,
    generic_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Rescale the perturbation-specific part of predicted mean profiles.

    `means` are mean counts per cell (perturbations x genes); `credit_total` is the sum over
    perturbations of each emitted group's sampling noise divided by the number of scored
    genes (what the submission can claim). The spread B of the predicted profiles about
    their mean is set to `rho_target` times that credit, by one amplitude on the
    perturbation-centred part; the mean response (generic part) is kept, or multiplied by
    `generic_scale` when given. Returns the new means and the amplitude applied.
    """
    b_basal = profile(control_mean)[0]
    delta = profile(means) - b_basal
    gen = delta.mean(0, keepdims=True)
    if generic_scale is not None:
        delta = generic_scale * gen + (delta - gen)
        gen = delta.mean(0, keepdims=True)
    G = means.shape[1] - 1
    B = ((delta - gen) ** 2).sum() / G
    amplitude = float(np.sqrt(rho_target * credit_total / B)) if B > 0 else 1.0
    delta = gen + amplitude * (delta - gen)
    out = np.expm1(b_basal + delta)
    out = out / out.sum(1, keepdims=True) * control_mean.sum()
    return out, amplitude
