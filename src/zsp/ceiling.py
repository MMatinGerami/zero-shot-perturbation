"""Noise ceiling: how well could any prediction agree with a held-out line's measured response?

A measured pseudobulk response is itself an estimate from ~50 to 100 cells. Split each
perturbation's cells (and the control cells) into two disjoint halves, compute the log fold
change in each half, and correlate the halves gene by gene. The Spearman-Brown formula turns
that split-half correlation into the reliability of the full-sample response, and the square
root of the reliability is the highest gene-wise correlation a noise-free prediction can reach
against it. A model's correlation divided by this ceiling is the share of the *reachable*
direction it recovers.

Two versions of every correlation are reported:

  raw      over the LFC as measured;
  centred  after subtracting, gene by gene, the mean LFC over all evaluated perturbations
           (the systematic response of Viñas Torné et al., Nat Biotechnol 2025). The challenge
           scores are anchored at a mean-response baseline, so the centred numbers measure
           the part of the response that actually earns points.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from zsp.data import cp10k, log_fold_change


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching rows of two (n, genes) arrays; NaN for a constant row."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(all="ignore"):
        return np.where(den > 0, (a * b).sum(1) / den, np.nan)


def spearman_brown(r_half: np.ndarray) -> np.ndarray:
    """Reliability of the full sample from the correlation between its two halves."""
    r = np.clip(np.asarray(r_half, dtype=np.float64), -0.999, 0.999)
    return np.where(r > 0, 2 * r / (1 + r), 0.0)


def _mean_rows(X, idx: np.ndarray) -> np.ndarray:
    return np.asarray(X[idx].mean(axis=0)).ravel()


def split_half_lfc(
    X, target: np.ndarray, perts: list[str], rng: np.random.Generator, control: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two independent LFC estimates per perturbation, each against its own half of the
    control cells, plus the full-sample control mean. X: cells x genes counts."""
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    ctrl = np.flatnonzero(target == control)
    rng.shuffle(ctrl)
    c_a, c_b = _mean_rows(X, ctrl[::2]), _mean_rows(X, ctrl[1::2])
    half_a, half_b = [], []
    for p in perts:
        idx = np.flatnonzero(target == p)
        rng.shuffle(idx)
        half_a.append(_mean_rows(X, idx[::2]))
        half_b.append(_mean_rows(X, idx[1::2]))
    lfc_a = log_fold_change(np.vstack(half_a), c_a)
    lfc_b = log_fold_change(np.vstack(half_b), c_b)
    return lfc_a, lfc_b, _mean_rows(X, ctrl)


def ceiling_table(lfc_a: np.ndarray, lfc_b: np.ndarray) -> dict[str, np.ndarray]:
    """Per-perturbation split-half r, reliability and ceiling, raw and centred."""
    out = {}
    for kind, (a, b) in {
        "raw": (lfc_a, lfc_b),
        "centred": (lfc_a - lfc_a.mean(0), lfc_b - lfc_b.mean(0)),
    }.items():
        r = rowwise_pearson(a, b)
        rel = spearman_brown(r)
        out[f"split_half_r_{kind}"] = r
        out[f"reliability_{kind}"] = rel
        out[f"ceiling_{kind}"] = np.sqrt(rel)
    return out


def expressed_mask(control_mean: np.ndarray, min_cp10k: float = 1.0) -> np.ndarray:
    return cp10k(control_mean)[0] >= min_cp10k
