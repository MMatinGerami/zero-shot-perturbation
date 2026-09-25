"""Zero-shot transfer of perturbation effects to an unseen cell context.

All models work in log2-fold-change space and return a predicted LFC matrix
(perturbations x genes) for the target context. The target context contributes only what
the challenge provides: its basal (non-targeting) expression profile.

A knockdown's measured response in context s is decomposed as

    LFC_s(p) = G_s + D_s(p)

where G_s is the *generic* response shared by (almost) all knockdowns in that context -
stress, slowed proliferation, the cost of CRISPRi itself - and D_s(p) is what is specific to
target p. The generic part is the thing a naive transfer gets most wrong, because it differs
between cell lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from zsp.data import Context, cp10k, log_fold_change


@dataclass
class SourceEffects:
    """Per-source decomposition, precomputed once."""

    name: str
    lfc: pd.DataFrame  # perts x genes
    generic: np.ndarray  # (genes,)
    specific: pd.DataFrame  # perts x genes  (lfc - generic)
    basal: np.ndarray  # CP10k control
    own_lfc: pd.Series  # target gene's own LFC per perturbation (knockdown depth)


def decompose(ctx: Context, min_cells: int = 10) -> SourceEffects:
    keep = ctx.n_cells.reindex(ctx.pert.index).fillna(0) >= min_cells
    pert = ctx.pert.loc[keep]
    lfc = pd.DataFrame(
        log_fold_change(pert.to_numpy(), ctx.control),
        index=pert.index,
        columns=ctx.genes,
    )
    generic = np.median(lfc.to_numpy(), axis=0)
    specific = lfc - generic
    sym_to_gene = pd.Series(ctx.genes, index=ctx.symbols.loc[ctx.genes].to_numpy())
    sym_to_gene = sym_to_gene[~sym_to_gene.index.duplicated()]
    own = {}
    for p in lfc.index:
        if p in sym_to_gene.index:
            own[p] = lfc.at[p, sym_to_gene[p]]
    return SourceEffects(ctx.name, lfc, generic, specific, cp10k(ctx.control)[0], pd.Series(own))


@dataclass
class Prediction:
    lfc: pd.DataFrame  # perts x genes
    uncertainty: pd.Series = field(
        default_factory=pd.Series
    )  # per perturbation, higher = less sure


def _stack(sources: list[SourceEffects], perts: list[str], attr: str) -> np.ndarray:
    """(n_sources, n_perts, n_genes) with NaN where a source lacks a perturbation."""
    n_genes = len(sources[0].generic)
    out = np.full((len(sources), len(perts), n_genes), np.nan)
    for i, s in enumerate(sources):
        df = getattr(s, attr)
        present = [p for p in perts if p in df.index]
        idx = [perts.index(p) for p in present]
        out[i, idx] = df.loc[present].to_numpy()
    return out


def predict_control(sources, target_basal, perts, genes, symbols=None) -> Prediction:
    return Prediction(pd.DataFrame(0.0, index=perts, columns=genes))


def predict_mean_transfer(sources, target_basal, perts, genes, symbols=None) -> Prediction:
    """Average each target's measured LFC over sources; unseen targets get the generic response."""
    lfc = _stack(sources, perts, "lfc")
    generic = np.mean([s.generic for s in sources], axis=0)
    pred = np.nanmean(lfc, axis=0) if np.isfinite(lfc).any() else np.zeros_like(lfc[0])
    missing = ~np.isfinite(pred).any(1)
    pred[missing] = generic
    return Prediction(pd.DataFrame(np.nan_to_num(pred), index=perts, columns=genes))


@dataclass
class CalibratedTransfer:
    """Generic + shrunk specific effect, gated by target basal expression.

    * generic   : mean of source generic responses
    * specific  : mean source-specific effect x alpha (shrinkage for cross-context decay)
    * gating    : genes barely expressed in the target cannot show measurable change
    * self      : a target's own gene is set to the typical knockdown depth if expressed
    alpha / gate are fitted on the sources only (leave-one-source-out), never on the target.
    """

    alpha: float = 0.6
    gate_cp10k: float = 0.05

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        spec = _stack(sources, perts, "specific")
        generic = np.mean([s.generic for s in sources], axis=0)
        with np.errstate(all="ignore"):
            d_mean = np.nanmean(spec, axis=0)
            d_std = np.nanstd(spec, axis=0)
        n_src = np.isfinite(spec[:, :, 0]).sum(0)
        d_mean = np.nan_to_num(d_mean)
        pred = generic[None, :] + self.alpha * d_mean
        gate = target_basal >= self.gate_cp10k
        pred[:, ~gate] = 0.0
        own_depth = np.nanmedian(np.concatenate([s.own_lfc.to_numpy() for s in sources]))
        names = list(symbols) if symbols is not None else list(genes)
        gene_pos = {g: i for i, g in enumerate(names)}
        for i, p in enumerate(perts):
            j = gene_pos.get(p)
            if j is not None and gate[j]:
                pred[i, j] = own_depth
        # uncertainty: source disagreement on the specific effect, inflated when few sources saw p
        with np.errstate(all="ignore"):
            disagreement = np.nanmean(np.nan_to_num(d_std) ** 2, axis=1)
        unc = disagreement + 1.0 / np.maximum(n_src, 0.5)
        return Prediction(
            pd.DataFrame(pred, index=perts, columns=genes), pd.Series(unc, index=perts)
        )


def lfc_to_counts(lfc: np.ndarray, control_counts: np.ndarray, pseudo: float = 0.1) -> np.ndarray:
    """Invert the LFC around the target control, returning mean counts per cell."""
    basal = cp10k(control_counts)[0]
    cp = np.clip(np.exp2(lfc) * (basal + pseudo) - pseudo, 0, None)
    lib = control_counts.sum()
    return cp / 1e4 * lib
