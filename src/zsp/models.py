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

from dataclasses import dataclass, field, replace

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
    with np.errstate(all="ignore"):
        generic = np.nanmedian(lfc.to_numpy(), axis=0)  # NaN where the context lacks the gene
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
    with np.errstate(all="ignore"):
        generic = np.nanmean([s.generic for s in sources], axis=0)
        pred = np.nanmean(lfc, axis=0) if np.isfinite(lfc).any() else np.zeros_like(lfc[0])
    missing = ~np.isfinite(pred).any(1)  # target measured in no source -> generic response
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
        with np.errstate(all="ignore"):
            generic = np.nan_to_num(np.nanmean([s.generic for s in sources], axis=0))
            d_mean = np.nanmean(spec, axis=0)
            d_std = np.nanstd(spec, axis=0)
        n_src = np.isfinite(spec).any(axis=2).sum(0)
        d_mean = np.nan_to_num(d_mean)
        pred = generic[None, :] + self.alpha * d_mean
        gate = np.nan_to_num(target_basal) >= self.gate_cp10k
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


# --------------------------------------------------------------------------------------
# Context-dependent transfer: which sources are informative for this basal state, and how
# large should the transferred response be?


def basal_similarity(target_basal: np.ndarray, sources: list[SourceEffects]) -> np.ndarray:
    """Pearson correlation of log1p(CP10k) basal profiles, target vs each source, over the
    genes both measured. Lies in [-1, 1]; a proxy for 'how much like this line is that one'."""
    t = np.log1p(np.asarray(target_basal, dtype=np.float64))
    out = np.zeros(len(sources))
    for i, s in enumerate(sources):
        b = np.log1p(s.basal)
        ok = np.isfinite(t) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        out[i] = np.corrcoef(t[ok], b[ok])[0, 1]
    return out


def _weighted_transfer(sources, target_basal, perts, weights) -> np.ndarray:
    """Weighted NaN-aware mean of source LFCs; targets no source measured get the weighted
    generic response. `weights` are per source and renormalised per (pert, gene) over the
    sources that actually measured that entry."""
    lfc = _stack(sources, perts, "lfc")
    w = np.asarray(weights, dtype=np.float64)[:, None, None]
    present = np.isfinite(lfc)
    num = np.nansum(np.where(present, lfc, 0.0) * w, axis=0)
    den = (present * w).sum(axis=0)
    with np.errstate(all="ignore"):
        pred = num / den
    gen = np.stack([s.generic for s in sources])
    gen_ok = np.isfinite(gen)
    with np.errstate(all="ignore"):
        generic = np.nansum(np.where(gen_ok, gen, 0.0) * w[:, 0, :], axis=0) / (
            gen_ok * w[:, 0, :]
        ).sum(axis=0)
    missing = ~np.isfinite(pred).any(axis=1)
    pred[missing] = generic
    return np.nan_to_num(pred)


@dataclass
class WeightedTransfer:
    """Mean transfer where each source is weighted by softmax(similarity / temperature).
    temperature -> inf recovers plain mean transfer; -> 0 copies the most similar source."""

    temperature: float = 0.1

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature)
        pred = _weighted_transfer(sources, target_basal, perts, w / w.sum())
        return Prediction(pd.DataFrame(pred, index=perts, columns=genes))


@dataclass
class ScaledTransfer:
    """Mean transfer times one global response scale. Tests the overshoot hypothesis: a
    source line's response magnitude need not be the new line's."""

    scale: float = 1.0

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        base = predict_mean_transfer(sources, target_basal, perts, genes, symbols)
        return Prediction(base.lfc * self.scale)


@dataclass
class GeneScaledTransfer:
    """Mean transfer with a per-gene transfer coefficient beta_g, learned from the sources
    only: each source in turn is predicted from the others, and beta_g is the ridge-shrunk
    (toward 1) slope of truth on prediction pooled over those pseudo-targets. Genes whose
    responses transfer between lines keep beta ~ 1; genes that do not are damped.
    `lam` is the ridge strength relative to the median per-gene sum of squares."""

    lam: float = 1.0
    beta: np.ndarray | None = None

    def fit(self, sources) -> GeneScaledTransfer:
        n_genes = len(sources[0].generic)
        xy, xx = np.zeros(n_genes), np.zeros(n_genes)
        for i, tgt in enumerate(sources):
            others = [s for j, s in enumerate(sources) if j != i]
            if not others:
                continue
            perts = [p for p in tgt.lfc.index if any(p in s.lfc.index for s in others)]
            if not perts:
                continue
            pred = _weighted_transfer(others, None, perts, np.ones(len(others)))
            truth = tgt.lfc.loc[perts].to_numpy()
            ok = np.isfinite(truth)
            xy += np.where(ok, pred * truth, 0.0).sum(axis=0)
            xx += np.where(ok, pred * pred, 0.0).sum(axis=0)
        ridge = self.lam * np.median(xx[xx > 0]) if (xx > 0).any() else 1.0
        self.beta = (xy + ridge) / (xx + ridge)
        return self

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        if self.beta is None:
            self.fit(sources)
        base = predict_mean_transfer(sources, target_basal, perts, genes, symbols)
        return Prediction(base.lfc * self.beta[None, :])


# --------------------------------------------------------------------------------------
# Candidate uncertainty signals, all computable from the sources and the basal state alone.


def uncertainty_scores(sources, target_basal, perts) -> pd.DataFrame:
    """One row per perturbation, one column per candidate signal (higher = less certain):

    disagreement       mean over genes of the variance of the specific effect across sources
    disagreement_norm  the same divided by the mean squared transferred effect (size-free)
    n_sources_inv      1 / number of sources that measured the target
    effect_magnitude   mean |transferred LFC| (a *confidence* signal only if big effects are
                       easier; included so the confound is measured, not assumed away)
    basal_distance     1 - mean basal similarity of the sources that measured the target
    """
    spec = _stack(sources, perts, "specific")
    lfc = _stack(sources, perts, "lfc")
    sim = basal_similarity(target_basal, sources)
    with np.errstate(all="ignore"):
        var = np.nanvar(spec, axis=0)
        disagreement = np.nanmean(var, axis=1)
        mean_lfc = np.nanmean(lfc, axis=0)
        magnitude = np.nanmean(np.abs(mean_lfc), axis=1)
        msq = np.nanmean(mean_lfc**2, axis=1)
    seen = np.isfinite(spec).any(axis=2)  # (sources, perts)
    n_src = seen.sum(axis=0)
    with np.errstate(all="ignore"):
        dist = 1 - (seen * sim[:, None]).sum(axis=0) / np.maximum(n_src, 1)
    return pd.DataFrame(
        {
            "disagreement": np.nan_to_num(disagreement),
            "disagreement_norm": np.nan_to_num(disagreement / (msq + 1e-6)),
            "n_sources_inv": 1.0 / np.maximum(n_src, 0.5),
            "effect_magnitude": np.nan_to_num(magnitude),
            "basal_distance": np.where(n_src > 0, dist, 1.0),
        },
        index=perts,
    )


# --------------------------------------------------------------------------------------
# Round 4: averaging sources shrinks response magnitude (sources disagree on the sign of
# many genes, so their mean is closer to zero than any of them), and the DE-based challenge
# metrics need magnitude. Two aggregations that keep the sign consensus of a weighted mean
# but do not lose scale.


@dataclass
class NormRestoredTransfer:
    """Weighted transfer, then each target's response is rescaled so that its L2 norm equals
    the weighted mean of the source responses' norms. The direction is the consensus, the
    size is what a single source typically shows. `temperature` as in WeightedTransfer."""

    temperature: float = 0.1

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature)
        w = w / w.sum()
        pred = _weighted_transfer(sources, target_basal, perts, w)
        lfc = _stack(sources, perts, "lfc")
        with np.errstate(all="ignore"):
            norms = np.sqrt(np.nansum(np.nan_to_num(lfc) ** 2, axis=2))  # (sources, perts)
            seen = np.isfinite(lfc).any(axis=2)
            target_norm = (norms * seen * w[:, None]).sum(0) / np.maximum(
                (seen * w[:, None]).sum(0), 1e-12
            )
            own = np.linalg.norm(pred, axis=1)
            scale = np.where(own > 0, target_norm / own, 1.0)
        scale = np.where(seen.any(0), scale, 1.0)  # unseen targets keep the generic response
        return Prediction(pd.DataFrame(pred * scale[:, None], index=perts, columns=genes))


@dataclass
class MedianTransfer:
    """Per-gene weighted median of the source responses (weights from basal similarity).
    A median keeps the magnitude of the typical source instead of averaging towards zero."""

    temperature: float = 0.1

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature)
        w = w / w.sum()
        lfc = _stack(sources, perts, "lfc")  # (sources, perts, genes)
        pred = _weighted_median(lfc, w)
        base = _weighted_transfer(sources, target_basal, perts, w)  # generic for unseen
        missing = ~np.isfinite(lfc).any(axis=(0, 2))
        pred[missing] = base[missing]
        return Prediction(pd.DataFrame(np.nan_to_num(pred), index=perts, columns=genes))


def _weighted_median(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted median over axis 0 of a (sources, perts, genes) array with NaN gaps."""
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    ws = np.take_along_axis(np.broadcast_to(w[:, None, None], x.shape), order, axis=0)
    ws = np.where(np.isfinite(xs), ws, 0.0)
    cum = np.cumsum(ws, axis=0)
    total = cum[-1]
    with np.errstate(all="ignore"):
        hit = cum >= 0.5 * total[None]
    idx = np.argmax(hit, axis=0)
    out = np.take_along_axis(xs, idx[None], axis=0)[0]
    return np.where(total > 0, out, np.nan)


def knockdown_depth(source: SourceEffects, floor: float = 0.1) -> float:
    """How deep a screen's knockdowns are: minus the median LFC of each target's own transcript,
    in log2 units (1.3 to 1.9 for the Replogle and Nadig screens, 0.3 to 0.4 for X-Atlas).
    Floored so that a screen with no own-target information cannot blow up a division."""
    if len(source.own_lfc) == 0 or not np.isfinite(source.own_lfc).any():
        return 1.0
    return max(float(-np.nanmedian(source.own_lfc)), floor)


@dataclass
class DepthAwareTransfer:
    """Weighted transfer that accounts for how deep each screen's knockdowns are.

    Two independent adjustments, both from `knockdown_depth`:
      * `depth_weight` k: the basal-similarity weight of each source is multiplied by
        depth**k, so shallow-knockdown screens count less (k = 0 changes nothing);
      * `normalise`: every source response is divided by its depth (response per log2 unit
        of knockdown) before averaging, and the consensus is multiplied by the sources'
        weighted mean depth, so a shallow screen's small response is read as a shallow
        knockdown rather than as a weak effect.
    With k = 0 and normalise = False this is WeightedTransfer."""

    temperature: float = 0.1
    depth_weight: float = 0.0
    normalise: bool = False

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        depth = np.array([knockdown_depth(s) for s in sources])
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature) * depth**self.depth_weight
        w = w / w.sum()
        if self.normalise:
            per_unit = [
                replace(s, lfc=s.lfc / d, generic=s.generic / d)
                for s, d in zip(sources, depth, strict=True)
            ]
            pred = _weighted_transfer(per_unit, target_basal, perts, w) * float((w * depth).sum())
        else:
            pred = _weighted_transfer(sources, target_basal, perts, w)
        return Prediction(pd.DataFrame(pred, index=perts, columns=genes))


@dataclass
class BasalModulatedTransfer:
    """Weighted transfer whose per-gene response is modulated by how the target line's basal
    expression of that gene differs from the sources'.

    delta_g = log1p(CP10k_target,g) - weighted mean over sources of log1p(CP10k_source,g).
    Genes are binned by delta; each bin gets one multiplier, the least-squares coefficient of
    the true response on the consensus response inside that bin, learned on the sources
    alone by leaving each source out in turn (`fit`). The multipliers are normalised to a
    count-weighted mean of 1, so the model changes the pattern of the response across genes
    (a gene the target line barely expresses cannot fall much; a gene it expresses far more
    than the sources may respond more) and never the overall size. `restore` then sets the
    size: "none" keeps whatever the modulation left, "consensus" restores each target's
    norm to that of plain weighted transfer, "source" to the weighted mean norm of the
    source responses (as NormRestoredTransfer)."""

    temperature: float = 0.1
    restore: str = "none"
    edges: tuple = (-np.inf, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, np.inf)
    max_perts: int = 2000
    multipliers_: np.ndarray | None = field(default=None, repr=False)

    def _weights(self, target_basal, sources) -> np.ndarray:
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature)
        return w / w.sum()

    @staticmethod
    def _delta(target_basal, sources, w) -> np.ndarray:
        t = np.log1p(np.asarray(target_basal, dtype=np.float64))
        b = np.stack([np.log1p(s.basal) for s in sources])
        ok = np.isfinite(b)
        with np.errstate(all="ignore"):
            ref = np.nansum(np.where(ok, b, 0.0) * w[:, None], axis=0) / (ok * w[:, None]).sum(0)
        return t - ref  # NaN where the target or every source lacks the gene

    def _bins(self, delta: np.ndarray) -> np.ndarray:
        return np.digitize(np.nan_to_num(delta), np.asarray(self.edges[1:-1]))

    def fit(self, sources) -> BasalModulatedTransfer:
        n_bins = len(self.edges) - 1
        num, den, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        rng = np.random.default_rng(0)
        for i, tgt in enumerate(sources):
            others = sources[:i] + sources[i + 1 :]
            if not others:
                continue
            perts = [p for p in tgt.lfc.index if any(p in o.lfc.index for o in others)]
            if len(perts) > self.max_perts:
                perts = sorted(rng.choice(perts, self.max_perts, replace=False))
            if not perts:
                continue
            w = self._weights(tgt.basal, others)
            c = _weighted_transfer(others, tgt.basal, perts, w)
            y = tgt.lfc.loc[perts].to_numpy()
            delta = self._delta(tgt.basal, others, w)
            bins = self._bins(delta)
            ok = np.isfinite(y) & np.isfinite(c) & np.isfinite(delta)[None]
            for b in range(n_bins):
                m = ok & (bins == b)[None]
                num[b] += float((y * c)[m].sum())
                den[b] += float((c * c)[m].sum())
                cnt[b] += int(m.sum())
        mult = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
        if cnt.sum() > 0 and (mult * cnt).sum() > 0:
            mult = mult / (mult * cnt).sum() * cnt.sum()  # relative modulation only
        self.multipliers_ = mult
        return self

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        if self.multipliers_ is None:
            self.fit(sources)
        w = self._weights(target_basal, sources)
        pred = _weighted_transfer(sources, target_basal, perts, w)
        delta = self._delta(target_basal, sources, w)
        mult = np.where(np.isfinite(delta), self.multipliers_[self._bins(delta)], 1.0)
        out = pred * mult[None, :]
        with np.errstate(all="ignore"):
            own = np.linalg.norm(out, axis=1)
            if self.restore == "consensus":
                target = np.linalg.norm(pred, axis=1)
            elif self.restore == "source":
                lfc = _stack(sources, perts, "lfc")
                norms = np.sqrt(np.nansum(np.nan_to_num(lfc) ** 2, axis=2))
                seen = np.isfinite(lfc).any(axis=2)
                target = (norms * seen * w[:, None]).sum(0) / np.maximum(
                    (seen * w[:, None]).sum(0), 1e-12
                )
                target = np.where(seen.any(0), target, own)
            else:
                target = own
            scale = np.where(own > 0, target / own, 1.0)
        return Prediction(pd.DataFrame(out * scale[:, None], index=perts, columns=genes))


@dataclass
class AgreementTransfer:
    """Weighted transfer that stops averaging where the sources disagree.

    For each (target, gene), agreement = |sum_s w_s sign(lfc_s)| / sum_s w_s over the sources
    that measured it (1 = all agree on the direction, 0 = evenly split). Where agreement is
    at least `threshold` the weighted mean is kept; below it, the value of the most similar
    source that measured the entry is used instead of an average that cancels towards zero.
    threshold = 0 is WeightedTransfer. `restore = "source"` then rescales each target to the
    weighted mean norm of the source responses, as NormRestoredTransfer does."""

    temperature: float = 0.1
    threshold: float = 0.5
    restore: str = "none"

    def __call__(self, sources, target_basal, perts, genes, symbols=None) -> Prediction:
        sim = basal_similarity(target_basal, sources)
        w = np.exp((sim - sim.max()) / self.temperature)
        w = w / w.sum()
        pred = _weighted_transfer(sources, target_basal, perts, w)
        lfc = _stack(sources, perts, "lfc")  # (sources, perts, genes)
        present = np.isfinite(lfc)
        wb = w[:, None, None] * present
        with np.errstate(all="ignore"):
            agree = np.abs((np.sign(np.nan_to_num(lfc)) * wb).sum(0)) / wb.sum(0)
        top = np.argmax(np.where(present, w[:, None, None], -np.inf), axis=0)
        nearest = np.take_along_axis(np.nan_to_num(lfc), top[None], axis=0)[0]
        measured = present.any(0)
        use_nearest = measured & (np.nan_to_num(agree, nan=1.0) < self.threshold)
        out = np.where(use_nearest, nearest, pred)
        if self.restore == "source":
            with np.errstate(all="ignore"):
                norms = np.sqrt(np.nansum(np.nan_to_num(lfc) ** 2, axis=2))
                seen = present.any(axis=2)
                target = (norms * seen * w[:, None]).sum(0) / np.maximum(
                    (seen * w[:, None]).sum(0), 1e-12
                )
                own = np.linalg.norm(out, axis=1)
                scale = np.where((own > 0) & seen.any(0), target / own, 1.0)
            out = out * scale[:, None]
        return Prediction(pd.DataFrame(out, index=perts, columns=genes))
