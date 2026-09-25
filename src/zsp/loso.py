"""Leave-one-source-out model selection on pseudobulk proxies of the six challenge metrics.

Cell-level scoring with `cell-eval2` takes ~20 minutes per model and line, far too slow for
a hyper-parameter grid. Instead, each source context in turn plays the held-out line (its
control profile is the only thing the model sees), and the prediction is compared with its
measured pseudobulk responses using cheap stand-ins for the challenge metrics:

  pds    rank of the true profile among all true profiles, by cosine to the prediction
  mse    squared error relative to predicting no change (1 = the control baseline)
  nmae   MAE on the truth's top-|LFC| genes, relative to predicting no change
  fid    fraction of the truth's top genes predicted with the right direction AND a
         magnitude above `MIN_LFC` - the scorer's DE metrics come from significance tests,
         so a correct sign at negligible magnitude earns nothing (rule v2; v1 counted the
         sign alone, was scale-invariant in four of six proxies, and therefore always chose
         the strongest damping, which the cell-level scorer then rejected)
  reach  fraction of the prediction's top genes whose true direction is right
  jac    Jaccard of the two top-gene sets

The selection rule is fixed before the hyper-parameter grid is run: maximise the unweighted
mean of the six oriented proxies (pds, 1-mse, 1-nmae, fid, reach, jac), averaged over
pseudo-targets. The held-out line never enters this loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = 200
MIN_LFC = 0.25  # log2 units; below this a "detected" DE gene is not credible


def _oriented(pds, mse, nmae, fid, reach, jac):
    return np.mean([pds, 1 - mse, 1 - nmae, fid, reach, jac])


def proxies(pred: np.ndarray, truth: np.ndarray, k: int = TOP_K) -> dict[str, float]:
    """Metric stand-ins for one pseudo-target. Rows = perturbations, columns = genes.
    Genes not measured in the pseudo-target (NaN in `truth`) are ignored everywhere."""
    ok = np.isfinite(truth).all(axis=0)
    pred, truth = np.nan_to_num(pred[:, ok]), truth[:, ok]
    n = truth.shape[0]

    # pds: normalised rank of the matching truth row under cosine similarity to each prediction
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    tn = truth / (np.linalg.norm(truth, axis=1, keepdims=True) + 1e-12)
    sim = pn @ tn.T  # (pred i, truth j)
    own = np.diag(sim)
    rank = (sim > own[:, None]).sum(axis=1)  # how many other truths look more like pred i
    pds = 1 - rank.mean() / max(n - 1, 1)

    mse = ((pred - truth) ** 2).mean() / ((truth**2).mean() + 1e-12)

    top_t = np.argsort(-np.abs(truth), axis=1)[:, :k]
    top_p = np.argsort(-np.abs(pred), axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    tt, pt = truth[rows, top_t], pred[rows, top_t]
    nmae = np.abs(pt - tt).mean() / (np.abs(tt).mean() + 1e-12)
    fid = ((np.sign(pt) == np.sign(tt)) & (np.abs(pt) >= MIN_LFC)).mean()
    reach = (np.sign(pred[rows, top_p]) == np.sign(truth[rows, top_p])).mean()
    jac = np.mean(
        [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top_t, top_p, strict=True)]
    )
    out = dict(pds=pds, mse=mse, nmae=nmae, fid=fid, reach=reach, jac=jac)
    out["mean_oriented"] = _oriented(**out)
    return out


def loso_scores(make_model, effects: dict, contexts: dict, source_names: list[str]) -> pd.Series:
    """Mean proxies over pseudo-targets for one model factory `make_model() -> predictor`.
    Models that need fitting receive the pseudo-target's own sources via `predictor.fit`."""
    rows = []
    for pseudo in source_names:
        srcs = [effects[n] for n in source_names if n != pseudo]
        tgt = effects[pseudo]
        perts = [p for p in tgt.lfc.index if any(p in s.lfc.index for s in srcs)]
        ctx = contexts[pseudo]
        basal = ctx_basal(ctx)
        model = make_model()
        if hasattr(model, "fit"):
            model.fit(srcs)
        pred = model(srcs, basal, perts, ctx.genes, ctx.symbols.loc[ctx.genes].to_numpy())
        rows.append(proxies(pred.lfc.to_numpy(), tgt.lfc.loc[perts].to_numpy()))
    return pd.DataFrame(rows).mean()


def ctx_basal(ctx) -> np.ndarray:
    from zsp.data import cp10k

    return cp10k(ctx.control)[0]
