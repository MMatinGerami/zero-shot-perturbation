"""Round 8: the pre-registered fluctuation-response analysis (results/prereg/, round 8).

For every single-cell line (HepG2, Jurkat, HCT116, HEK293T) the line plays the held-out role:
the other five screens are its sources, half of its control cells are the model's basal state,
and the other half plus up to 100 cells per perturbation are the truth. On every line:

  signal    for each target the line expresses (CP10k >= 1): r(fr, measured LFC) over expressed
            genes except the target, minus the mean r of 20 placebo genes' fr columns;
  blend     centred gene-wise r of (transfer + lambda * fr) against the truth, minus that of the
            transfer alone, per perturbation, for each lambda, estimator and base model.

Then, leave-one-line-out: for each held-out line (HepG2, Jurkat, HCT116) the estimator and
lambda are chosen on the OTHER three lines only, and the held-out line's numbers at that choice
are the verdict. Nothing is chosen on the line it is reported for.

Outputs: results/tables/fluctuation_{targets,blend,verdict}.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from zsp.ceiling import rowwise_pearson
from zsp.config import load_config
from zsp.data import align_union, cp10k, log_fold_change
from zsp.fluctuation import ESTIMATORS, fr_prediction, normalised_cells
from zsp.models import NormRestoredTransfer, WeightedTransfer, decompose, knockdown_depth
from zsp.store import load_context

LINES = ["hepg2", "jurkat", "hct116", "hek293t"]
VERDICT_LINES = ["hepg2", "jurkat", "hct116"]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
N_PLACEBO = 20
N_BOOT = 2000
BASES = {"norm_restored": NormRestoredTransfer(0.1), "weighted": WeightedTransfer(0.1)}

_spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).with_name("02_local_benchmark.py")
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def pseudobulk_truth(real, expr_min=1.0):
    X = real.X.tocsr()
    t = real.obs["target"].astype(str).to_numpy()
    ctrl = np.asarray(X[t == "non-targeting"].mean(0)).ravel()
    perts = sorted(set(t) - {"non-targeting"})
    means = np.vstack([np.asarray(X[t == p].mean(0)).ravel() for p in perts])
    return perts, log_fold_change(means, ctrl), cp10k(ctrl)[0] >= expr_min


def run_line(cfg, line, names, rng):
    proc = cfg.path("processed")
    contexts = align_union([load_context(n, proc) for n in names])
    by = {c.name: c for c in contexts}
    union = contexts[0].genes
    genes = union[by[line].measured]
    sources = [decompose(by[n]) for n in names if n != line]
    source_perts = set().union(*[set(s.lfc.index) for s in sources])
    real, basal_cells, perts = bench.build_eval_set(cfg, line, genes, source_perts, rng)
    t_perts, truth, expr = pseudobulk_truth(real)
    assert t_perts == sorted(perts)
    perts = t_perts
    symbols = np.asarray(real.var_names).astype(str)  # symbols, same order as `genes`
    basal_mean = np.asarray(basal_cells.mean(0)).ravel()
    target_basal = cp10k(basal_mean)[0]
    usable = target_basal >= 1.0
    depth = -float(np.median([knockdown_depth(s) for s in sources]))
    basal_union = bench.basal_union_for(by[line], target_basal)
    sym_union = contexts[0].symbols.loc[union]
    base_pred = {
        k: m(sources, basal_union, perts, union, sym_union).lfc.loc[:, genes].to_numpy()
        for k, m in BASES.items()
    }
    tc = truth[:, expr] - truth[:, expr].mean(0)
    target_rows, blend_rows = [], []
    for est in ESTIMATORS:
        Z = normalised_cells(basal_cells, est, seed=cfg.seed)
        fr, ok = fr_prediction(Z, symbols, perts, depth, usable)
        pos = {s: i for i, s in enumerate(symbols)}
        placebo_pool = np.flatnonzero(usable & expr)
        for i in np.flatnonzero(ok):
            j = pos[perts[i]]
            m = expr.copy()
            m[j] = False
            r = np.corrcoef(fr[i, m], truth[i, m])[0, 1]
            q = rng.choice(placebo_pool[placebo_pool != j], N_PLACEBO, replace=False)
            Zq = Z[:, q]
            bq = (Zq.T @ Z) / (Zq * Zq).sum(0)[:, None] * depth
            pl = rowwise_pearson(bq[:, m], np.broadcast_to(truth[i, m], (N_PLACEBO, m.sum())))
            target_rows.append(
                dict(
                    line=line,
                    estimator=est,
                    target=perts[i],
                    target_cp10k=target_basal[j],
                    r_target=r,
                    r_placebo=float(np.nanmean(pl)),
                )
            )
        for base, bp in base_pred.items():
            r0 = rowwise_pearson(bp[:, expr] - bp[:, expr].mean(0), tc)
            for lam in LAMBDAS:
                p = bp + lam * fr
                r = rowwise_pearson(p[:, expr] - p[:, expr].mean(0), tc)
                for i, pert in enumerate(perts):
                    blend_rows.append(
                        dict(
                            line=line,
                            estimator=est,
                            base=base,
                            lam=lam,
                            target=pert,
                            fr_applied=bool(ok[i]),
                            r_centred=r[i],
                            gain=r[i] - r0[i],
                        )
                    )
        print(f"{line}/{est}: {ok.sum()} of {len(perts)} targets usable", flush=True)
    return pd.DataFrame(target_rows), pd.DataFrame(blend_rows)


def verdict(targets: pd.DataFrame, blend: pd.DataFrame, rng) -> pd.DataFrame:
    targets = targets.assign(excess=targets.r_target - targets.r_placebo)
    rows = []
    for held in VERDICT_LINES:
        other = [ln for ln in LINES if ln != held]
        sel_t = targets[targets.line.isin(other)]
        est = sel_t.groupby(["estimator", "line"]).excess.median().groupby("estimator").mean()
        est_choice = est.idxmax()
        h = targets[(targets.line == held) & (targets.estimator == est_choice)]
        p_signal = wilcoxon(h.excess, alternative="greater").pvalue if len(h) > 5 else np.nan
        for base in BASES:
            b = blend[(blend.base == base) & (blend.estimator == est_choice)]
            sel = b[b.line.isin(other)].groupby(["lam", "line"]).gain.median()
            lam = float(sel.groupby("lam").mean().idxmax())
            g = b[(b.line == held) & (b.lam == lam)].gain.to_numpy()
            boots = [np.nanmean(rng.choice(g, len(g))) for _ in range(N_BOOT)]
            rows.append(
                dict(
                    held_out=held,
                    estimator=est_choice,
                    n_targets_usable=len(h),
                    excess_median=float(h.excess.median()),
                    excess_positive_share=float((h.excess > 0).mean()),
                    signal_p_one_sided=float(p_signal),
                    base=base,
                    lam=lam,
                    gain_mean=float(np.nanmean(g)),
                    gain_ci_low=float(np.quantile(boots, 0.025)),
                    gain_ci_high=float(np.quantile(boots, 0.975)),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    proc = cfg.path("processed")
    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    tables = cfg.path("results") / "tables"
    all_t, all_b = [], []
    for line in LINES:
        t, b = run_line(cfg, line, names, np.random.default_rng(cfg.seed))
        all_t.append(t)
        all_b.append(b)
    targets, blend = pd.concat(all_t), pd.concat(all_b)
    targets.to_csv(tables / "fluctuation_targets.csv", index=False)
    blend.to_csv(tables / "fluctuation_blend.csv", index=False)
    v = verdict(targets, blend, np.random.default_rng(cfg.seed))
    v.to_csv(tables / "fluctuation_verdict.csv", index=False)
    summary = (
        targets.assign(excess=targets.r_target - targets.r_placebo)
        .groupby(["line", "estimator"])[["r_target", "r_placebo", "excess"]]
        .median()
    )
    print(summary.round(3).to_string())
    print(v.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
