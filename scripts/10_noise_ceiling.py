"""Noise ceiling for the local benchmark: how much of the gene-wise direction error is fixable?

For every held-out line, the benchmark's own real-cell file (results/local_eval/<line>_real.h5ad,
up to 100 cells per perturbation plus the scorer's control cells) is split in half at random,
and the two halves' LFCs are correlated gene by gene (src/zsp/ceiling.py). Each model's
pseudobulk prediction is then correlated with the full-sample truth and divided by the ceiling.

Outputs:
  results/tables/noise_ceiling.csv          one row per held-out line (median and IQR)
  results/tables/noise_ceiling_models.csv   one row per line and model: r, share of ceiling,
                                            with a 95% bootstrap CI over perturbations
  results/figures/noise_ceiling.png
Genes: expressed in the held-out control (CP10k >= 1), as in scripts/08_response_scale.py.
"""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.ceiling import ceiling_table, expressed_mask, rowwise_pearson, split_half_lfc
from zsp.config import load_config
from zsp.data import log_fold_change

CONTROL = "non-targeting"
N_BOOT = 2000


def pseudobulk(path) -> tuple[np.ndarray, pd.DataFrame]:
    a = ad.read_h5ad(path)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    target = a.obs["target"].astype(str).to_numpy()
    control = np.asarray(X[target == CONTROL].mean(0)).ravel()
    perts = sorted(set(target) - {CONTROL})
    means = np.vstack([np.asarray(X[target == p].mean(0)).ravel() for p in perts])
    return control, pd.DataFrame(means, index=perts, columns=a.var_names)


def boot_median_ci(x: np.ndarray, rng) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    meds = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    work = cfg.path("results") / "local_eval"
    tables, figs = cfg.path("results") / "tables", cfg.path("results") / "figures"
    line_rows, model_rows = [], []
    for held in cfg["evaluation"]["held_out"]:
        real_path = work / f"{held}_real.h5ad"
        if not real_path.exists():
            continue
        a = ad.read_h5ad(real_path)
        target = a.obs["target"].astype(str).to_numpy()
        perts = sorted(set(target) - {CONTROL})
        lfc_a, lfc_b, ctrl = split_half_lfc(a.X, target, perts, rng, CONTROL)
        expr = expressed_mask(ctrl)
        ceil = ceiling_table(lfc_a[:, expr], lfc_b[:, expr])
        n_cells = pd.Series(target).value_counts().reindex(perts).to_numpy()
        row = {"held_out": held, "n_perturbations": len(perts), "genes_expressed": int(expr.sum())}
        row["cells_per_perturbation_median"] = float(np.median(n_cells))
        for k, v in ceil.items():
            row[f"{k}_median"] = float(np.nanmedian(v))
            row[f"{k}_q25"] = float(np.nanquantile(v, 0.25))
            row[f"{k}_q75"] = float(np.nanquantile(v, 0.75))
        line_rows.append(row)

        ctrl_real, pert_real = pseudobulk(real_path)
        truth = log_fold_change(pert_real.loc[perts].to_numpy(), ctrl_real)[:, expr]
        for pred_path in sorted(work.glob(f"{held}_*_pred.h5ad")):
            model = pred_path.name[len(held) + 1 : -len("_pred.h5ad")]
            if model == "control":
                continue
            ctrl_pred, pert_pred = pseudobulk(pred_path)
            pred = log_fold_change(pert_pred.loc[perts, pert_real.columns].to_numpy(), ctrl_pred)[
                :, expr
            ]
            out = {"held_out": held, "model": model}
            for kind, (p, t) in {
                "raw": (pred, truth),
                "centred": (pred - pred.mean(0), truth - truth.mean(0)),
            }.items():
                r = rowwise_pearson(p, t)
                share = r / np.where(ceil[f"ceiling_{kind}"] > 0, ceil[f"ceiling_{kind}"], np.nan)
                out[f"r_{kind}_median"] = float(np.nanmedian(r))
                out[f"r_{kind}_ci_low"], out[f"r_{kind}_ci_high"] = boot_median_ci(r, rng)
                out[f"ceiling_{kind}_median"] = float(np.nanmedian(ceil[f"ceiling_{kind}"]))
                out[f"share_of_ceiling_{kind}_median"] = float(np.nanmedian(share))
            model_rows.append(out)
            print(
                f"{held:7s} {model:16s} r {out['r_raw_median']:.2f} / ceiling "
                f"{out['ceiling_raw_median']:.2f}   centred r {out['r_centred_median']:.2f} / "
                f"{out['ceiling_centred_median']:.2f}",
                flush=True,
            )

    lines, models = pd.DataFrame(line_rows), pd.DataFrame(model_rows)
    lines.to_csv(tables / "noise_ceiling.csv", index=False)
    models.to_csv(tables / "noise_ceiling_models.csv", index=False)
    print(lines.round(3).T.to_string())

    figs.mkdir(parents=True, exist_ok=True)
    show = ["mean_transfer", "weighted", "norm_restored"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kind in zip(axes, ["raw", "centred"], strict=True):
        x = np.arange(len(lines))
        ax.bar(x, lines[f"ceiling_{kind}_median"], 0.7, color="#d9d9d9", label="noise ceiling")
        for i, m in enumerate(show):
            sub = models[models.model == m].set_index("held_out").reindex(lines.held_out)
            ax.bar(x - 0.2 + 0.2 * i, sub[f"r_{kind}_median"], 0.18, label=m.replace("_", " "))
        ax.set_xticks(x, lines.held_out)
        ax.set_title(f"gene-wise r, {kind}")
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("median over perturbations")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "noise_ceiling.png", dpi=150)


if __name__ == "__main__":
    main()
