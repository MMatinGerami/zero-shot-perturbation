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

import sys

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
from zsp.plotting import (
    DOUBLE,
    GREY_LIGHT,
    INK,
    LINE_LABELS,
    MM,
    MODEL_LABELS,
    SERIES,
    apply_style,
    panel_label,
    save,
)

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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="*", help="default: evaluation.held_out")
    args = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    work = cfg.path("results") / "local_eval"
    tables, figs = cfg.path("results") / "tables", cfg.path("results") / "figures"
    line_rows, model_rows = [], []
    for held in args.held_out or cfg["evaluation"]["held_out"]:
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
            if model == "control" or "@" in model:
                continue  # an @<context> run is scored against its own real file
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

    plot(lines, models, figs)


def plot(lines: pd.DataFrame, models: pd.DataFrame, figs) -> None:
    apply_style()
    show = ["mean_transfer", "weighted", "norm_restored"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.8, 52 * MM), sharey=True)
    for ax, letter, kind in zip(axes, "ab", ["raw", "centred"], strict=True):
        x = np.arange(len(lines))
        ax.bar(
            x, lines[f"ceiling_{kind}_median"], 0.72, color=GREY_LIGHT, label="Noise ceiling", lw=0
        )
        for i, (m, colour) in enumerate(zip(show, SERIES, strict=False)):
            sub = models[models.model == m].set_index("held_out").reindex(lines.held_out)
            ax.bar(
                x - 0.22 + 0.22 * i,
                sub[f"r_{kind}_median"],
                0.2,
                color=colour,
                label=MODEL_LABELS[m],
                lw=0,
            )
        ax.set_xticks(x, [LINE_LABELS.get(h, h) for h in lines.held_out])
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlabel("Held-out cell line")
        panel_label(ax, letter, x=-0.16 if letter == "a" else -0.06)
    axes[0].set_ylabel("Median gene-wise r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    save(fig, figs / "noise_ceiling")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no recomputation
        res = load_config().path("results")
        plot(
            pd.read_csv(res / "tables" / "noise_ceiling.csv"),
            pd.read_csv(res / "tables" / "noise_ceiling_models.csv"),
            res / "figures",
        )
    else:
        main()
