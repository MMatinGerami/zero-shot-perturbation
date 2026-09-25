"""Leave-one-context-out benchmark scored with the official challenge metrics (cell-eval2).

For each held-out cell line:
  * sources = all other contexts (their perturbation responses are known);
  * the held-out line contributes only half of its control cells ("basal state"),
    exactly the information the challenge gives; the other half is kept for scoring;
  * each model predicts every evaluated knockdown, predictions are emitted as cells, and
    `cell-eval2 --preset vcc2026` scores them against the real cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.data import align_union, cp10k
from zsp.emit import emit_cells
from zsp.models import (
    CalibratedTransfer,
    decompose,
    lfc_to_counts,
    predict_control,
    predict_mean_transfer,
)
from zsp.store import load_context

MODELS = {
    "control": predict_control,
    "mean_transfer": predict_mean_transfer,
    "calibrated": CalibratedTransfer(),
}


def loso_calibrated(res, held: str):
    """CalibratedTransfer with (alpha, gate) chosen by leave-one-source-out for this held-out
    line (scripts/06); falls back to the defaults if the LOSO table has not been produced."""
    path = res / "tables" / "loso_alpha.csv"
    if not path.exists():
        return CalibratedTransfer()
    t = pd.read_csv(path)
    t = t[t.held_out == held]
    if t.empty:
        return CalibratedTransfer()
    best = t.loc[t.mean_pearson_loso.idxmax()]
    print(f"{held}: LOSO-selected alpha={best.alpha}, gate={best.gate_cp10k}", flush=True)
    return CalibratedTransfer(alpha=float(best.alpha), gate_cp10k=float(best.gate_cp10k))


def symbol_index(symbols) -> pd.Index:
    """Gene symbols as the feature index (what the scorer matches targets against), made unique."""
    idx = pd.Index(list(symbols), name="gene")
    return idx if idx.is_unique else pd.Index(ad.utils.make_index_unique(idx), name="gene")


def build_eval_set(cfg, held: str, genes: pd.Index, source_perts: set[str], rng):
    spec = cfg["contexts"][held]
    a = ad.read_h5ad(cfg.path("raw") / spec["file"], backed="r")
    ev = cfg["evaluation"]
    target = a.obs["gene"].astype(str)
    counts = target.value_counts()
    eligible = [
        p
        for p in counts.index
        if p != "non-targeting"
        and counts[p] >= ev["min_cells_per_perturbation"]
        and p in source_perts
    ]
    perts = sorted(rng.choice(eligible, min(ev["n_perturbations"], len(eligible)), replace=False))
    gene_pos = a.var_names.get_indexer(genes)
    ctrl_idx = np.flatnonzero(target.to_numpy() == "non-targeting")
    rng.shuffle(ctrl_idx)
    basal_idx, eval_ctrl_idx = np.sort(ctrl_idx[::2]), np.sort(ctrl_idx[1::2])
    rows, labels = [eval_ctrl_idx], ["non-targeting"] * len(eval_ctrl_idx)
    for p in perts:
        idx = np.flatnonzero(target.to_numpy() == p)
        idx = np.sort(
            rng.choice(idx, min(len(idx), ev["max_cells_per_perturbation"]), replace=False)
        )
        rows.append(idx)
        labels += [p] * len(idx)
    take = np.concatenate(rows)
    order = np.argsort(take)
    X = np.asarray(a.X[take[order]])[:, gene_pos]
    X = X[np.argsort(order)]
    real = ad.AnnData(
        sp.csr_matrix(X.astype(np.float32)),
        obs=pd.DataFrame({"target": labels}),
        var=pd.DataFrame(index=symbol_index(a.var.loc[genes, "gene_name"])),
    )
    real.obs_names = [f"cell{i}" for i in range(real.n_obs)]
    basal = sp.csr_matrix(np.asarray(a.X[basal_idx])[:, gene_pos].astype(np.float32))
    return real, basal, perts


def cell_eval(pred_path, real_path, outdir):
    """Run the official scorer; its full log is kept next to the results."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cell-eval2",
        "run",
        "-ap",
        str(pred_path),
        "-ar",
        str(real_path),
        "-o",
        str(outdir),
        "--preset",
        "vcc2026",
        "--input-type",
        "counts",
    ]
    with open(outdir / "cell_eval2.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = (outdir / "cell_eval2.log").read_text().splitlines()[-15:]
        raise RuntimeError("cell-eval2 failed:\n" + "\n".join(tail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--n-perts", type=int, help="override evaluation.n_perturbations (smoke runs)")
    args = ap.parse_args()
    cfg = load_config()
    if args.n_perts:
        cfg["evaluation"]["n_perturbations"] = args.n_perts
    rng = np.random.default_rng(cfg.seed)
    proc, res = cfg.path("processed"), cfg.path("results")
    work = res / "local_eval"
    work.mkdir(exist_ok=True)
    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    rows = []
    for held in args.held_out or cfg["evaluation"]["held_out"]:
        contexts = align_union([load_context(n, proc) for n in names])
        by_name = {c.name: c for c in contexts}
        # sources keep every gene any of them measured; the held-out line is scored on the
        # genes it measured itself (the scorer needs real counts for every gene)
        union = contexts[0].genes
        genes = union[by_name[held].measured]
        sources = [decompose(by_name[n]) for n in names if n != held]
        source_perts = set().union(*[set(s.lfc.index) for s in sources])
        real, basal_cells, perts = build_eval_set(cfg, held, genes, source_perts, rng)
        real_path = work / f"{held}_real.h5ad"
        real.write_h5ad(real_path)
        control_mean = np.asarray(basal_cells.mean(0)).ravel()
        target_basal = cp10k(control_mean)[0]
        for name, model in MODELS.items():
            if args.models and name not in args.models:
                continue
            predictor = loso_calibrated(res, held) if name == "calibrated" else model
            basal_union = np.full(len(union), np.nan)
            basal_union[by_name[held].measured] = target_basal
            pred = predictor(sources, basal_union, perts, union, contexts[0].symbols.loc[union])
            pred.lfc = pred.lfc.loc[:, genes]
            means = lfc_to_counts(pred.lfc.to_numpy(), control_mean)
            blocks, labels = [basal_cells], ["non-targeting"] * basal_cells.shape[0]
            n_per = real.obs["target"].value_counts()
            for i, p in enumerate(perts):
                blocks.append(emit_cells(basal_cells, control_mean, means[i], int(n_per[p]), rng))
                labels += [p] * int(n_per[p])
            pred_ad = ad.AnnData(
                sp.vstack(blocks).tocsr(),
                obs=pd.DataFrame({"target": labels}),
                var=pd.DataFrame(index=real.var_names),
            )
            pred_ad.obs_names = [f"pred{i}" for i in range(pred_ad.n_obs)]
            pred_path = work / f"{held}_{name}_pred.h5ad"
            pred_ad.write_h5ad(pred_path)
            out = work / f"{held}_{name}"
            cell_eval(pred_path, real_path, out)
            if len(pred.uncertainty):
                pred.uncertainty.rename("uncertainty").to_frame().to_csv(out / "uncertainty.csv")
            rows.append({"held_out": held, "model": name, "outdir": str(out)})
            print(f"{held} / {name}: scored -> {out}", flush=True)
    (res / "tables").mkdir(exist_ok=True)
    table = res / "tables" / "local_runs.csv"
    new = pd.DataFrame(rows)
    if table.exists():  # merge: another held-out line may have been run separately
        old = pd.read_csv(table)
        old = old[~(old.held_out + "/" + old.model).isin(new.held_out + "/" + new.model)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(table, index=False)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
