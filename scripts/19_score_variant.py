"""Score one transfer variant on a held-out line with the full scorer, from the cache.

The benchmark (scripts/02) reloads every aligned context, which takes tens of GB; this
script builds the prediction from the cache of scripts/17 (the same code path as the fast
evaluator, scripts/18), emits cells from the cached basal half, applies the production
post-processing (generic scale, rho-targeted amplitude from the emitted groups' own
sampling noise, exact-mean emission) and runs `cell-eval2 --preset vcc2026` against the
benchmark's real evaluation set. The real-side DE table is written once per held-out line
and reused.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.emit import emit_cells, emit_cells_exact
from zsp.profile import calibrate_amplitude, emitted_group_noise

sys.path.insert(0, str(Path(__file__).resolve().parent))
fast = importlib.import_module("18_fast_eval")

PERFECT = {"pds": 1, "mse": 0, "nmae": 0, "fid": 1, "reach": 1, "jac": 1}
COLS = {
    "pds": "pds_cosine",
    "mse": "expr_mse_unbiased_capped_norm",
    "nmae": "de_wilcoxon_lfc_nmae",
    "fid": "de_wilcoxon_direction_fidelity_yield_raw",
    "reach": "de_wilcoxon_direction_reach_raw",
    "jac": "de_wilcoxon_sig_jaccard",
}


def cell_eval(
    pred_path: Path, real_path: Path, outdir: Path, de_real: Path, write_degenes: bool = False
) -> None:
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
    cmd += ["--de-real", str(de_real)] if de_real.exists() else []
    if write_degenes or not de_real.exists():
        cmd += ["--write-degenes"]
    with open(outdir / "cell_eval2.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = (outdir / "cell_eval2.log").read_text().splitlines()[-15:]
        raise RuntimeError("cell-eval2 failed:\n" + "\n".join(tail))
    if not de_real.exists() and (outdir / "de_real.parquet").exists():
        shutil.copy(outdir / "de_real.parquet", de_real)


def scaled_table(outdir: Path, baseline_dir: Path) -> pd.DataFrame:
    u = pd.read_csv(outdir / "agg_results.csv", index_col=0).loc["mean"]
    b = pd.read_csv(baseline_dir / "baseline_agg.csv", index_col=0).loc["mean"]
    rows = {}
    for k, col in COLS.items():
        raw, base = float(u[col]), float(b[col])
        s = (raw - base) / (PERFECT[k] - base)
        rows[k] = {"raw": raw, "baseline": base, "scaled": min(s, 1.0) if k == "mse" else s}
    t = pd.DataFrame(rows).T
    t.loc["mse", "scaled"] = max(t.loc["mse", "scaled"], 0.0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", default="h1")
    ap.add_argument("--variant", required=True, help="JSON dict of model options (scripts/18)")
    ap.add_argument("--name", required=True, help="run name, e.g. v6_candidate")
    ap.add_argument("--rho-target", type=float, default=None)
    ap.add_argument("--generic-scale", type=float, default=None)
    ap.add_argument("--exact", action="store_true", help="emit cells with exact pooled means")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--write-degenes", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    res = cfg.path("results")
    cache_dir, work = res / "cache", res / "local_eval"
    held = args.held_out
    all_names = cfg.context_names(cfg["evaluation"].get("extra_sources", []))
    names = args.sources or [n for n in all_names if n != held]
    real = fast.RealSide(held, cache_dir, work)
    cache = fast.Cache(cache_dir, names)
    basal_npz = np.load(cache_dir / f"{held}_basal.npz")
    basal_cells = sp.csr_matrix(ad.read_h5ad(cache_dir / f"{held}_basal_cells.h5ad").X)
    control_mean = basal_npz["control_mean"]
    v = json.loads(args.variant)
    means = fast.predict_variant(cache, real, basal_npz, basal_cells, names, v)
    rng = np.random.default_rng(cfg.seed)
    real_ad = ad.read_h5ad(work / f"{held}_real.h5ad", backed="r")
    n_per = real_ad.obs["target"].astype(str).value_counts()
    n_cells = int(n_per[real.perts[0]])
    amplitude = 1.0
    if args.rho_target or args.generic_scale is not None:
        noise = emitted_group_noise(basal_cells, n_cells, rng)
        credit_total = noise * len(real.perts) / (means.shape[1] - 1)
        means, amplitude = calibrate_amplitude(
            means, control_mean, credit_total, args.rho_target or 1.0, args.generic_scale
        )
        print(f"emitted-group noise {noise:.4f}; amplitude {amplitude:.3f}", flush=True)
    emit = emit_cells_exact if args.exact else emit_cells
    blocks, labels = [basal_cells], ["non-targeting"] * basal_cells.shape[0]
    for i, p in enumerate(real.perts):
        blocks.append(emit(basal_cells, control_mean, means[i], int(n_per[p]), rng))
        labels += [p] * int(n_per[p])
    pred = ad.AnnData(
        sp.vstack(blocks).tocsr(),
        obs=pd.DataFrame({"target": labels}),
        var=pd.DataFrame(index=real.genes),
    )
    pred.obs_names = [f"pred{i}" for i in range(pred.n_obs)]
    out = work / f"{held}_{args.name}"
    pred_path = work / f"{held}_{args.name}_pred.h5ad"
    pred.write_h5ad(pred_path)
    b_hat = fast.profile(means)
    print(
        f"fast evaluator on the emitted means: pds {real.pds(b_hat):.3f}, "
        f"mse(no credit) {real.mse(b_hat, None):.3f}",
        flush=True,
    )
    cell_eval(
        pred_path,
        work / f"{held}_real.h5ad",
        out,
        work / f"{held}_de_real.parquet",
        args.write_degenes,
    )
    json.dump(
        {
            "variant": v,
            "rho_target": args.rho_target,
            "generic_scale": args.generic_scale,
            "exact": args.exact,
            "amplitude": amplitude,
            "sources": names,
        },
        open(out / "variant.json", "w"),
        indent=1,
    )
    t = scaled_table(out, work / f"{held}_baseline")
    t.to_csv(out / "leaderboard_scale.csv")
    print(t.round(4).to_string())
    print(f"overall (local scale, 1 = perfect): {t['scaled'].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
