"""Local scores on the leaderboard's scale: 0 is the mean-response baseline, 1 a perfect score.

The challenge rescales each vcc2026 metric so that 0 is the organisers' generic-response
baseline: every knockdown predicted to have the held-out line's own average response. That
baseline is an oracle (it reads the real held-out cells), so beating it means the prediction
carries perturbation-specific signal. For each evaluation set in results/local_eval this
builds the baseline with `cell-eval2 baseline` and scores every run against it with
`cell-eval2 score`.

A run is scored only if it was made on the evaluation file now on disk: same perturbations,
and the scorer's own fingerprint of the real cells must match. Runs scored on an earlier set
(for example some four-screen runs) are listed as stale.

Writes results/tables/local_leaderboard_scale.csv (one row per held-out line x run).
"""

from __future__ import annotations

import argparse
import subprocess

import anndata as ad
import pandas as pd

from zsp.config import load_config

SHORT = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
    "avg_score": "overall",
}


def run(cmd: list[str], log) -> bool:
    """Run a cell-eval2 step; False if the scorer refused a run made on other real cells."""
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        if "source_fingerprint" in log.read_text():
            return False
        raise RuntimeError(f"{cmd[1]} failed, see {log}")
    return True


def build_baseline(real_path, outdir) -> None:
    if (outdir / "baseline_agg.csv").exists():
        return
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"baseline: {real_path.name}", flush=True)
    cmd = ["cell-eval2", "baseline", "-ar", str(real_path), "-o", str(outdir)]
    run([*cmd, "--preset", "vcc2026", "--input-type", "counts"], outdir / "cell_eval2.log")


def real_perts(real_path) -> frozenset[str]:
    obs = ad.read_h5ad(real_path, backed="r").obs
    return frozenset(obs["target"].astype(str)) - {"non-targeting"}


def run_perts(run_dir) -> frozenset[str]:
    return frozenset(pd.read_csv(run_dir / "results.csv", usecols=["perturbation"]).perturbation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="*", help="evaluation sets, e.g. h1 hepg2@h1")
    args = ap.parse_args()
    cfg = load_config()
    work = cfg.path("results") / "local_eval"
    reals = work.glob("*_real*.h5ad")
    sets = sorted(p.name.removesuffix(".h5ad").replace("_real", "") for p in reals)
    sets = [s for s in sets if s != "smoke" and (not args.held_out or s in args.held_out)]
    rows = []
    for eval_set in sets:
        held, _, within = eval_set.partition("@")
        real_path = work / f"{held}_real{'@' + within if within else ''}.h5ad"
        base = work / f"{eval_set}_baseline"
        build_baseline(real_path, base)
        perts = real_perts(real_path)
        for run_dir in sorted(work.glob(f"{held}_*")):
            name = run_dir.name.removeprefix(f"{held}_")
            if not (run_dir / "agg_results.csv").exists() or name.startswith("baseline"):
                continue
            # runs on the other evaluation set of this line belong to that set
            if (within and not name.endswith(f"@{within}")) or (not within and "@" in name):
                continue
            model = name.removesuffix(f"@{within}")
            row = {"held_out": eval_set, "model": model}
            if run_perts(run_dir) != perts:
                rows.append(row | {"status": "stale evaluation set"})
                continue
            out = run_dir / "leaderboard_scale.csv"
            cmd = ["cell-eval2", "score", "--user-agg", str(run_dir / "agg_results.csv")]
            cmd += ["--baseline-agg", str(base / "baseline_agg.csv")]
            cmd += ["--baseline-meta", str(base / "baseline_meta.json")]
            cmd += ["--user-meta", str(run_dir / "run_meta.json"), "-o", str(out)]
            if not run(cmd, run_dir / "score.log"):
                # same perturbations, other cells: scored before the current file was written
                rows.append(row | {"status": "stale evaluation set"})
                continue
            score = pd.read_csv(out).set_index("metric")["from_baseline"]
            rows.append(row | {"status": "scored"} | {SHORT[k]: v for k, v in score.items()})
    table = pd.DataFrame(rows)
    path = cfg.path("results") / "tables" / "local_leaderboard_scale.csv"
    table.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(3).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
