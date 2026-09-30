"""Paired comparison of two local runs scored on the same real cells.

For every metric, the per-perturbation difference (candidate minus base) is averaged and a 95 %
bootstrap CI is taken over perturbations, the unit of replication. mse and nmae are oriented so
that a positive difference is always an improvement. The same held-out knockdowns must have
been scored in both runs (checked); the benchmark guarantees it when the base and the candidate
come from the same evaluation set (same `--eval-within`, same held-out line and seed).

Usage:
  uv run python scripts/15_paired_compare.py --base weighted --candidate weighted+cd4 \
      --held-out hepg2 jurkat hct116 --out results/tables/round9_paired.csv
Several --candidate / --base pairs can be given in order (`--pairs base:cand ...`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zsp.config import load_config

PER_PERT = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped": "mse",
    "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
}
LOWER_IS_BETTER = {"mse", "nmae"}
N_BOOT = 2000


def per_pert(run_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(run_dir / "results.csv")
    r = r[r.metric.isin(PER_PERT)].assign(metric=lambda d: d.metric.map(PER_PERT))
    return r.pivot_table(index="perturbation", columns="metric", values="value")


def compare(work: Path, held: str, base: str, cand: str, rng) -> list[dict]:
    a, b = per_pert(work / f"{held}_{base}"), per_pert(work / f"{held}_{cand}")
    if set(a.index) != set(b.index):
        raise ValueError(f"{held}: {base} and {cand} scored different perturbations")
    rows = []
    for m in PER_PERT.values():
        d = (b[m] - a.loc[b.index, m]).dropna().to_numpy()
        if m in LOWER_IS_BETTER:
            d = -d
        boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        rows.append(
            dict(
                held_out=held,
                base=base,
                candidate=cand,
                metric=m,
                n=len(d),
                improvement=float(d.mean()),
                ci_low=float(np.quantile(boots, 0.025)),
                ci_high=float(np.quantile(boots, 0.975)),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="base:candidate run names")
    ap.add_argument("--held-out", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    work = cfg.path("results") / "local_eval"
    rows = []
    for pair in args.pairs:
        base, cand = pair.split(":")
        for held in args.held_out:
            if (work / f"{held}_{cand}" / "results.csv").exists():
                rows += compare(work, held, base, cand, rng)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    show = out.assign(
        cell=lambda d: (
            d.improvement.map("{:+.3f}".format)
            + " ["
            + d.ci_low.map("{:+.3f}".format)
            + ", "
            + d.ci_high.map("{:+.3f}".format)
            + "]"
        )
    )
    print(
        show.pivot_table(
            index=["held_out", "candidate"], columns="metric", values="cell", aggfunc="first"
        ).to_string()
    )


if __name__ == "__main__":
    main()
