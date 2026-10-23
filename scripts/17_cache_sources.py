"""Cache source responses and held-out basal states for the fast pseudobulk evaluator.

Loading all eight aligned contexts at once takes tens of GB; the fast evaluator (scripts/18)
only needs each source's response to a few hundred perturbations. This script loads one
context at a time, decomposes it exactly as the benchmark does, keeps the rows it needs and
writes a compact cache under results/cache/. It also rebuilds, with the benchmark's own
generator, the basal half of each held-out line's control cells, so the evaluator sees the
same basal state the benchmark gave the models and never the scorer's half.

Outputs (results/cache/):
  genes.parquet                union gene axis (Ensembl id, symbol), in align_union order
  perts.txt                    perturbations cached (held-out evaluation sets + challenge panel)
  <source>.npz                 lfc (perts x genes, float32, NaN = not measured), generic,
                               basal (CP10k), measured, n_cells; own_lfc in <source>_own.parquet
  <held>_basal_cells.h5ad      the basal half of the held-out line's control cells
  <held>_basal.npz             control_mean (measured genes), genes (union positions), perts
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from zsp.config import load_config
from zsp.data import Context, align_union
from zsp.models import decompose
from zsp.store import load_context

sys.path.insert(0, str(Path(__file__).resolve().parent))
bench = importlib.import_module("02_local_benchmark")


def light_contexts(names: list[str], proc: Path) -> list[Context]:
    """Contexts without their perturbation matrices: enough for the union gene axis."""
    out = []
    for n in names:
        g = pd.read_parquet(proc / f"{n}_genes.parquet")
        genes = pd.Index(g.index, name="gene_id")
        ncell = pd.read_parquet(proc / f"{n}_ncells.parquet")["n_cells"]
        empty = pd.DataFrame(index=pd.Index([], dtype=str), columns=genes, dtype=float)
        out.append(Context(n, genes, g["symbol"], g["control"].to_numpy(np.float64), empty, ncell))
    return align_union(out)


def pert_index(proc: Path, name: str) -> pd.Index:
    """The perturbations a context's matrix holds, read without loading the matrix."""
    return pd.read_parquet(proc / f"{name}_pert.parquet", columns=[]).index.astype(str)


def needed_perts(work: Path, panel_csvs: list[Path], held_outs: list[str]) -> list[str]:
    need: set[str] = set()
    for panel_csv in panel_csvs:
        need |= set(pd.read_csv(panel_csv)["target_gene"].astype(str))
    for held in held_outs:
        p = work / f"{held}_real.h5ad"
        if p.exists():
            obs = ad.read_h5ad(p, backed="r").obs["target"].astype(str)
            need |= set(obs.unique()) - {"non-targeting"}
    return sorted(need)


def cache_source(name: str, proc: Path, union: list[Context], need: list[str], out: Path):
    by = {c.name: c for c in union}
    ctx = load_context(name, proc).reindex_genes(union[0].genes, union[0].symbols)
    se = decompose(ctx)
    lfc = se.lfc.reindex(need)
    np.savez_compressed(
        out / f"{name}.npz",
        lfc=lfc.to_numpy(np.float32),
        generic=se.generic.astype(np.float32),
        basal=se.basal.astype(np.float32),
        measured=by[name].measured,
        n_cells=ctx.n_cells.reindex(need).fillna(0).to_numpy(np.int32),
        n_perts_total=np.int64(len(se.lfc)),
    )
    se.own_lfc.rename("own_lfc").to_frame().to_parquet(out / f"{name}_own.parquet")
    print(
        f"{name}: {len(se.lfc)} perturbations, {int(np.isfinite(lfc.to_numpy()).any(1).sum())}"
        f"/{len(need)} needed ones measured",
        flush=True,
    )
    del ctx, se, lfc
    gc.collect()


def cache_heldout(cfg, held: str, core: list[str], union: list[Context], proc: Path, out: Path):
    """Replay build_eval_set with the benchmark's seed and keep the basal half."""
    by = {c.name: c for c in union}
    genes = union[0].genes[by[held].measured]
    source_perts: set[str] = set()
    for n in core:
        if n == held:
            continue
        nc = pd.read_parquet(proc / f"{n}_ncells.parquet")["n_cells"]
        idx = pert_index(proc, n)
        keep = nc.reindex(idx).fillna(0) >= 10
        source_perts |= set(idx[keep.to_numpy()])
    rng = np.random.default_rng(cfg.seed)
    real, basal_cells, perts = bench.build_eval_set(cfg, held, genes, source_perts, rng)
    on_disk = ad.read_h5ad(cfg.path("results") / "local_eval" / f"{held}_real.h5ad", backed="r")
    disk_perts = sorted(set(on_disk.obs["target"].astype(str)) - {"non-targeting"})
    if disk_perts != list(perts):
        raise RuntimeError(f"{held}: replayed evaluation set differs from {held}_real.h5ad")
    if list(on_disk.var_names) != list(real.var_names):
        raise RuntimeError(f"{held}: replayed gene axis differs from {held}_real.h5ad")
    basal = ad.AnnData(basal_cells, var=pd.DataFrame(index=real.var_names))
    basal.obs_names = [f"basal{i}" for i in range(basal.n_obs)]
    basal.write_h5ad(out / f"{held}_basal_cells.h5ad")
    control_mean = np.asarray(basal_cells.mean(0)).ravel()
    np.savez_compressed(
        out / f"{held}_basal.npz",
        control_mean=control_mean,
        gene_pos=np.flatnonzero(by[held].measured),
        perts=np.array(perts),
    )
    print(
        f"{held}: basal half {basal.n_obs} cells, {len(perts)} evaluated perturbations", flush=True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="*", default=["h1", "hepg2"])
    ap.add_argument("--skip-sources", action="store_true")
    ap.add_argument(
        "--bundle",
        nargs="*",
        default=[],
        help="controls bundles whose pert_counts.csv name the panels to cache (default "
        "data/raw/vcc); pass both bundles so validation and final targets are cached",
    )
    args = ap.parse_args()
    cfg = load_config()
    proc, res = cfg.path("processed"), cfg.path("results")
    out = res / "cache"
    out.mkdir(exist_ok=True)
    extras = list(cfg["evaluation"].get("extra_sources", []))
    names = cfg.context_names(extras)
    core = [n for n in names if n not in extras]
    union = light_contexts(names, proc)
    pd.DataFrame(
        {"symbol": union[0].symbols.loc[union[0].genes].to_numpy()}, index=union[0].genes
    ).to_parquet(out / "genes.parquet")
    bundles = [Path(b) for b in args.bundle] or [cfg.path("raw") / "vcc"]
    need = needed_perts(res / "local_eval", [b / "pert_counts.csv" for b in bundles], args.held_out)
    (out / "perts.txt").write_text("\n".join(need) + "\n")
    print(f"union genes {len(union[0].genes)}; perturbations needed {len(need)}", flush=True)
    if not args.skip_sources:
        for n in names:
            cache_source(n, proc, union, need, out)
    for held in args.held_out:
        cache_heldout(cfg, held, core, union, proc, out)


if __name__ == "__main__":
    main()
