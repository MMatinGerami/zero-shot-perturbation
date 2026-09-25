"""Stream the X-Atlas/Orion genome-wide Perturb-seq screens (Huang et al. 2025) into a
context, without ever holding a screen on disk.

The Hugging Face release (Xaira-Therapeutics/X-Atlas-Orion, CC BY-NC-SA 4.0) is ~63 GB per
cell line in ~170 parquet batches of ~18k cells. Each batch is downloaded, checked against
its sha256, reduced, and deleted:

  * per-target sums of raw counts and cell numbers  -> pseudobulk context (data/processed)
  * a fixed random subset of single cells            -> evaluation set for holding the line
    (all cells of `--n-eval-targets` targets, capped   out like the challenge does
    per target, plus a sample of non-targeting cells)  (data/raw/xatlas/<line>_eval_cells.h5ad)

Only cells that pass the dual-guide filter are used. Genes are restricted to the union of
the challenge panel and the genes the other sources measured (mapped through the 2024-A
GRCh38 annotation the atlas was aligned to); everything else is discarded to keep the
accumulator small. Progress is checkpointed so the run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from zsp.config import load_config
from zsp.data import CONTROL, Context
from zsp.store import save_context

REPO = "Xaira-Therapeutics/X-Atlas-Orion"
CONTROL_LABEL = "Non-Targeting"


def hf_listing(line: str) -> list[dict]:
    """Batch files for one cell line, with their sha256 from the LFS metadata."""
    out = subprocess.run(
        ["curl", "-sL", f"https://huggingface.co/api/datasets/{REPO}?blobs=true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = [
        {"path": s["rfilename"], "size": s.get("size"), "sha256": (s.get("lfs") or {}).get("oid")}
        for s in json.loads(out)["siblings"]
        if s["rfilename"].startswith(f"data/{line}_Batch") and s["rfilename"].endswith(".parquet")
    ]
    return sorted(files, key=lambda f: int(f["path"].rsplit("Batch", 1)[1].split(".")[0]))


def download(path: str, dest: Path, sha256: str | None) -> None:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    for _attempt in range(4):
        subprocess.run(["curl", "-sL", "-C", "-", "-o", str(dest), url], check=False)
        if sha256 is None:
            return
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() == sha256:
            return
        dest.unlink(missing_ok=True)
    raise RuntimeError(f"checksum mismatch after retries: {path}")


def batch_to_csr(table, n_tokens: int) -> sp.csr_matrix:
    tok = table.column("gene_token_id").combine_chunks()
    val = table.column("gene_expression").combine_chunks()
    indptr = tok.offsets.to_numpy().astype(np.int64)
    return sp.csr_matrix(
        (val.values.to_numpy().astype(np.float32), tok.values.to_numpy().astype(np.int32), indptr),
        shape=(len(table), n_tokens),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, choices=["HCT116", "HEK293T"])
    ap.add_argument("--n-eval-targets", type=int, default=250)
    ap.add_argument("--max-cells-per-eval-target", type=int, default=100)
    ap.add_argument("--n-control-cells", type=int, default=20000)
    ap.add_argument("--max-batches", type=int, default=None, help="debug: stop early")
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    raw = cfg.path("raw") / "xatlas"
    raw.mkdir(parents=True, exist_ok=True)
    name = args.line.lower()

    # --- gene universe: challenge panel + genes any existing source measured
    gm = pd.read_parquet(raw / "gene_metadata.parquet").set_index("gene_token_id").sort_index()
    panel = set(pd.read_csv(cfg.path("raw") / "vcc" / "gene_names.csv")["gene_name"].astype(str))
    src_genes: set[str] = set()
    for n in cfg["contexts"]:
        g = cfg.path("processed") / f"{n}_genes.parquet"
        if g.exists():
            src_genes |= set(pd.read_parquet(g).index)
    keep = (gm["gene_name"].isin(panel) | gm["ensembl_id"].isin(src_genes)).to_numpy()
    keep_tok = np.flatnonzero(keep)
    genes = pd.Index(gm["ensembl_id"].to_numpy()[keep_tok], name="gene_id")
    dup = genes.duplicated()
    keep_tok, genes = keep_tok[~dup], genes[~dup]
    symbols = pd.Series(gm["gene_name"].to_numpy()[keep_tok], index=genes)
    print(
        f"{args.line}: keeping {len(genes):,} of {len(gm):,} genes "
        f"({int(symbols.isin(panel).sum()):,} panel symbols matched)",
        flush=True,
    )

    # --- targets: library order; evaluation subset fixed before any data is seen
    lib = pd.read_csv(raw / "guide_library.csv")
    targets = pd.Index(sorted(lib["target_gene_name"].astype(str).unique()))
    k562 = cfg.path("processed") / "k562_pert.parquet"
    eligible = targets.intersection(pd.read_parquet(k562).index) if k562.exists() else targets
    eval_targets = set(rng.choice(np.asarray(eligible), args.n_eval_targets, replace=False))
    t_pos = pd.Series(np.arange(len(targets)), index=targets)

    ckpt = raw / f"{name}_checkpoint.pkl"
    if ckpt.exists():
        state = pickle.load(open(ckpt, "rb"))
        print(f"resuming after {len(state['done'])} batches", flush=True)
    else:
        state = {
            "done": [],
            "sums": np.zeros((len(targets), len(genes)), dtype=np.float64),
            "n_cells": np.zeros(len(targets), dtype=np.int64),
            "control_sum": np.zeros(len(genes), dtype=np.float64),
            "n_control": 0,
            "eval_blocks": [],
            "eval_labels": [],
            "eval_per_target": {},
            "n_eval_control": 0,
            "n_control_seen": 0,
        }

    files = hf_listing(args.line)
    if args.max_batches:
        files = files[: args.max_batches]
    total_gb = sum(f["size"] or 0 for f in files) / 1e9
    print(f"{len(files)} batches, {total_gb:.1f} GB to stream", flush=True)
    # expected control cells ~ 5 % of cells; sample so that about n_control_cells are kept
    p_ctrl = min(1.0, args.n_control_cells / (0.05 * 18000 * max(len(files), 1)))

    for i, f in enumerate(files):
        if f["path"] in state["done"]:
            continue
        dest = raw / Path(f["path"]).name
        if not dest.exists():
            download(f["path"], dest, f["sha256"])
        table = pq.read_table(dest)
        table = table.filter(table.column("pass_guide_filter").to_numpy() == 1)
        X = batch_to_csr(table, len(gm))[:, keep_tok].tocsr()
        tgt = table.column("gene_target").to_numpy(zero_copy_only=False).astype(str)
        is_ctrl = tgt == CONTROL_LABEL
        known = ~is_ctrl & pd.Index(tgt).isin(targets)
        # pseudobulk sums per target via a one-hot indicator
        rows = t_pos.loc[tgt[known]].to_numpy()
        ind = sp.csr_matrix(
            (np.ones(known.sum()), (rows, np.flatnonzero(known))), shape=(len(targets), X.shape[0])
        )
        state["sums"] += (ind @ X).toarray()
        state["n_cells"] += np.bincount(rows, minlength=len(targets))
        state["control_sum"] += np.asarray(X[is_ctrl].sum(axis=0)).ravel()
        state["n_control"] += int(is_ctrl.sum())
        # evaluation cells
        take, labels = [], []
        for j in np.flatnonzero(is_ctrl):
            state["n_control_seen"] += 1
            if rng.random() < p_ctrl:
                take.append(j)
                labels.append(CONTROL)
                state["n_eval_control"] += 1
        for j in np.flatnonzero(known):
            t = tgt[j]
            if (
                t in eval_targets
                and state["eval_per_target"].get(t, 0) < args.max_cells_per_eval_target
            ):
                state["eval_per_target"][t] = state["eval_per_target"].get(t, 0) + 1
                take.append(j)
                labels.append(t)
        if take:
            state["eval_blocks"].append(X[np.array(take)])
            state["eval_labels"].extend(labels)
        state["done"].append(f["path"])
        if not args.keep_files:
            dest.unlink(missing_ok=True)
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            pickle.dump(state, open(ckpt, "wb"), protocol=5)
        print(
            f"[{i + 1}/{len(files)}] {Path(f['path']).name}: {X.shape[0]:,} cells, "
            f"{int(is_ctrl.sum())} control; cumulative eval cells {len(state['eval_labels']):,}",
            flush=True,
        )

    # --- write the context (means per cell) and the evaluation cells
    n = state["n_cells"]
    ok = n > 0
    pert = pd.DataFrame(state["sums"][ok] / n[ok, None], index=targets[ok], columns=genes)
    control = state["control_sum"] / max(state["n_control"], 1)
    ctx = Context(name, genes, symbols, control, pert, pd.Series(n[ok], index=targets[ok]))
    save_context(ctx, cfg.path("processed"))
    ev = ad.AnnData(
        sp.vstack(state["eval_blocks"]).tocsr(),
        obs=pd.DataFrame({"gene": state["eval_labels"]}),
        var=pd.DataFrame({"gene_name": symbols.to_numpy()}, index=genes),
    )
    ev.obs_names = [f"cell{i}" for i in range(ev.n_obs)]
    ev.write_h5ad(raw / f"{name}_eval_cells.h5ad")
    print(
        f"wrote context {name}: {int(ok.sum()):,} targets (median {int(np.median(n[ok]))} cells), "
        f"{state['n_control']:,} control cells; eval set {ev.n_obs:,} cells "
        f"({state['n_eval_control']:,} control, {len(state['eval_per_target'])} targets)",
        flush=True,
    )


if __name__ == "__main__":
    main()
