"""Save / load Contexts as parquet + npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zsp.data import Context


def save_context(ctx: Context, outdir: Path) -> None:
    ctx.pert.to_parquet(outdir / f"{ctx.name}_pert.parquet")
    pd.DataFrame(
        {"symbol": ctx.symbols.loc[ctx.genes].to_numpy(), "control": ctx.control},
        index=ctx.genes,
    ).to_parquet(outdir / f"{ctx.name}_genes.parquet")
    ctx.n_cells.rename("n_cells").to_frame().to_parquet(outdir / f"{ctx.name}_ncells.parquet")


def load_context(name: str, outdir: Path) -> Context:
    pert = pd.read_parquet(outdir / f"{name}_pert.parquet")
    g = pd.read_parquet(outdir / f"{name}_genes.parquet")
    n = pd.read_parquet(outdir / f"{name}_ncells.parquet")["n_cells"]
    genes = pd.Index(g.index, name="gene_id")
    pert.columns = genes
    return Context(name, genes, g["symbol"], g["control"].to_numpy(np.float64), pert, n)
