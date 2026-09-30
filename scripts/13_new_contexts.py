"""Round 9: two more source contexts, built into the same pseudobulk representation.

  h1    VCC 2025 H1 hESC CRISPRi (Roohani et al., Cell 2025), all three released splits pooled;
        genes are symbols, mapped to Ensembl through the other contexts' own annotations.
  cd4   genome-scale CRISPRi in primary human CD4+ T cells (Zhu et al., Cell 2026,
        doi:10.1016/j.cell.2026.08.002). Only the publisher's differential-expression statistics
        are used; see `cd4` below for how they enter.

Usage: uv run python scripts/13_new_contexts.py h1 [cd4]
"""

from __future__ import annotations

import sys

from zsp.config import load_config
from zsp.data import context_from_symbol_cells, ensembl_by_symbol
from zsp.store import load_context, save_context

EXISTING = ["k562", "rpe1", "hepg2", "jurkat", "hct116", "hek293t"]


def h1(cfg) -> None:
    raw, proc = cfg.path("raw"), cfg.path("processed")
    ens = ensembl_by_symbol([load_context(n, proc) for n in EXISTING])
    paths = [raw / "h1" / f"adata_{s}.h5ad" for s in ("Training", "Validation", "Test")]
    ctx = context_from_symbol_cells(paths, "h1", ens)
    save_context(ctx, proc)
    print(
        f"h1: {ctx.pert.shape[0]} perturbations x {len(ctx.genes)} genes, "
        f"median {ctx.n_cells.median():.0f} cells per perturbation",
        flush=True,
    )


def main(which: list[str]) -> None:
    cfg = load_config()
    for name in which or ["h1"]:
        {"h1": h1}[name](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
