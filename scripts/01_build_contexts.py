"""Pseudobulk every public screen into a Context (data/processed/<name>_*.parquet)."""

from __future__ import annotations

import sys

from zsp.config import load_config
from zsp.data import context_from_bulk, context_from_singlecell
from zsp.store import save_context


def main(names: list[str]) -> None:
    cfg = load_config()
    raw, out = cfg.path("raw"), cfg.path("processed")
    for name, spec in cfg["contexts"].items():
        if names and name not in names:
            continue
        if spec["kind"] == "prebuilt":  # written by its own streaming script
            continue
        loader = context_from_bulk if spec["kind"] == "bulk" else context_from_singlecell
        ctx = loader(raw / spec["file"], name)
        save_context(ctx, out)
        print(
            f"{name}: {ctx.pert.shape[0]} perturbations x {len(ctx.genes)} genes",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
