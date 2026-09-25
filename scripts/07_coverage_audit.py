"""Coverage map: which genes, targets and contexts support each prediction.

"Not measured" is distinguished from "measured at zero" throughout: a gene absent from a
source's released matrix is NaN in that source, never 0. Writes

  results/tables/coverage_genes.csv    one row per challenge-panel gene: measured in which
                                       sources, detected in the challenge controls
  results/tables/coverage_targets.csv  one row per challenge target: measured in which sources
  results/tables/coverage_summary.csv  the headline numbers quoted in the README
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from zsp.config import load_config
from zsp.data import align_union
from zsp.store import load_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/raw/vcc")
    args = ap.parse_args()
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle)

    names = [n for n in cfg["contexts"] if (proc / f"{n}_pert.parquet").exists()]
    contexts = align_union([load_context(n, proc) for n in names])
    union = contexts[0]
    symbols = union.symbols.loc[union.genes]

    panel = pd.read_csv(bundle / "gene_names.csv")["gene_name"].astype(str)
    targets = pd.read_csv(bundle / "pert_counts.csv")["target_gene"].astype(str)

    # --- genes: measured per source (by symbol), detected in the challenge controls
    sym_measured = pd.DataFrame(
        {c.name: pd.Series(c.measured, index=symbols.to_numpy()) for c in contexts}
    )
    sym_measured = sym_measured[~sym_measured.index.duplicated()]
    genes = pd.DataFrame(index=pd.Index(panel, name="gene"))
    for c in contexts:
        genes[f"measured_{c.name}"] = sym_measured[c.name].reindex(panel).fillna(False).to_numpy()
    genes["n_sources"] = genes.filter(like="measured_").sum(axis=1)
    for ctx in ("A", "B", "C"):
        a = ad.read_h5ad(bundle / f"context_{ctx}.h5ad")
        frac = np.asarray((a.X > 0).mean(axis=0)).ravel()
        genes[f"frac_cells_detected_{ctx}"] = (
            pd.Series(frac, index=a.var_names).reindex(panel).to_numpy()
        )
    detected = genes.filter(like="frac_cells_detected_").max(axis=1) >= 0.01
    genes["detected_in_controls"] = detected
    genes.to_csv(tables / "coverage_genes.csv")

    # --- targets: which sources measured the knockdown
    tgt = pd.DataFrame(index=pd.Index(targets, name="target_gene"))
    for c in contexts:
        tgt[f"measured_{c.name}"] = targets.isin(c.pert.index).to_numpy()
    tgt["n_sources"] = tgt.sum(axis=1)
    tgt["own_gene_in_panel"] = targets.isin(panel).to_numpy()
    tgt.to_csv(tables / "coverage_targets.csv")

    summary = {
        "panel_genes": len(panel),
        "panel_genes_detected_in_controls_1pct": int(detected.sum()),
        "panel_genes_measured_in_any_source": int((genes["n_sources"] > 0).sum()),
        "panel_genes_measured_in_all_sources": int((genes["n_sources"] == len(contexts)).sum()),
        "panel_genes_detected_but_unmeasured": int((detected & (genes["n_sources"] == 0)).sum()),
        "targets": len(targets),
        "targets_measured_in_any_source": int((tgt["n_sources"] > 0).sum()),
        "targets_measured_in_one_source_only": int((tgt["n_sources"] == 1).sum()),
    }
    for c in contexts:
        summary[f"genes_released_{c.name}"] = int(c.measured.sum())
        summary[f"targets_measured_{c.name}"] = int(tgt[f"measured_{c.name}"].sum())
    pd.Series(summary, name="value").to_csv(tables / "coverage_summary.csv")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
