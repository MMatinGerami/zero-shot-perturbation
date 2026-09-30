"""How transferable is a knockdown's response between cell lines, once measurement noise is removed?

For each single-cell screen (HepG2, Jurkat, HCT116, HEK293T), every cell is assigned at random to
half A or B, and each half gives its own pseudobulk LFC per perturbation (against its own half of
the control cells). For a pair of lines and a perturbation both screened:

  r_obs     gene-wise Pearson r between the two lines' full-sample LFCs (genes expressed in both);
  rel       split-half reliability of each line's LFC (Spearman-Brown, src/zsp/ceiling.py);
  r_true    r_obs / sqrt(rel_1 * rel_2), the classical correction for attenuation: the
            correlation the two responses would have if both were measured without noise.

r_true is what a *perfect* transfer from one line could reach against a noise-free truth in the
other; it is the ceiling on the transfer idea itself, separate from the noise ceiling of
scripts/10. Reported raw and centred (each line's mean response over the shared perturbations
subtracted first). Perturbations with a reliability below MIN_REL in either line are left out of
r_true (the correction divides by a noisy small number there); their count is reported.

Output: results/tables/transferability_pairs.csv, results/tables/transferability_perturbations.csv
"""

from __future__ import annotations

import itertools

import anndata as ad
import numpy as np
import pandas as pd

from zsp.ceiling import rowwise_pearson, spearman_brown
from zsp.config import load_config
from zsp.data import cp10k, log_fold_change

LINES = ["hepg2", "jurkat", "hct116", "hek293t"]
CONTROL = "non-targeting"
MIN_CP10K = 1.0
MIN_REL = 0.2
MIN_SHARED = 30
CHUNK = 20_000


def halves(path, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Full-sample and split-half LFCs (perturbations x genes) and the control CP10k."""
    a = ad.read_h5ad(path, backed="r")
    target = a.obs["gene"].astype(str).to_numpy()
    codes, uniq = pd.factorize(target)
    half = rng.integers(0, 2, len(target))
    key = codes * 2 + half
    sums = np.zeros((2 * len(uniq), a.n_vars))
    for start in range(0, a.n_obs, CHUNK):
        block = a.X[start : start + CHUNK]
        X = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        np.add.at(sums, key[start : start + CHUNK], X.astype(np.float64))
    n = np.bincount(key, minlength=2 * len(uniq)).astype(np.float64)
    sa, sb, na, nb = sums[0::2], sums[1::2], n[0::2], n[1::2]
    ok = (na > 0) & (nb > 0)
    genes = pd.Index(a.var_names.astype(str))
    c = int(np.flatnonzero(uniq == CONTROL)[0])
    ctrl_a, ctrl_b = sa[c] / na[c], sb[c] / nb[c]
    ctrl = (sa[c] + sb[c]) / (na[c] + nb[c])
    keep = ok & (uniq != CONTROL)
    perts = pd.Index(uniq[keep])
    full = log_fold_change((sa[keep] + sb[keep]) / (na[keep] + nb[keep])[:, None], ctrl)
    lfc_a = log_fold_change(sa[keep] / na[keep][:, None], ctrl_a)
    lfc_b = log_fold_change(sb[keep] / nb[keep][:, None], ctrl_b)
    mk = lambda x: pd.DataFrame(x, index=perts, columns=genes)  # noqa: E731
    return mk(full), mk(lfc_a), mk(lfc_b), pd.Series(cp10k(ctrl)[0], index=genes)


def reliability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return spearman_brown(rowwise_pearson(a, b))


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    tables = cfg.path("results") / "tables"
    data = {}
    for line in LINES:
        path = cfg.path("raw") / cfg["contexts"][line]["file"]
        data[line] = halves(path, rng)
        print(f"{line}: {data[line][0].shape[0]} perturbations", flush=True)

    pair_rows, pert_rows = [], []
    for l1, l2 in itertools.combinations(LINES, 2):
        f1, a1, b1, c1 = data[l1]
        f2, a2, b2, c2 = data[l2]
        perts = f1.index.intersection(f2.index)
        genes = f1.columns.intersection(f2.columns)
        genes = genes[
            (c1.loc[genes] >= MIN_CP10K).to_numpy() & (c2.loc[genes] >= MIN_CP10K).to_numpy()
        ]
        if len(perts) < MIN_SHARED:
            print(f"{l1}-{l2}: {len(perts)} shared perturbations, skipped", flush=True)
            continue
        frames = dict(f1=f1, a1=a1, b1=b1, f2=f2, a2=a2, b2=b2)
        m = {k: v.loc[perts, genes].to_numpy() for k, v in frames.items()}
        row = dict(line_1=l1, line_2=l2, n_shared=len(perts), n_genes=len(genes))
        for kind in ["raw", "centred"]:
            x = {k: (v - v.mean(0) if kind == "centred" else v) for k, v in m.items()}
            r_obs = rowwise_pearson(x["f1"], x["f2"])
            rel1, rel2 = reliability(x["a1"], x["b1"]), reliability(x["a2"], x["b2"])
            good = (rel1 >= MIN_REL) & (rel2 >= MIN_REL)
            with np.errstate(all="ignore"):
                r_true = np.where(good, r_obs / np.sqrt(rel1 * rel2), np.nan)
            row |= {
                f"r_obs_{kind}_median": float(np.nanmedian(r_obs)),
                f"rel_1_{kind}_median": float(np.nanmedian(rel1)),
                f"rel_2_{kind}_median": float(np.nanmedian(rel2)),
                f"n_reliable_{kind}": int(good.sum()),
                f"r_true_{kind}_median": float(np.nanmedian(r_true)),
                f"r_true_{kind}_q25": float(np.nanquantile(r_true, 0.25)),
                f"r_true_{kind}_q75": float(np.nanquantile(r_true, 0.75)),
            }
            for p, ro, ra, rb, rt in zip(perts, r_obs, rel1, rel2, r_true, strict=True):
                pert_rows.append(
                    dict(
                        line_1=l1,
                        line_2=l2,
                        kind=kind,
                        target=p,
                        r_obs=ro,
                        rel_1=ra,
                        rel_2=rb,
                        r_true=rt,
                    )
                )
        pair_rows.append(row)
        print(
            f"{l1}-{l2}: n={len(perts)}  r_obs {row['r_obs_centred_median']:.2f}  "
            f"r_true {row['r_true_centred_median']:.2f} (centred, "
            f"{row['n_reliable_centred']} reliable)",
            flush=True,
        )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(tables / "transferability_pairs.csv", index=False)
    pd.DataFrame(pert_rows).to_csv(tables / "transferability_perturbations.csv", index=False)
    print(pairs.round(3).T.to_string())


if __name__ == "__main__":
    main()
