"""Fast pseudobulk evaluation of transfer variants on a held-out line.

Implements the two expression members of the vcc2026 suite as specified in cell-eval2's
docs/vcc2026_metrics: perturbation discrimination (`pds_cosine`) and the unbiased, capped,
normalised expression error (`expr_mse_unbiased_capped_norm`), on pooled profiles, together
with the two anchors the leaderboard scales by: the baseline (the context's mean response
given to every perturbation) and the replicate (one half of the real cells scored against the
other, five splits). A prediction enters as mean counts per cell, so its side carries no
sampling noise, which is what an emitter of independent cells earns back through the
jackknife credit; the result is therefore a slightly optimistic stand-in for the scorer's
number, and relative comparisons between variants are what it is for.

The four differential-expression members need cells and the Wilcoxon test and stay with the
full scorer (scripts/02). Rank candidates here in seconds, confirm there.

Inputs come from scripts/17 (results/cache). Variants are built from the cached source
responses: source weights, norm restoration, a global shrinkage, a cross-source agreement
gate, and deflation of the predicted response along the held-out line's own control-cell
axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.config import load_config
from zsp.models import lfc_to_counts

TS = 5e4  # the scorer's profile normalisation


# ----------------------------------------------------------------------------- real side


def profile(counts: np.ndarray) -> np.ndarray:
    """log1p of a total-normalised pooled profile; rows are groups."""
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    return np.log1p(TS * counts / counts.sum(1, keepdims=True))


def jackknife(X: sp.csr_matrix, chunk: int = 1000) -> float:
    """Delete-one-cell jackknife of a group's pooled profile (the spec's C)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    P = np.asarray(X.sum(0)).ravel().astype(np.float64)
    S = P.sum()
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    s1 = np.zeros_like(P)
    s2 = np.zeros_like(P)
    for start in range(0, n, chunk):
        Y = X[start : start + chunk].toarray().astype(np.float64)
        V = np.log1p(TS * (P[None, :] - Y) / (S - lib[start : start + chunk])[:, None])
        s1 += V.sum(0)
        s2 += (V**2).sum(0)
    return float((n - 1) / n * (s2 - s1**2 / n).sum())


def group_stats(X: sp.csr_matrix) -> tuple[np.ndarray, float, np.ndarray]:
    """(pooled profile, jackknife C, mean counts per cell) of one group."""
    P = np.asarray(X.sum(0)).ravel().astype(np.float64)
    return profile(P)[0], jackknife(X), P / X.shape[0]


class RealSide:
    """Everything the two expression members read from the held-out line's real cells."""

    def __init__(self, held: str, cache: Path, work: Path):
        path = cache / f"{held}_real_stats.npz"
        real = ad.read_h5ad(work / f"{held}_real.h5ad")
        self.genes = pd.Index(real.var_names)
        target = real.obs["target"].astype(str).to_numpy()
        self.perts = sorted(set(target) - {"non-targeting"})
        X = sp.csr_matrix(real.X)
        if path.exists():
            z = np.load(path)
            if list(z["perts"]) == self.perts:
                for k in z.files:
                    setattr(self, k, z[k])
                self.perts = list(self.perts)
                self._split_source = (X, target)
                return
        print(f"{held}: computing real-side statistics for {len(self.perts)} perturbations")
        ctrl = X[target == "non-targeting"]
        self.b_ctrl, self.C_ctrl, self.m_ctrl = group_stats(ctrl)
        rows = [group_stats(X[target == p]) for p in self.perts]
        self.b_pert = np.stack([r[0] for r in rows])
        self.C_pert = np.array([r[1] for r in rows])
        self.m_pert = np.stack([r[2] for r in rows])
        self.own = self.genes.get_indexer(self.perts)  # -1 when the target gene is not measured
        self._split_source = (X, target)
        self.anchors = self._anchors(X, target)
        np.savez_compressed(
            path,
            perts=np.array(self.perts),
            b_ctrl=self.b_ctrl,
            C_ctrl=self.C_ctrl,
            m_ctrl=self.m_ctrl,
            b_pert=self.b_pert,
            C_pert=self.C_pert,
            m_pert=self.m_pert,
            own=self.own,
            anchors=self.anchors,
        )

    # the two anchors -----------------------------------------------------------------
    def _anchors(self, X, target, n_splits: int = 5) -> np.ndarray:
        """[b_pds, b_mse, r_pds, r_mse]: baseline = mean response for every perturbation
        (no sampling credit, its profiles being identical); replicate = half against half,
        each half with its own control, averaged over splits."""
        base = profile(self.m_pert.mean(0, keepdims=True)).repeat(len(self.perts), 0)
        b_pds = self.pds(base)
        b_mse = self.mse(base, None)
        r_pds, r_mse = [], []
        for seed in range(n_splits):
            rng = np.random.default_rng(seed)
            halves = {}
            for name in ["non-targeting", *self.perts]:
                idx = np.flatnonzero(target == name)
                rng.shuffle(idx)
                halves[name] = (np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2 :]))
            ref = RealSplit(self, X, halves, side=1)
            sub = [group_stats(X[halves[p][0]]) for p in self.perts]
            b_hat = np.stack([s[0] for s in sub])
            C_hat = np.array([s[1] for s in sub])
            b_ctrl_a = group_stats(X[halves["non-targeting"][0]])[0]
            r_pds.append(ref.pds(b_hat, b_ctrl_pred=b_ctrl_a))
            r_mse.append(ref.mse(b_hat, C_hat))
        return np.array([b_pds, b_mse, float(np.mean(r_pds)), float(np.mean(r_mse))])

    # metrics -------------------------------------------------------------------------
    def _panel_mask(self) -> np.ndarray:
        keep = np.ones(len(self.genes), bool)
        keep[self.own[self.own >= 0]] = False
        return keep

    def pds(self, b_hat: np.ndarray, b_ctrl_pred: np.ndarray | None = None) -> float:
        """Mean of 1 - rank/(n-1) of the true match by cosine distance, all panel target
        genes removed. The predicted effect is measured against the reference control
        unless the caller supplies its own (the replicate's halves each use their own)."""
        keep = self._panel_mask()
        bc = self.b_ctrl if b_ctrl_pred is None else b_ctrl_pred
        dh = (b_hat - bc)[:, keep]
        dr = (self.b_pert - self.b_ctrl)[:, keep]
        nh = np.linalg.norm(dh, axis=1)
        nr = np.linalg.norm(dr, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = 1 - (dh @ dr.T) / np.outer(nh, nr)
        d[(nh == 0)[:, None] | (nr == 0)[None, :]] = 1.0
        n = len(self.perts)
        own = np.diag(d)
        k = (d < own[:, None]).sum(1) + 0.5 * ((d == own[:, None]).sum(1) - 1)
        return float(np.mean(1 - k / (n - 1)))

    def mse(self, b_hat: np.ndarray, C_hat: np.ndarray | None) -> float:
        """Sum over the panel of corrected squared distances to the real profiles, over the
        sum of corrected real effect sizes. The prediction's sampling credit per perturbation
        is min(C_hat, C_p) (nothing when it has no cells, C_hat None), and the total credit
        is capped by rho = min(1, B / total credit) with B the across-perturbation spread of
        the predicted profiles, as in the specification."""
        n = len(self.perts)
        N, D, credit, G = (np.zeros(n) for _ in range(4))
        for i in range(n):
            mask = np.ones(len(self.genes), bool)
            if self.own[i] >= 0:
                mask[self.own[i]] = False
            G[i] = mask.sum()
            credit[i] = 0.0 if C_hat is None else min(C_hat[i], self.C_pert[i]) / G[i]
            N[i] = (((b_hat[i] - self.b_pert[i])[mask] ** 2).sum() - self.C_pert[i]) / G[i]
            D[i] = (
                ((self.b_pert[i] - self.b_ctrl)[mask] ** 2).sum() - self.C_pert[i] - self.C_ctrl
            ) / G[i]
        rho = 1.0
        if credit.sum() > 0:
            inv = 1 / G
            wmean = (b_hat * inv[:, None]).sum(0) / inv.sum()
            B = (((b_hat - wmean) ** 2) * inv[:, None]).sum()
            rho = float(min(1.0, B / credit.sum()))
        self.last = {
            "rho": rho,
            "credit_share": float(credit.sum() / D.sum()),
            "max_credit_share": float((self.C_pert / G).sum() / D.sum()),
        }
        return float((N.sum() - rho * credit.sum()) / D.sum())

    def scaled(self, pds_raw: float, mse_raw: float) -> tuple[float, float]:
        b_pds, b_mse, r_pds, r_mse = self.anchors
        s_pds = (pds_raw - b_pds) / (r_pds - b_pds)
        s_mse = float(np.clip((mse_raw - b_mse) / (r_mse - b_mse), 0, 1))
        return s_pds, s_mse


class RealSplit(RealSide):
    """A RealSide whose reference is one half of the cells (for the replicate anchor)."""

    def __init__(self, full: RealSide, X, halves: dict, side: int):
        self.genes, self.perts, self.own = full.genes, full.perts, full.own
        self.b_ctrl, self.C_ctrl, _ = group_stats(X[halves["non-targeting"][side]])
        rows = [group_stats(X[halves[p][side]]) for p in self.perts]
        self.b_pert = np.stack([r[0] for r in rows])
        self.C_pert = np.array([r[1] for r in rows])


# --------------------------------------------------------------------------- model side


class Cache:
    def __init__(self, cache: Path, names: list[str]):
        g = pd.read_parquet(cache / "genes.parquet")
        self.genes = pd.Index(g.index)
        self.symbols = g["symbol"]
        self.perts = pd.Index((cache / "perts.txt").read_text().split())
        self.src = {}
        for n in names:
            z = np.load(cache / f"{n}.npz")
            self.src[n] = {k: z[k] for k in z.files}

    def stack(self, names: list[str], perts: list[str]):
        pos = self.perts.get_indexer(perts)
        if (pos < 0).any():
            raise KeyError("perturbation not in cache")
        L = np.stack([self.src[n]["lfc"][pos].astype(np.float64) for n in names])
        generic = np.stack([self.src[n]["generic"].astype(np.float64) for n in names])
        basal = np.stack([self.src[n]["basal"].astype(np.float64) for n in names])
        return L, generic, basal


def basal_similarity(target_basal: np.ndarray, basal: np.ndarray) -> np.ndarray:
    t = np.log1p(target_basal)
    out = np.zeros(len(basal))
    for i, b in enumerate(np.log1p(basal)):
        ok = np.isfinite(t) & np.isfinite(b)
        out[i] = np.corrcoef(t[ok], b[ok])[0, 1] if ok.sum() >= 10 else 0.0
    return out


def transfer(L, generic, w, restore_norm: bool):
    """Weighted NaN-aware mean of source responses (missing targets get the weighted generic
    response), optionally rescaled per target to a single source's typical norm. Mirrors
    zsp.models._weighted_transfer and NormRestoredTransfer."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    present = np.isfinite(L)
    num = (np.where(present, L, 0.0) * w[:, None, None]).sum(0)
    den = (present * w[:, None, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = num / den
    gok = np.isfinite(generic)
    gen = (np.where(gok, generic, 0.0) * w[:, None]).sum(0) / np.maximum(
        (gok * w[:, None]).sum(0), 1e-12
    )
    missing = ~np.isfinite(pred).any(1)
    pred[missing] = gen
    pred = np.nan_to_num(pred)
    if restore_norm:
        norms = np.sqrt((np.nan_to_num(L) ** 2).sum(2))
        seen = present.any(2)
        target_norm = (norms * seen * w[:, None]).sum(0) / np.maximum(
            (seen * w[:, None]).sum(0), 1e-12
        )
        own = np.linalg.norm(pred, axis=1)
        scale = np.where((own > 0) & seen.any(0), target_norm / np.maximum(own, 1e-12), 1.0)
        pred = pred * scale[:, None]
    return pred, present


def agreement_gate(L, w, kind: str, threshold: float) -> np.ndarray:
    """Per (perturbation, gene) multiplier in [0, 1] from how much the sources agree.
    'sign': weighted fraction of sources agreeing with the consensus sign, hard-thresholded;
    'z': |mean| / (sd / sqrt(n)), linearly ramped to 1 at `threshold`. Entries measured by a
    single source pass a 'sign' gate at half strength and get z = |mean| / sd_pooled."""
    w = np.asarray(w, dtype=np.float64)[:, None, None]
    present = np.isfinite(L)
    n = present.sum(0)
    wsum = (present * w).sum(0)
    mean = (np.where(present, L, 0.0) * w).sum(0) / np.maximum(wsum, 1e-12)
    if kind == "sign":
        agree = (np.where(present, np.sign(L) == np.sign(mean)[None], False) * w).sum(0)
        frac = agree / np.maximum(wsum, 1e-12)
        gate = np.where(frac >= threshold, 1.0, 0.0)
        gate = np.where(n == 1, 0.5, gate)
        return np.where(n == 0, 1.0, gate)
    if kind in ("z", "zhard"):
        dev = np.where(present, (L - mean[None]) ** 2, 0.0).sum(0)
        sd = np.sqrt(dev / np.maximum(n - 1, 1))
        pooled = np.sqrt(np.nanmean(dev.sum() / max(n.sum() - (n > 0).sum(), 1)))
        se = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), pooled)
        z = np.abs(mean) / np.maximum(se, 1e-9)
        if kind == "zhard":
            return np.where(n == 0, 1.0, (z >= threshold).astype(float))
        return np.where(n == 0, 1.0, np.clip(z / threshold, 0, 1))
    raise ValueError(kind)


class Deflator:
    """Top-k principal axes of the held-out line's own control cells (log1p of TS-normalised
    cells, centred); `apply` removes eta times a predicted effect's projection on them."""

    def __init__(self, basal_cells: sp.csr_matrix, k: int, seed: int = 0):
        from sklearn.utils.extmath import randomized_svd

        X = basal_cells.toarray().astype(np.float32)
        X = np.log1p(TS * X / np.maximum(X.sum(1, keepdims=True), 1))
        self.mean = X.mean(0)
        _, _, vt = randomized_svd(X - self.mean, n_components=k, random_state=seed)
        self.V = vt.T.astype(np.float64)  # genes x k

    def apply(self, delta: np.ndarray, eta: float) -> np.ndarray:
        return delta - eta * (delta @ self.V) @ self.V.T


_DEFLATORS: dict[int, Deflator] = {}


def predict_variant(cache: Cache, real: RealSide, basal_npz, basal_cells, names, v: dict):
    """Mean counts per cell (perts x held-out genes) for one variant."""
    L, generic, basal = cache.stack(names, real.perts)
    gene_pos = basal_npz["gene_pos"]
    control_mean = basal_npz["control_mean"]
    target_basal = np.full(len(cache.genes), np.nan)
    target_basal[gene_pos] = control_mean / control_mean.sum() * 1e4
    if v.get("temperature") is None:
        w = np.ones(len(names))
    else:
        sim = basal_similarity(target_basal, basal)
        w = np.exp((sim - sim.max()) / v["temperature"])
    pred, _ = transfer(L, generic, w, v.get("restore_norm", False))
    if v.get("drop_generic"):
        gen = np.nan_to_num(np.nanmean(generic, 0))
        pred = pred - gen[None, :]
    if v.get("gate"):
        gate = agreement_gate(L, w, v["gate"], v["gate_threshold"])
        pred = pred * np.maximum(gate, v.get("gate_floor", 0.0))
    pred = pred * v.get("shrink", 1.0)
    if v.get("lfc_floor"):
        pred = np.where(np.abs(pred) < v["lfc_floor"], 0.0, pred)
    v["_nnz_per_pert"] = float((pred[:, gene_pos] != 0).sum(1).mean())
    lfc = pred[:, gene_pos]
    means = lfc_to_counts(lfc, control_mean)
    b_basal = profile(control_mean)[0]
    if v.get("deflate_k") or v.get("rho_target") or v.get("generic_scale") is not None:
        delta = profile(means) - b_basal
        gen = delta.mean(0, keepdims=True)
        if v.get("deflate_k"):
            k = int(v["deflate_k"])
            if k not in _DEFLATORS:
                _DEFLATORS[k] = Deflator(basal_cells, k)
            eta = v.get("deflate_eta", 1.0)
            if v.get("deflate_centred"):
                delta = gen + _DEFLATORS[k].apply(delta - gen, eta)
            else:
                delta = _DEFLATORS[k].apply(delta, eta)
            gen = delta.mean(0, keepdims=True)
        if v.get("generic_scale") is not None:
            delta = v["generic_scale"] * gen + (delta - gen)
            gen = delta.mean(0, keepdims=True)
        if v.get("rho_target"):
            # amplitude of the perturbation-specific part such that the across-perturbation
            # spread B equals rho_target times the sampling credit the prediction can claim;
            # the credit is taken as the real groups' own (what matched cells would earn)
            G = len(real.genes) - 1
            credit = (real.C_pert / G).sum()
            B = ((delta - gen) ** 2).sum() / G
            a = np.sqrt(v["rho_target"] * credit / B) if B > 0 else 1.0
            a = min(a, v.get("max_amplitude", 1.0)) if v.get("cap_amplitude", True) else a
            delta = gen + a * (delta - gen)
            v["_amplitude"] = float(a)
        means = np.expm1(np.clip(b_basal + delta, 0.0, None))
        means = means / means.sum(1, keepdims=True) * control_mean.sum()
    return means


def diagnostics(real: RealSide, b_hat: np.ndarray) -> dict:
    dh = b_hat - real.b_ctrl
    dr = real.b_pert - real.b_ctrl
    err = b_hat - real.b_pert
    shared = err.mean(0)
    gen_h, gen_r = dh.mean(0), dr.mean(0)
    return {
        "energy_ratio": float(np.median(np.linalg.norm(dh, axis=1) / np.linalg.norm(dr, axis=1))),
        "generic_cos": float(gen_h @ gen_r / (np.linalg.norm(gen_h) * np.linalg.norm(gen_r))),
        "generic_norm_ratio": float(np.linalg.norm(gen_h) / np.linalg.norm(gen_r)),
        "shared_error_share": float(len(err) * (shared**2).sum() / (err**2).sum()),
    }


VARIANTS = {
    "control": {"shrink": 0.0},
    "mean": {},
    "v5": {"restore_norm": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", default="h1")
    ap.add_argument("--sources", nargs="*", help="default: every cached context but the held-out")
    ap.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANTS),
        help="names from VARIANTS or inline JSON dicts, e.g. '{\"shrink\":0.4}'",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--credit",
        choices=["none", "full"],
        default="none",
        help="sampling credit assumed for the prediction: none (noise-free profiles, no "
        "cells) or full (cells whose within-group scatter matches the real cells', "
        "C_hat = C_p, and whose pooled profile is exact)",
    )
    args = ap.parse_args()
    cfg = load_config()
    res = cfg.path("results")
    cache_dir, work = res / "cache", res / "local_eval"
    all_names = cfg.context_names(cfg["evaluation"].get("extra_sources", []))
    names = args.sources or [n for n in all_names if n != args.held_out]
    real = RealSide(args.held_out, cache_dir, work)
    cache = Cache(cache_dir, names)
    basal_npz = np.load(cache_dir / f"{args.held_out}_basal.npz")
    if list(basal_npz["perts"]) != real.perts:
        raise RuntimeError("cached basal state was built for a different evaluation set")
    basal_cells = sp.csr_matrix(ad.read_h5ad(cache_dir / f"{args.held_out}_basal_cells.h5ad").X)
    b_pds, b_mse, r_pds, r_mse = real.anchors
    print(
        f"anchors: baseline pds {b_pds:.3f} mse {b_mse:.3f}; "
        f"replicate pds {r_pds:.3f} mse {r_mse:.3f}"
    )
    real.mse(real.b_pert, real.C_pert)
    print(
        "maximal sampling credit as a share of the real effect energy: "
        f"{real.last['max_credit_share']:.3f}"
    )
    rows = []
    for spec in args.variants:
        v = VARIANTS[spec] if spec in VARIANTS else json.loads(spec)
        means = predict_variant(cache, real, basal_npz, basal_cells, names, v)
        b_hat = profile(means)
        C_hat = real.C_pert if args.credit == "full" else None
        pds_raw, mse_raw = real.pds(b_hat), real.mse(b_hat, C_hat)
        s_pds, s_mse = real.scaled(pds_raw, mse_raw)
        row = {
            "variant": spec
            if spec in VARIANTS
            else json.dumps({k: x for k, x in v.items() if not k.startswith("_")}, sort_keys=True),
            "pds_raw": pds_raw,
            "mse_raw": mse_raw,
            "pds": s_pds,
            "mse": s_mse,
            "pds+mse": s_pds + s_mse,
            "credit": args.credit,
            "rho": real.last["rho"],
            **diagnostics(real, b_hat),
        }
        rows.append(row)
        print(
            f"{row['variant']:>60s}  pds {pds_raw:.3f} ({s_pds:+.3f})"
            f"  mse {mse_raw:.3f} ({s_mse:.3f})"
            f"  energy {row['energy_ratio']:.2f}  generic cos {row['generic_cos']:+.2f}"
            f" x{row['generic_norm_ratio']:.2f}  rho {row['rho']:.2f}"
            f"  amp {v.get('_amplitude', 1.0):.2f}  nnz/pert {v.get('_nnz_per_pert', 0):.0f}",
            flush=True,
        )
    out = Path(args.out or res / "tables" / f"fast_eval_{args.held_out}.csv")
    table = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        if "credit" not in old:
            old["credit"] = "none"
        key = lambda t: t.variant + "|" + t.credit.astype(str)  # noqa: E731
        table = pd.concat([old[~key(old).isin(key(table))], table], ignore_index=True)
    table.to_csv(out, index=False)


if __name__ == "__main__":
    main()
