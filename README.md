# Zero-shot perturbation response prediction — with a confidence score per prediction

**Given only the resting state of a cell line it has never seen perturbed, can a model predict
what each CRISPRi knockdown will do to it — and say which of its predictions to trust?**

This is the problem posed by the
[2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026)
(Arc Institute; task paper in [*Cell*](https://www.cell.com/cell/fulltext/S0092-8674(26)00931-1)):
predict knockdown responses in six cell lines whose perturbation data are hidden, from their
unperturbed expression alone. It is a hard problem for a specific reason — a 2025 *Nature
Methods* study showed that deep-learning and foundation models
[do not yet beat simple linear baselines](https://doi.org/10.1038/s41592-025-02772-6) at
predicting perturbation effects. So this repository starts where that finding points: build
strong, honest transfer baselines first, understand *why* transfer fails across cell contexts,
and attach a calibrated uncertainty to every prediction so a downstream user knows which
knockdowns the model is guessing at.

## Results

### Challenge leaderboard (validation contexts A–C)

Scores are the organisers' baseline-normalised metrics (0 = matches their reference baseline,
negative = worse); `results/tables/leaderboard.csv` keeps every entry.

| Entry | Rank | Overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|
| mean transfer, 4 screens (25 Sep) | 572 | 0.069 | 0.40 | 0.00 | 0.014 | −0.05 | 0.08 | −0.02 |
| calibrated transfer, 4 screens (25 Sep) | 605 | 0.056 | 0.29 | 0.00 | 0.012 | −0.02 | 0.08 | −0.02 |
| **weighted transfer, 6 screens (26 Sep)** | **559** | **0.079** | 0.42 | 0.00 | **0.095** | −0.18 | **0.16** | −0.02 |

The first two entries used only the genes shared by all four public screens (6,203 of the
18,533 panel genes) and specific effects from K562 alone. The third adds the two X-Atlas
screens (every target measured in ≥3 lines, 18,291 genes predicted) and weights sources by
basal similarity; fold-change accuracy and direction reach rose, direction fidelity fell — the
same trade the local benchmark showed when sources were added (below).

### Coverage — what the data can and cannot support

`scripts/07_coverage_audit.py` maps every panel gene and every challenge target to the sources
that measured it ("not measured" is NaN throughout, never zero):

| | Genes | Targets |
|---|---|---|
| Challenge panel | 18,533 | 300 |
| Detected (≥1 % of control cells) | 13,859 | – |
| Measured in ≥1 of the four essential/genome-wide screens (Replogle K562/RPE1, Nadig HepG2/Jurkat) | 10,917 | 272 (all K562-only) |
| Measured in ≥1 of six screens (+ X-Atlas HCT116/HEK293T) | **18,291** | **300** (each in ≥3 lines) |
| Measured in all six | 6,199 | 0 |
| Detected but measured nowhere | 96 | 0 |

The 300 validation targets are non-essential genes: none is in the 2,393-gene essential
libraries used for RPE1, HepG2 and Jurkat, so with those four screens the target-specific
signal came from a single line. The two genome-wide X-Atlas/Orion screens (HCT116 and
HEK293T, [Huang et al. 2025](https://doi.org/10.1101/2025.06.11.659105), CC BY-NC-SA 4.0)
close that gap; `scripts/00_xatlas_pseudobulk.py` streams their 126 GB release batch by batch
into pseudobulk contexts and a fixed evaluation subset of cells, keeping under 1 GB on disk.

### Local benchmark (raw `cell-eval2` metrics, 200 held-out perturbations per line)

Held-out lines are scored on every gene they released (HepG2 9,624; Jurkat 8,882); sources
contribute all genes they measured. Hyper-parameters were chosen by the leave-one-source-out
rule (`src/zsp/loso.py`) before any of these scores was seen. Higher is better except
`mse`/`nmae`; 95 % bootstrap CIs over perturbations are in `results/tables/local_benchmark.csv`.

**Six screens (five sources per held-out line):**

| Held out | Model | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| HepG2 | mean transfer | **0.83** | **0.85** | 0.83 | 0.31 | 0.43 | **0.16** |
| HepG2 | weighted transfer | 0.83 | 0.89 | **0.81** | **0.38** | **0.46** | 0.13 |
| Jurkat | mean transfer | 0.84 | **1.22** | 0.82 | 0.29 | 0.46 | 0.11 |
| Jurkat | weighted transfer | **0.84** | 1.25 | **0.81** | **0.33** | **0.49** | **0.12** |

Going from three to five sources cut mean transfer's expression error (HepG2 1.03 → 0.85,
Jurkat 1.90 → 1.22) and lowered its direction fidelity (0.45 → 0.31, 0.49 → 0.29): averaging
more lines shrinks the magnitudes the DE tests need. Weighting sources by basal similarity
recovers part of that fidelity in both lines and is the better model by the pre-defined
summary; it is the model behind the rank-559 entry.

**Four screens (three sources per held-out line), all models:**

| Held out | Model | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| HepG2 | control | 0.49 | 1.07 | 1.02 | 0.01 | 0.09 | 0.11 |
| HepG2 | mean transfer | 0.82 | 1.03 | **0.81** | **0.45** | 0.47 | **0.13** |
| HepG2 | weighted transfer | **0.83** | 1.04 | 0.82 | 0.41 | **0.48** | 0.13 |
| HepG2 | scaled (×0.4) | 0.79 | 0.88 | 0.90 | 0.08 | 0.29 | 0.11 |
| HepG2 | gene-scaled | 0.81 | 0.99 | 0.90 | 0.26 | 0.31 | 0.12 |
| HepG2 | calibrated (α 0.4) | 0.71 | **0.85** | 0.87 | 0.13 | 0.34 | 0.12 |
| Jurkat | control | 0.52 | 1.04 | 1.00 | 0.02 | 0.08 | 0.08 |
| Jurkat | mean transfer | **0.84** | 1.90 | 0.82 | **0.49** | 0.49 | 0.10 |
| Jurkat | weighted transfer | 0.83 | 1.64 | **0.81** | 0.45 | **0.51** | **0.11** |
| Jurkat | scaled (×0.4) | 0.80 | **0.98** | 0.90 | 0.09 | 0.33 | 0.08 |
| Jurkat | gene-scaled | 0.78 | 1.16 | 0.88 | 0.27 | 0.40 | 0.09 |
| Jurkat | calibrated (α 0.4) | 0.69 | 1.08 | 0.88 | 0.18 | 0.37 | 0.08 |

**What the numbers say**

- **Transferring the measured response as-is is still the model to beat.** Every way of
  damping it — a global scale, per-gene transfer coefficients, or the generic/specific
  decomposition — buys expression error (`mse`) at the cost of every DE-based metric, because
  those metrics are computed from significance tests that need the full magnitude. The
  leave-one-source-out proxies pointed the same way for `mse` but four of the six proxies are
  scale-invariant, so the pre-registered rule chose the strongest damping; the cell-level
  scorer overrules it. A selection rule for this challenge has to include a magnitude-sensitive
  DE proxy.
- **Weighting sources by basal similarity is the one change that helps without a trade-off**:
  on Jurkat it cuts `mse` from 1.90 to 1.64 while matching or beating mean transfer on
  `reach`, `nmae` and `jac`; on HepG2 the two are within CI of each other on everything.
- **Mean transfer overshoots Jurkat** (`mse` 1.90, worse than predicting no change at 1.04):
  the sources' responses are larger than Jurkat's. A per-context response scale is the right
  fix in principle, but it must be predicted from the basal state without touching the DE
  magnitude the scorer rewards — an open problem, not a solved one.
- **None of five candidate uncertainty signals supports selective prediction.** Keeping the
  50 % most-certain targets changes every metric by less than 0.07 for every signal
  (`results/tables/uncertainty_gain_at_half_coverage.csv`); raw disagreement and effect
  magnitude *lower* `pds` when used for selection, because the targets sources agree on are the
  weak-effect ones. The rank correlations that looked promising in an earlier version of this
  benchmark were a magnitude confound, not a usable confidence score.
- **Pre-registered biology** (`results/prereg/biological_examples.md`, fixed before any
  per-target score was read): the four "shared-response" knockdowns (AARS, POP7, GTF3C6,
  ACTR2) did not transfer better than the four "context-dependent" ones (HSD17B12, GABPA, E4F1,
  GLB1) — p ≥ 0.44 on every metric with n = 8 vs 8 — and the "decline" rule (highest
  normalised disagreement → lower `reach`/`jac`) was rejected: those targets scored *higher*
  on `reach` in both lines. Both are recorded as they came out (`results/tables/prereg_*.csv`).

## Data

Public genome-scale CRISPRi Perturb-seq screens, each reduced to per-perturbation pseudobulk
profiles (mean raw UMI counts per cell) plus the non-targeting control profile:

| Context | Source | Perturbations | Role |
|---|---|---|---|
| K562 (CML) | [Replogle et al., 2022](https://doi.org/10.1016/j.cell.2022.05.013), genome-wide | 9,866 | training |
| RPE1 (retinal epithelium) | Replogle et al., 2022 | 2,393 | training |
| HepG2 (liver) | [Nadig et al., 2025](https://doi.org/10.1038/s41588-025-02169-3) | 2,393 | held-out (single cells kept for scoring) |
| Jurkat (T-ALL) | Nadig et al., 2025 | 2,393 | held-out (single cells kept for scoring) |

Contexts are aligned on their shared genes (Ensembl IDs). `scripts/download_data.sh` fetches
everything from figshare and GEO (~16 GB, resumable).

## Local benchmark = the challenge, replayed on public data

For each held-out line, the model sees the other contexts' full perturbation responses plus
**only the held-out line's non-targeting cells** — exactly the information the challenge
provides. Predictions are emitted as single cells (resampled from that line's own control
cells, rescaled by the predicted fold change, integer-rounded) and scored with the organisers'
own tool, `cell-eval2` under the `vcc2026` preset: the six competition metrics, unweighted.
The held-out line's control cells are split in half — one half is the "basal state" the model
is allowed to see, the other half is reserved for scoring — so the scorer never sees a cell the
model was given.

## Method

Every knockdown response in a source context is decomposed as

    LFC(p) = generic + specific(p)

where the **generic** response (the median LFC over all knockdowns) captures what every
CRISPRi perturbation does to that cell line — stress, slowed proliferation, the cost of the
machinery itself — and **specific(p)** is what the target actually does. The models compared:

| Model | Prediction for the unseen context |
|---|---|
| Control | no change (the floor every model must beat) |
| Mean transfer | average of the target's measured LFC across source contexts |
| Calibrated transfer | mean source generic + shrunk mean specific effect; genes not expressed in the target's basal state are gated to zero; the target's own transcript is set to the typical knockdown depth; **uncertainty** = disagreement between sources on the specific effect, inflated for targets few sources measured |

Everything is fitted on source contexts only; the held-out line contributes nothing but its
basal profile.

## Design decisions

- **Scored by the organisers' code, not by a re-implementation.** `cell-eval2` is a dependency,
  the `vcc2026` preset is used verbatim, and the run log is kept beside every result.
- **No leakage from the held-out line.** Its perturbation cells are never read by any model;
  its control cells are split so the scorer's controls are disjoint from the model's.
- **Genes are indexed by symbol in every scored file**, because the scorer excludes each
  perturbation's own target gene by name — a silent mismatch would inflate scores, and the
  tool refuses to run rather than guess.
- **Uncertainty is evaluated, not just emitted.** Per-perturbation scores are joined with the
  model's uncertainty to ask whether it ranks its own failures (`scripts/04_*`).
- **Config-driven and tested.** Every number comes from `configs/default.yaml`; the
  pseudobulk, decomposition, transfer, gating and cell-emission steps have unit tests.

## Reproduce

```bash
uv sync
make data          # ~16 GB from figshare + GEO (resumable)
make contexts      # pseudobulk every screen
uv run python scripts/00_xatlas_pseudobulk.py --line HCT116   # streams ~47 GB, keeps ~1 GB
make coverage      # coverage map (results/tables/coverage_*.csv)
make loso          # pre-registered hyper-parameter selection on sources only
make benchmark     # leave-one-context-out, scored with cell-eval2 (~25 min per model and line)
make collect       # results/tables/local_benchmark.csv, with bootstrap CIs
make uncertainty   # candidate uncertainty signals vs per-target scores
uv run python scripts/09_prereg_examples.py
make test
```

## Limitations

- **Few source contexts.** Transfer quality is bounded by how many cell lines have measured
  a target; a target seen only in K562 is a K562 story, not a cross-context one. The X-Atlas
  screens (HCT116, HEK293T) are being added for exactly this reason.
- **Released gene sets differ between screens** (8.2k–9.6k genes each; 10,917 panel genes in
  at least one), so the local benchmark under-represents lowly expressed and lineage-specific
  genes; the challenge panel has 18,533.
- **Pseudobulk training discards cell-level structure.** Bimodal responses (a perturbation
  that only affects a subpopulation) are averaged away; the emitted cells inherit only the
  target line's basal heterogeneity.
- **Local scores are not leaderboard scores.** The challenge normalises each metric against
  organiser-built baselines and averages over three hidden lines; the local numbers here are
  raw `cell-eval2` metrics on public lines and are only comparable *between models in this
  repository*.
- **No validated uncertainty yet.** The five signals tried are all functions of the sources
  and the basal state; none ranks failures usefully, and the analysis is kept so the negative
  result is visible rather than hidden.


## References

- Replogle et al., *Mapping information-rich genotype-phenotype landscapes with genome-scale
  Perturb-seq*, [Cell 2022](https://doi.org/10.1016/j.cell.2022.05.013).
- Nadig et al., *Transcriptome-wide analysis of differential expression in perturbation
  atlases*, [Nat Genet 2025](https://doi.org/10.1038/s41588-025-02169-3).
- Ahlmann-Eltze, Huber & Anders, *Deep-learning-based gene perturbation effect prediction does
  not yet outperform simple linear baselines*, [Nat Methods 2025](https://doi.org/10.1038/s41592-025-02772-6).
- Roohani et al., *Virtual Cell Challenge: Toward a Turing test for the virtual cell*,
  [Cell 2025](https://doi.org/10.1016/j.cell.2025.06.008); the 2026 edition,
  [Cell 2026](https://www.cell.com/cell/fulltext/S0092-8674(26)00931-1).

## License

MIT © 2026 Matin Gerami
