# Zero-shot perturbation response prediction, with a confidence score per prediction

Given only the resting state of a cell line it has never seen perturbed, can a model predict what each CRISPRi knockdown will do to it, and say which of its predictions to trust?

This is the problem posed by the [2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026) (Arc Institute; task paper in [*Cell*](https://www.cell.com/cell/fulltext/S0092-8674(26)00931-1)): predict knockdown responses in six cell lines whose perturbation data are hidden, from their unperturbed expression alone. A 2025 *Nature Methods* study showed that deep-learning and foundation models [do not yet beat simple linear baselines](https://doi.org/10.1038/s41592-025-02772-6) at predicting perturbation effects. This repository therefore starts from strong transfer baselines, measures why transfer fails across cell contexts, and evaluates candidate uncertainty scores so that a user knows which knockdowns the model is guessing at.

Built from September 2026 onwards, at the start of my M1, and still in progress: the final test set is released on 22 October 2026.

## Results

### Challenge leaderboard (validation contexts A to C)

Scores are the organisers' baseline-normalised metrics (0 = matches their reference baseline, negative = worse). `results/tables/leaderboard.csv` keeps every entry.

| Entry | Rank | Overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|
| mean transfer, 4 screens (25 Sep) | 572 | 0.069 | 0.40 | 0.00 | 0.014 | −0.05 | 0.08 | −0.02 |
| calibrated transfer, 4 screens (25 Sep) | 605 | 0.056 | 0.29 | 0.00 | 0.012 | −0.02 | 0.08 | −0.02 |
| **weighted transfer, 6 screens (26 Sep)** | **559** | **0.079** | 0.42 | 0.00 | **0.095** | −0.18 | **0.16** | −0.02 |

The first two entries used only the genes shared by all four public screens (6,203 of the 18,533 panel genes) and specific effects from K562 alone. The third adds the two X-Atlas screens (every target measured in at least 3 lines, 18,291 genes predicted) and weights sources by basal similarity. Fold-change accuracy and direction reach rose, direction fidelity fell, the same trade the local benchmark shows when sources are added (below).

### Coverage: what the data can and cannot support

`scripts/07_coverage_audit.py` maps every panel gene and every challenge target to the sources that measured it ("not measured" is NaN throughout, never zero):

| | Genes | Targets |
|---|---|---|
| Challenge panel | 18,533 | 300 |
| Detected (at least 1% of control cells) | 13,859 | |
| Measured in at least one of the four essential or genome-wide screens (Replogle K562/RPE1, Nadig HepG2/Jurkat) | 10,917 | 272 (all K562-only) |
| Measured in at least one of six screens (adding X-Atlas HCT116/HEK293T) | **18,291** | **300** (each in at least 3 lines) |
| Measured in all six | 6,199 | 0 |
| Detected but measured nowhere | 96 | 0 |

The 300 validation targets are non-essential genes: none is in the 2,393-gene essential libraries used for RPE1, HepG2 and Jurkat, so with those four screens the target-specific signal came from a single line. The two genome-wide X-Atlas/Orion screens (HCT116 and HEK293T, [Huang et al. 2025](https://doi.org/10.1101/2025.06.11.659105), CC BY-NC-SA 4.0) close that gap. `scripts/00_xatlas_pseudobulk.py` streams their 126 GB release batch by batch into pseudobulk contexts and a fixed evaluation subset of cells, keeping under 1 GB on disk.

### Local benchmark (raw `cell-eval2` metrics, 200 held-out perturbations per line)

Held-out lines are scored on every gene they released (HepG2 9,624; Jurkat 8,882; HCT116 19,202); sources contribute all genes they measured. Hyper-parameters were chosen by the leave-one-source-out rule (`src/zsp/loso.py`) before any of these scores was seen. Higher is better except `mse` and `nmae`; 95% bootstrap CIs over perturbations are in `results/tables/local_benchmark.csv`.

**Six screens (five sources per held-out line):**

| Held out | Model | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| HepG2 | mean transfer | 0.83 | **0.85** | 0.83 | 0.31 | 0.43 | **0.16** |
| HepG2 | weighted transfer | 0.83 | 0.89 | 0.81 | 0.38 | 0.46 | 0.13 |
| HepG2 | weighted median | **0.84** | 0.97 | 0.87 | 0.28 | 0.36 | 0.11 |
| HepG2 | norm-restored transfer | 0.83 | 1.00 | **0.80** | **0.44** | **0.51** | 0.13 |
| Jurkat | mean transfer | 0.84 | 1.22 | 0.82 | 0.29 | 0.46 | 0.11 |
| Jurkat | weighted transfer | **0.84** | 1.25 | 0.81 | 0.33 | 0.49 | 0.12 |
| Jurkat | weighted median | 0.84 | **1.22** | 0.85 | 0.24 | 0.40 | 0.10 |
| Jurkat | norm-restored transfer | 0.84 | 1.73 | **0.80** | **0.50** | **0.55** | **0.13** |

Going from three to five sources cut mean transfer's expression error (HepG2 1.03 to 0.85, Jurkat 1.90 to 1.22) and lowered its direction fidelity (0.45 to 0.31, 0.49 to 0.29): averaging more lines shrinks the magnitudes the DE tests need. Weighting sources by basal similarity recovers part of that fidelity in both lines; it is the model behind the rank-559 entry.

**Restoring the magnitude is the round-4 result.** Averaging five sources gives the right direction but a shrunken response. Rescaling each target's consensus response to the size a single source typically shows (norm-restored transfer) raises direction fidelity from 0.31 to 0.44 on HepG2 and from 0.29 to 0.50 on Jurkat, raises reach by 0.08 to 0.09, and lowers `nmae` in both lines, at the cost of expression error (`mse` 0.85 to 1.00 and 1.22 to 1.73). The per-gene weighted median, the other way to keep a single source's magnitude, does not help: the median of five noisy responses is smaller than their mean, not larger. By the pre-registered summary (mean of the oriented metrics) norm-restored transfer beats weighted transfer on HepG2 (0.018 vs 0.016) and loses on Jurkat (−0.085 vs −0.047), entirely through `mse`. On the leaderboard the `mse` term has read exactly 0.00 for all three entries so far, so this trade is the next entry to test there.

**Hold-out from another study: HCT116 (X-Atlas).** The four Replogle and Nadig screens share a lab lineage; HCT116 was produced by a different lab with a different protocol and chemistry, so holding it out (with HEK293T from the same release still among the five sources) asks whether transfer survives a change of study. Scored on 19,202 genes, 100 cells per perturbation:

| Held out | Model | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| HCT116 | control | 0.51 | **1.45** | 1.02 | 0.06 | 0.28 | **0.37** |
| HCT116 | mean transfer | 0.64 | 5.16 | **0.74** | **0.49** | 0.59 | 0.23 |
| HCT116 | weighted transfer | **0.65** | 4.91 | 0.74 | 0.46 | **0.61** | 0.19 |

Two things differ from HepG2 and Jurkat. First, the transferred responses are three to four times too large (`mse` 5.2 against 1.4 for predicting no change): HCT116's measured knockdown effects are far weaker than the sources', and no model here predicts a per-context response scale. Second, 73 of the 200 held-out perturbations have no significant DE gene in the real data, and `cell-eval2` scores an empty predicted set against an empty real set as a Jaccard of 1, which is where the control's 0.37 comes from; only 36 perturbations have enough real DE genes for `nmae`. Direction fidelity and reach, computed only where the real data has DE genes, are the highest of the three lines. The response scale, already the open problem on Jurkat, is therefore the dominant error on a line from another study.

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

What the numbers say:

- **Transferring the measured response as-is is still the model to beat.** Every way of damping it (a global scale, per-gene transfer coefficients, or the generic/specific decomposition) lowers expression error (`mse`) at the cost of every DE-based metric, because those metrics come from significance tests that need the full magnitude. The leave-one-source-out proxies pointed the same way for `mse`, but four of the six proxies are scale-invariant, so the pre-registered rule chose the strongest damping and the cell-level scorer overrules it. A selection rule for this challenge has to include a magnitude-sensitive DE proxy.
- **Weighting sources by basal similarity is the one change that helps without a trade-off.** On Jurkat it cuts `mse` from 1.90 to 1.64 while matching or beating mean transfer on `reach`, `nmae` and `jac`; on HepG2 the two are within CI of each other on everything.
- **Mean transfer overshoots Jurkat** (`mse` 1.90, worse than predicting no change at 1.04): the sources' responses are larger than Jurkat's. HCT116 (above) shows the same failure at a larger scale. A per-context response scale is the right fix in principle, but it must be predicted from the basal state without reducing the DE magnitude the scorer rewards. This is an open problem in this repository.
- **None of five candidate uncertainty signals supports selective prediction.** Keeping the 50% most-certain targets changes every metric by less than 0.06 for every signal, averaged over the three held-out lines (`results/tables/uncertainty_gain_at_half_coverage.csv`); raw disagreement and effect magnitude lower `pds` when used for selection, because the targets sources agree on are the weak-effect ones. The rank correlations that looked promising in an earlier version of this benchmark were a magnitude confound, not a usable confidence score.
- **Pre-registered biology** (`results/prereg/biological_examples.md`, fixed before any per-target score was read): the four "shared-response" knockdowns (AARS, POP7, GTF3C6, ACTR2) did not transfer better than the four "context-dependent" ones (HSD17B12, GABPA, E4F1, GLB1), p ≥ 0.44 on every metric with n = 8 vs 8, and the "decline" rule (highest normalised disagreement, therefore lower `reach` and `jac`) was rejected: those targets scored higher on `reach` in HepG2 and Jurkat, and lower in HCT116 only by 0.06 (one-sided p = 0.23). All results are recorded as they came out (`results/tables/prereg_*.csv`).

## Data

Public genome-scale CRISPRi Perturb-seq screens, each reduced to per-perturbation pseudobulk profiles (mean raw UMI counts per cell) plus the non-targeting control profile:

| Context | Source | Perturbations | Role |
|---|---|---|---|
| K562 (CML) | [Replogle et al., 2022](https://doi.org/10.1016/j.cell.2022.05.013), genome-wide | 9,866 | training |
| RPE1 (retinal epithelium) | Replogle et al., 2022 | 2,393 | training |
| HepG2 (liver) | [Nadig et al., 2025](https://doi.org/10.1038/s41588-025-02169-3) | 2,393 | held-out (single cells kept for scoring) |
| Jurkat (T-ALL) | Nadig et al., 2025 | 2,393 | held-out (single cells kept for scoring) |
| HCT116, HEK293T | [Huang et al., 2025](https://doi.org/10.1101/2025.06.11.659105) (X-Atlas/Orion), genome-wide | see coverage table | training |

Contexts are aligned on their shared genes (Ensembl IDs). `scripts/download_data.sh` fetches everything from figshare and GEO (~16 GB, resumable).

## Local benchmark: the challenge replayed on public data

For each held-out line, the model sees the other contexts' full perturbation responses plus only the held-out line's non-targeting cells, which is exactly the information the challenge provides. Predictions are emitted as single cells (resampled from that line's own control cells, rescaled by the predicted fold change, integer-rounded) and scored with the organisers' own tool, `cell-eval2` under the `vcc2026` preset: the six competition metrics, unweighted. The held-out line's control cells are split in half, one half as the "basal state" the model is allowed to see and the other half reserved for scoring, so the scorer never sees a cell the model was given.

## Method

Every knockdown response in a source context is decomposed as

    LFC(p) = generic + specific(p)

where the generic response (the median LFC over all knockdowns) captures what every CRISPRi perturbation does to that cell line (stress, slowed proliferation, the cost of the machinery itself) and specific(p) is what the target does. The models compared:

| Model | Prediction for the unseen context |
|---|---|
| Control | no change (the floor every model must beat) |
| Mean transfer | average of the target's measured LFC across source contexts |
| Weighted transfer | the same average, weighted by each source's basal similarity to the target line (softmax of Pearson similarity, temperature chosen by the leave-one-source-out rule) |
| Norm-restored transfer | weighted transfer, then each target's response is rescaled so that its L2 norm equals the weighted mean of the source responses' norms: consensus direction, single-source magnitude |
| Weighted median transfer | per-gene weighted median of the source responses, same weights |
| Scaled transfer | mean transfer times one global response scale (0.4 by the leave-one-source-out rule) |
| Gene-scaled transfer | mean transfer with a per-gene transfer coefficient, ridge-fitted across sources |
| Calibrated transfer | mean source generic + shrunk mean specific effect; genes not expressed in the target's basal state are gated to zero; the target's own transcript is set to the typical knockdown depth; uncertainty = disagreement between sources on the specific effect, inflated for targets few sources measured |

Everything is fitted on source contexts only; the held-out line contributes nothing but its basal profile.

## Design decisions

- **Scored by the organisers' code, not by a re-implementation.** `cell-eval2` is a dependency, the `vcc2026` preset is used verbatim, and the run log is kept beside every result.
- **No leakage from the held-out line.** Its perturbation cells are never read by any model; its control cells are split so the scorer's controls are disjoint from the model's.
- **Genes are indexed by symbol in every scored file**, because the scorer excludes each perturbation's own target gene by name; a silent mismatch would inflate scores, and the tool refuses to run rather than guess.
- **Uncertainty is evaluated, not only emitted.** Per-perturbation scores are joined with the model's uncertainty to ask whether it ranks its own failures (`scripts/04_*`).
- **Config-driven and tested.** Every number comes from `configs/default.yaml`; the pseudobulk, decomposition, transfer, gating and cell-emission steps have unit tests.

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

### Docker

```bash
docker build -t zsp . && docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" zsp make test
```

## Limitations

- **Few source contexts.** Transfer quality is bounded by how many cell lines have measured a target; a target seen only in K562 is a K562 result, not a cross-context one.
- **The response scale is inherited, not predicted.** Every transfer model carries the sources' effect sizes into the new line; on HCT116 they are three to four times too large. Predicting a per-context scale from the basal state, without shrinking the DE magnitude the scorer rewards, is the main open problem.
- **Released gene sets differ between screens** (8.2k to 9.6k genes each; 10,917 panel genes in at least one), so the local benchmark under-represents lowly expressed and lineage-specific genes; the challenge panel has 18,533.
- **Pseudobulk training discards cell-level structure.** Bimodal responses (a perturbation that only affects a subpopulation) are averaged away; the emitted cells inherit only the target line's basal heterogeneity.
- **Local scores are not leaderboard scores.** The challenge normalises each metric against organiser-built baselines and averages over three hidden lines; the local numbers here are raw `cell-eval2` metrics on public lines and are only comparable between models in this repository.
- **No validated uncertainty yet.** The five signals tried are all functions of the sources and the basal state; none ranks failures usefully. The analysis is kept so that the negative result stays visible.

## References

- Replogle et al., *Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq*, [Cell 2022](https://doi.org/10.1016/j.cell.2022.05.013).
- Nadig et al., *Transcriptome-wide analysis of differential expression in perturbation atlases*, [Nat Genet 2025](https://doi.org/10.1038/s41588-025-02169-3).
- Huang et al., *X-Atlas/Orion*, [bioRxiv 2025](https://doi.org/10.1101/2025.06.11.659105).
- Ahlmann-Eltze, Huber & Anders, *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines*, [Nat Methods 2025](https://doi.org/10.1038/s41592-025-02772-6).
- Roohani et al., *Virtual Cell Challenge: Toward a Turing test for the virtual cell*, [Cell 2025](https://doi.org/10.1016/j.cell.2025.06.008); the 2026 edition, [Cell 2026](https://www.cell.com/cell/fulltext/S0092-8674(26)00931-1).

## License

MIT © 2026 Matin Gerami
