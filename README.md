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

### Challenge leaderboard (validation contexts A–C, 25 Sep 2026)

Scores are the organisers' baseline-normalised metrics (0 = matches their reference baseline,
negative = worse); `results/tables/leaderboard.csv` keeps every entry.

| Entry | Rank | Overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|
| mean transfer | 572 | **0.069** | 0.40 | 0.00 | 0.014 | −0.05 | 0.08 | −0.02 |
| calibrated transfer (α = 1, no gate) | 605 | 0.056 | 0.29 | 0.00 | 0.012 | −0.02 | 0.08 | −0.02 |

Two facts about the validation panel bound what any transfer model can do here: 272 of the
300 targets are measured only in the K562 genome-wide screen (none are in the 2,393-gene
essential libraries used for RPE1, HepG2 and Jurkat), so the target-specific signal comes from
a single source; and the four public screens share only 6,203 of the 18,533 panel genes, so
the remaining genes are predicted unchanged.

### Local benchmark (raw `cell-eval2` metrics, 200 held-out perturbations per line)

Higher is better except `mse` and `nmae`. `results/tables/local_benchmark.csv`.

| Held out | Model | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| HepG2 | control | 0.50 | 1.08 | 1.01 | 0.01 | 0.08 | 0.13 |
| HepG2 | mean transfer | **0.84** | 0.80 | **0.71** | **0.48** | **0.63** | **0.16** |
| HepG2 | calibrated (α = 0.4, LOSO) | 0.70 | **0.74** | 0.82 | 0.15 | 0.45 | 0.15 |
| Jurkat | control | 0.50 | 1.05 | 1.01 | 0.01 | 0.10 | 0.12 |
| Jurkat | mean transfer | **0.84** | 1.77 | **0.75** | **0.52** | **0.55** | **0.14** |
| Jurkat | calibrated (α = 0.2, LOSO) | 0.64 | **1.02** | 0.91 | 0.13 | 0.28 | 0.06 |

**What the numbers say**

- **Mean transfer is the model to beat**, locally and on the leaderboard. The decomposition
  with shrinkage wins only on expression error (`mse`), and loses on every DE-based metric.
  The shrinkage factor was chosen by leave-one-source-out *Pearson correlation* of pseudobulk
  profiles — the wrong objective: the DE metrics reward getting the *magnitude* of the
  response right, and shrinking specific effects toward zero removes exactly that.
- **Discrimination and expression error pull in opposite directions.** On Jurkat, mean
  transfer's `mse` (1.77) is worse than predicting no change (1.05): the K562/RPE1/HepG2
  responses overshoot Jurkat's, while the shrunk model stays under the control floor (1.02).
  A per-context response *scale* — not a global shrinkage — is what the data ask for.
- **The uncertainty ranks failures on the DE metrics but not on discrimination.** Keeping
  the 50 % most-certain targets raises Jaccard from 0.15 to 0.27 and direction reach from
  0.45 to 0.58 on HepG2 (0.28 → 0.33 on Jurkat), yet *lowers* `pds` on both lines
  (0.70 → 0.54, 0.64 → 0.55); the rank correlation between uncertainty and `pds` is positive
  (ρ = 0.48, 0.35). Source disagreement is measured in absolute log-fold-change units, so it
  scales with effect size — and large-effect knockdowns are both the ones sources disagree
  about and the easiest to tell apart. As defined, the score is a magnitude proxy, not an
  error estimate; it needs normalising by effect size before it can be trusted for
  selective prediction (`results/figures/uncertainty.png`, `results/tables/uncertainty_*.csv`).

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
make benchmark     # leave-one-context-out, scored with cell-eval2 (~1 h on a laptop)
make collect       # results/tables/local_benchmark.csv
make uncertainty   # does uncertainty predict failure?
make test
```

## Limitations

- **Three source contexts.** Transfer quality is bounded by how many cell lines have measured
  a target; a target seen only in K562 is a K562 story, not a cross-context one. More public
  screens (the challenge allows any) are the cheapest improvement available.
- **Shared-gene intersection is small** (~6.8k genes across four lines) because the public
  screens were filtered differently; the challenge panel is 18.5k genes, so the local
  benchmark under-represents lowly expressed and lineage-specific genes.
- **Pseudobulk training discards cell-level structure.** Bimodal responses (a perturbation
  that only affects a subpopulation) are averaged away; the emitted cells inherit only the
  target line's basal heterogeneity.
- **Local scores are not leaderboard scores.** The challenge normalises each metric against
  organiser-built baselines and averages over three hidden lines; the local numbers here are
  raw `cell-eval2` metrics on public lines and are only comparable *between models in this
  repository*.
- **Uncertainty is source disagreement**, so a target all sources agree on but that behaves
  differently in the new line is confidently wrong — the failure mode the selective-prediction
  analysis is there to measure, not to hide.


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
