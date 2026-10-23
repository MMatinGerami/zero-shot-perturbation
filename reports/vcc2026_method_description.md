# Virtual Cell Challenge 2026: method description (draft)

Team: Matin Gerami (individual entry; M1 GENIOMHE-AI, Université Paris-Saclay).
Final entry: `final_v10`, submitted 23 Oct 2026. Code: github.com/MMatinGerami/zero-shot-perturbation.

This draft follows the items the rules ask finalists to describe. It is written so that it can
be sent as is if the entry is selected as a finalist.

## Summary

The entry is a transfer model. For each target gene it estimates the knockdown response from
public CRISPRi Perturb-seq screens in other cell lines, combines those estimates with a
confidence gate, and renders the combined response as single cells drawn from the test
context's own control cells. There is no neural network. Every hyper-parameter was chosen on
public held-out cell lines or on the validation leaderboard, never on the final contexts.

## Datasets used in training

All public. Each screen is reduced to one pseudobulk profile per target (mean raw UMI counts
per cell) and one for its non-targeting controls.

| Screen | Cell line | Reference |
|---|---|---|
| Replogle et al. 2022, genome-wide | K562 | doi:10.1016/j.cell.2022.05.013 |
| Replogle et al. 2022, essential genes | RPE1 | doi:10.1016/j.cell.2022.05.013 |
| Nadig et al. 2025, essential genes | HepG2, Jurkat | doi:10.1038/s41588-025-02169-3 |
| X-Atlas/Orion (Huang et al. 2025), genome-wide | HCT116, HEK293T | doi:10.1101/2025.06.11.659105 |
| Virtual Cell Challenge 2025 | H1 hESC | doi:10.1016/j.cell.2025.06.008 |
| Zhu et al. 2026, genome-wide, published DE statistics | primary CD4+ T cells (48 h stimulation) | doi:10.1016/j.cell.2026.08.002 |

The challenge's own control cells for contexts A to F were used only as model inputs (the
basal state of each context), as the challenge intends.

## Data processing and model inputs

- Counts are library-size normalised to counts per 10,000 and turned into log2 fold changes
  against each screen's own controls (pseudocount 0.1). Screens are aligned on the union of
  their genes; a gene a screen did not measure is missing, never zero.
- The CD4 data are released as DE statistics rather than counts; their fold changes were
  converted to the same scale through the published base means.
- For a test context, the inputs are its control cells and the list of 300 target symbols.

## Model

1. **Norm-restored transfer.** For each target, the response is the equal-weight mean of the
   log fold changes of the screens that measured it, rescaled so that its overall size equals
   the typical size of a single screen's response. Averaging keeps the direction the screens
   agree on; the rescaling undoes the shrinkage that averaging disagreeing screens causes.
2. **Confidence gate.** Each (target, gene) entry is multiplied by min(1, |z| / 3), where z is
   the mean across screens divided by its standard error. Genes the screens disagree about are
   shrunk toward no change.
3. **Generic response.** The response shared by all knockdowns of a context (its mean over
   targets) is scaled by 0.7.
4. **Control-axis deflation.** The predicted response is projected off the top 50 principal
   axes of the test context's own control cells, so that the prediction does not move along
   directions of ordinary cell-to-cell variation.

## Non-learned components and rendering

These steps are stated explicitly because they depend on how the scorer works rather than on
biology.

- **Amplitude.** The perturbation-specific part of the response is scaled by one number per
  context, chosen so that the spread of the predicted profiles equals 2.5 times the sampling
  noise of a 400-cell group of that context's control cells. The value 2.5 was chosen so that
  the number of genes the scorer's own Wilcoxon test calls significant per knockdown is in the
  hundreds, as the validation leaderboard indicated the reference does.
- **Cell emission.** For each target, 400 control cells of the context are resampled and
  rescaled gene by gene so that the group's pooled counts equal the predicted mean exactly.
  30 % of the cells carry the response and 70 % stay at their control state, scaled so the
  pooled mean is unchanged (uneven knockdown is a known property of CRISPRi). Counts are made
  integer with exact column totals.

## Training, selection and inference

There is no gradient training. Choices were made in pre-registered rounds, each written down
before its results were read (`results/prereg/`):

- Leave-one-screen-out selection on the public screens, scored with cell-eval2 under the
  `vcc2026` preset against held-out HepG2, Jurkat, HCT116 and H1 cells.
- Validation leaderboard entries between 25 Sep and 18 Oct 2026 (seven scored entries).
- Inference on the final contexts reads only their control cells.

The repository README reports negative and null results alongside the adopted ones, including
two ceilings measured from the data: noise-corrected, a knockdown's gene-wise response
correlates only about 0.47 between two cell lines, which bounds any transfer model.

## Compute

One laptop (Apple M5 Pro, 24 GB, CPU only). Building a final submission takes about 30
minutes; no GPU was used.

## Open points to confirm with the organisers before sending

- Whether averaging measured public responses counts as a model trained on public data, as the
  rules allow, rather than as published results pasted into the entry. The method above is
  described so that the answer can be judged from it.
