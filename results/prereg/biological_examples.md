# Pre-registered biological examples

Written before any per-target score was inspected (only aggregate tables had been seen).
The targets below are drawn from the 16 knockdowns that the local benchmark evaluates in
*both* held-out lines (HepG2 and Jurkat). Hypotheses are fixed here; the evaluation script
(`scripts/09_prereg_examples.py`) fills in the numbers and does not change them.

## A. Responses expected to be shared across cell contexts

Core-machinery knockdowns whose primary consequence is a conserved stress programme:

| Target | Why the response should transfer |
|---|---|
| AARS | alanyl-tRNA synthetase; loss triggers the integrated stress response (ATF4 targets) in every proliferating line |
| POP7 | RNase P/MRP subunit; tRNA/rRNA processing, ribosome-biogenesis stress |
| GTF3C6 | Pol III transcription factor; tRNA supply |
| ACTR2 | Arp2/3 complex; actin branching, growth |

Prediction: for these, mean transfer from other lines gives above-median `pds`, `fid` and
`reach` in both held-out lines.

## B. Responses expected to depend on the cell context

| Target | Why the response should differ between lines |
|---|---|
| HSD17B12 | very-long-chain fatty-acid elongation; lipid handling is hepatocyte-specialised (HepG2) and minor in T-ALL (Jurkat) |
| GABPA | ETS-family TF; its target set (mitochondrial biogenesis, cell-cycle genes) is wired differently in myeloid/lymphoid vs epithelial lines |
| E4F1 | TF at the p53/cell-cycle interface; HepG2 is p53-wild-type, Jurkat is p53-mutant |
| GLB1 | lysosomal beta-galactosidase; lysosomal load differs strongly between the two lines |

Prediction: for these, transfer scores are below the median of the evaluated targets in at
least one of the two lines, and the two lines' scores differ more than for group A.

## C. Where the model should decline

Rule, not names: in each held-out line the 20 targets with the highest `disagreement_norm`
(sources disagree about the *shape* of the specific response) are predicted to have lower
`reach` and `jac` than the remaining targets. If this fails, the signal is not usable for
declining predictions, whatever its rank correlation says.

## Evaluation (fixed)

- Model: mean transfer (the reference model), per-target scores from `results.csv`.
- Group A vs B: Mann-Whitney U on each metric, both lines pooled (n = 8 vs 8 → low power; the
  test is reported with its p-value, and the direction of the difference is the claim).
- Rule C: Mann-Whitney U, 20 vs 180 targets per line.
- Pathway enrichment may be used to *describe* what was predicted; it is not the test.
