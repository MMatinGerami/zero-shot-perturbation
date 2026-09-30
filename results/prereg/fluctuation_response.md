# Pre-registration: round 8, fluctuation-response prediction from the target line's own control cells

Written 30 Sep 2026, committed before any score of the model below on a held-out line.

## Why

Rounds 5 to 7 showed that reweighting, rescaling or selecting among the six source screens
does not fix the direction error, and the noise ceiling (`scripts/10_noise_ceiling.py`) shows the
error is not measurement noise: the best model recovers 48 % (HepG2), 36 % (Jurkat) and 24 %
(HCT116) of the reproducible gene-wise signal. Every model so far uses the target line only
through its *mean* basal profile. The one other thing the challenge provides is the
cell-to-cell variation of ~18,000 unperturbed cells per line.

## Hypothesis

For a linear stochastic system near steady state, the fluctuation-response relation links the
response to a small push on one variable to the stationary covariance: the response of gene g
to lowering gene p is, to first order, proportional to Cov(g, p) / Var(p) in the unperturbed
state. Applied here: a knockdown of p in the target line shifts gene g by

    fr(g, p) = beta_gp * d,     beta_gp = Cov_target(g, p) / Var_target(p)

estimated on the target line's own control cells (log1p CP10k, log library size regressed out),
with d the typical knockdown depth of the sources (median own-target log2 FC). This is
information specific to the new context that no source screen contains.

Known reasons it may fail, stated in advance: CRISPRi pushes far outside the linear regime;
single-cell covariance is dominated by cell size and cycle; lowly expressed targets have
almost no measurable variance; the relation assumes symmetric regulation.

## Exploratory look already taken (disclosed)

Before writing this, a raw-regression version (no library-size correction, no shrinkage) was
run once on the scorer's control cells of the three held-out lines. Median r between
-beta and the measured LFC, over targets expressed at CP10k >= 1: HepG2 0.105 (56 targets),
Jurkat 0.117 (67), HCT116 -0.032 (27). Placebo (the covariance column of 20 random expressed
genes) gave 0.050 / 0.067 / -0.010; the target-specific excess was +0.053 / +0.064 / -0.043,
positive for 57 % / 63 % / 48 % of targets. Those numbers motivated this round. They are not
evidence for it and are not reused below.

## Fixed analysis

Implementation: `src/zsp/fluctuation.py`. The model sees only the basal half of the held-out
line's control cells (as every other model does); the scorer's half is never read.

1. **Estimator** (chosen on source lines only). Candidates: (a) raw regression, (b) the same
   after regressing out log library size, (c) (b) on kNN-smoothed cells (k = 10 in a 50-PC
   space, metacell-style). Chosen by the median target-minus-placebo excess on the lines
   *other than* the held-out one, among the single-cell lines HepG2, Jurkat, HCT116, HEK293T.
2. **Primary test (signal).** On each held-out line, for every evaluated target with control
   CP10k >= 1: r(fr, measured LFC) over expressed genes excluding the target itself, minus
   the mean r of 20 placebo genes. One-sided Wilcoxon signed-rank on the excess.
   Supported if p < 0.05 on at least 2 of the 3 held-out lines.
3. **Secondary test (does it add to transfer).** Model `FluctuationBlend`: prediction =
   weighted transfer (or norm-restored) + lambda * fr for targets with CP10k >= 1, unchanged
   otherwise. lambda from {0, 0.25, 0.5, 1, 2}, chosen by the gene-wise centred r gain on the
   single-cell lines other than the held-out one. Supported if the centred gene-wise r
   improves on at least 2 of 3 held-out lines with a paired bootstrap 95 % CI excluding 0,
   and the cell-eval2 mean over the six metrics is not lower than the base model's.
4. If lambda = 0 is chosen, or the primary test fails, the round is reported as negative in
   the README with these numbers, like rounds 5 to 7.

Nothing in this file is edited after the held-out scores exist; corrections go in a dated
addendum below.

## Addendum, 30 Sep 2026 (after the held-out scores; nothing above was changed)

Outcome by the fixed rules (`results/tables/fluctuation_verdict.csv`, estimator chosen on the
other lines: library-size regression for every held-out line):

- Primary test **supported**: target-minus-placebo excess median +0.075 in HepG2 (57 targets,
  67 % positive, one-sided Wilcoxon p = 0.018) and +0.055 in Jurkat (67 targets, 58 %, p = 0.013);
  -0.011 in HCT116 (28 targets, p = 0.80). Two of three lines.
- Secondary test **not supported**: no line has a blend gain whose CI excludes 0 on the positive
  side (HepG2 -0.025 [-0.040, -0.010]; Jurkat +0.007 [-0.001, 0.014]; HCT116 -0.014 [-0.029,
  -0.001], norm-restored base).

A flaw in the pre-registered lambda rule, found when reading the output: it used the median
gain over *all* 200 perturbations, and the blend changes only the 28 to 67 targets the line
expresses, so the median is set by unchanged perturbations and the rule chose lambda = 2 by a
near-tie. Post hoc, restricting the gain to the changed perturbations (mean gain at lambda
0.25 / 0.5 / 1 / 2, norm-restored base): Jurkat +0.008 / +0.013 / +0.015 / +0.002; HepG2
-0.002 / -0.011 / -0.040 / -0.103; HCT116 and HEK293T negative throughout. A corrected rule
chooses lambda = 0 for every held-out line, so the conclusion does not change: the covariance
signal is real but does not add to transfer.
