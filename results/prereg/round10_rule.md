# Pre-registration: round 10 (18 Oct 2026), rendering and amplitude

Written 18 Oct 2026, after reading the vcc2026 metric specification and the public
leaderboard, and after running the fast pseudobulk evaluator (`scripts/18`) on the H1 and
HepG2 hold-outs, but before any cell-level (full scorer) score of the candidates below and
before any leaderboard score of them.

## What was learned, and from where

- The specification (cell-eval2, `docs/vcc2026_metrics`) credits a submission's own sampling
  noise in the expression error, estimated from the scatter of its cells, up to the real
  group's noise and capped in total by the spread of its predicted profiles (`rho`). Cells
  resampled independently (every entry so far) add exactly the noise they are credited for,
  so the credit nets to zero. A group whose pooled profile equals the prediction keeps the
  credit. On the local H1 and HepG2 sets the maximal credit is 45 % and 56 % of the real
  effect energy at 100 cells per knockdown; at the challenge's 400 cells it is smaller, by
  an amount that depends on how weak the real effects are.
- The fast evaluator reproduces the scorer's `pds` and `mse` within a few percent on H1
  (control 1.009 vs 1.014, mean 1.42 vs 1.48, v5 2.02 vs 2.12 raw `mse`).
- No uniform shrinkage of v5 or mean transfer gets below the baseline's `mse` on H1 (best
  about 0.99 against 0.961): the direction error, not the amplitude, is the limit, as the
  response-scale audit said in September.
- A per-entry confidence gate (|mean| over its standard error across sources, ramped to 1 at
  z = 3) raises `pds` on both lines and lowers the error per unit of spread. Deflating the
  prediction along the context's own control-cell axes raises `pds` on H1 only and removes
  the generic response on HepG2, where it carries real energy; it is not adopted.
- With the credit, the error is a decreasing function of amplitude up to the point where
  the spread equals the credit (`rho` = 1) and rises beyond it when the direction is poor.
  The amplitude is therefore set from the prediction and the context's control cells alone:
  spread = credit (`zsp.profile.calibrate_amplitude`, `--rho-target 1.0`).

## The candidates

- **v5-exact**: the v5 model (norm-restored transfer, eight sources, equal weights) with
  cells emitted at exact pooled means (`zsp.emit.emit_cells_exact`). Prediction: `pds` and
  the four DE members unchanged within noise; `mse` above 0 by about the credit's share of
  the effect energy on the validation contexts. Its leaderboard `mse` is the measurement of
  that share, which cannot be computed without the hidden truth.
- **v6 candidate**: `GatedTransfer` (norm-restored, equal weights, z0 = 3), generic response
  scaled by 0.7, amplitude set to `rho` = 1, exact emission. On the fast evaluator it beats
  v5 on both lines on `pds` (H1 scaled 0.83 vs 0.79; HepG2 1.12 vs 1.00) and on `mse` under
  full credit (H1 0.18 vs 0.00; HepG2 0.46 vs 0.55 is the one loss, from the smaller spread).
  The DE members are unknown before the full scorer runs. Expected risk: the gate shrinks
  weakly supported genes, which should raise `jac` and `fid` and may lower `nmae` where
  the reference's significant genes are among the shrunk.

## The rule (the one already in `final_test_protocol.md`)

A candidate replaces v5 as the final model only if, on the H1 hold-out scored by cell-eval2
(paired over the 200 knockdowns, `scripts/15`), it beats v5 on at least four of six metrics
with no metric's confidence interval entirely below 0, and it is not worse than v5 on the
validation leaderboard. v5-exact is tested first because it isolates the rendering step; the
v6 candidate is tested against v5-exact as well as v5, so that the model change and the
rendering change are not confounded.

Hyper-parameters (z0 = 3, generic scale 0.7, rho target 1.0) were chosen on the fast
evaluator over the H1 and HepG2 hold-outs, so their `pds` and `mse` on H1 are not an
independent test; the DE members on H1 and every member on the leaderboard are.

Deadline context: the final test contexts are released on 22 Oct; after that date the
protocol allows no model choice, so any adoption happens on or before 21 Oct.

## Addendum, 18 Oct 2026 (after the H1 cell-level scores; nothing above was changed)

H1 hold-out, official scorer, 200 knockdowns at 100 cells each; paired differences against
v5 (`results/tables/round10_h1_paired.csv`), DE-call decomposition (`scripts/21`):

| run | pds raw | mse raw | fid raw | jac raw | genes called (median) | precision of calls | coverage |
|---|---|---|---|---|---|---|---|
| v5 (independent cells) | 0.809 | 2.13 | 0.489 | 0.050 | 1,027 | 0.53 | 1.00 |
| v5-exact | 0.839 | 1.57 | 0.445 | 0.047 | 457 | 0.55 | 1.00 |
| v6 candidate | 0.859 | 0.82 | 0.309 | 0.028 | 58 | 0.61 | 0.50 |
| v6 + floor 0.1 | 0.874 | 0.83 | 0.333 | 0.030 | 78 | 0.61 | 0.72 |
| v6 + floor 0.2 | 0.880 | 0.84 | 0.350 | 0.028 | 117 | 0.58 | 0.98 |

The reference calls a median of 70 genes per knockdown here. Two facts decide the DE members:

1. **Direction precision is close to chance whatever is called**: 0.53 over 1,027 calls, 0.61
   over the 58 most confident. The sources' consensus sign is right for barely more than half
   of the genes, confident or not. This is the direction problem of the September audit seen
   from the DE side, and no gate fixes it.
2. **Fidelity on this set is coverage**: v5 calls 15 times more genes than the reference and
   scores 0.49 through full coverage at chance precision; the candidates call fewer than the
   reference and lose through coverage. On the leaderboard the baseline already sits at
   chance with full coverage (0.505 to 0.522), so what counts there is precision at full
   coverage. v5's leaderboard fid of -0.016 is that chance precision. A set calling about as
   many genes as the reference at 0.58 to 0.61 precision would score +0.2 to +0.3 on that
   member, but the reference's count at 400 cells is unknown, and calling fewer than it is
   charged in proportion. The local set (100 cells) cannot calibrate the count.

Decision under the rule: neither candidate replaces v5 on the H1 evidence (fid and jac
confidence intervals below 0 for both). v5-exact and the v6 candidate are submitted to the
validation leaderboard today as measurements, not as adoptions: v5-exact to measure the
sampling credit's share on the challenge contexts (its mse) and whether its fid holds with
400 cells; the v6 candidate to measure the gate's effect on precision and coverage at 400
cells. The v6 candidate is submitted first so that the public rank shows v5-exact.

## Addendum, 18 Oct 2026, evening: the two leaderboard readings and the plan for 19 to 21 Oct

| entry | overall | rank | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|
| v5 (2 Oct, independent cells) | 0.137 | 433 then | 0.535 | 0.000 | 0.101 | -0.016 | 0.207 | -0.005 |
| v5-exact | 0.132 | 618 | 0.602 | 0.000 | 0.110 | -0.138 | 0.205 | 0.012 |
| v6 gated, rho 1, generic 0.7 | 0.081 | 804 | 0.625 | 0.175 | 0.154 | -0.667 | 0.177 | 0.020 |

What the board adds to the H1 evidence:

1. `mse` leaves 0 only at the calibrated amplitude (0.81 to 0.84); at amplitude 1 the credit
   does not cover the bias on the validation contexts either.
2. `fid` follows the number of genes called. Measured on context A with the scorer's own test
   against the context's controls (`scripts/22`): about 1,000 calls per knockdown (v5,
   independent cells) gave -0.016; 148 calls (v5-exact) gave -0.138; 17 calls (v6) gave
   -0.667. The reference therefore calls on the order of a few hundred genes at 400 cells,
   and anything under that is charged in proportion at a precision near 0.5.
3. The gate's other gains held on the board: `pds` +0.09, `nmae` +0.05, `jac` +0.025.

The pooled profile wants the calibrated amplitude; the DE test wants several hundred calls.
Two knobs that leave the pooled profile untouched or nearly so: a responder fraction in the
emitter (a fraction of the cells carries the response, scaled so the pooled mean is exact;
uneven knockdown is a known feature of CRISPRi) and a higher `rho` target (amplitude above
the calibrated one, which costs `mse`). Calls per knockdown on A/B/C: rho 2.5 with a 0.3
responder fraction 327/419/469; rho 2.0 with 0.3: 226/272/321.

Entries for 19 Oct (built tonight, `results/submissions/val_v9_*.vcc`): the gated model at
rho 2.5 and responder fraction 0.3, and the same with per-context source weights (A: CD4 x3,
Jurkat x2; B: HCT116 x3, HEK293T x2, RPE1 x2; C: HCT116 x3, HEK293T x2; H1 x0.3 everywhere).
Expectation written before the scores: `fid` back near -0.02, `mse` near 0, `pds` and `jac`
as in v6, overall about 0.16; the weighted variant tests whether identity weighting helps
`pds` on A. If the first holds, 20 Oct tries rho 2.0 with the same fraction to buy back `mse`.
