# Pre-registration: leaderboard entry v5 (round 9 sources)

Written 1 Oct 2026, before any scoring with CD4 as a source had been read. Already seen at
that point, and therefore not evidence for the rule below: the H1 hold-out scores of the
four 6-source models (control, mean, weighted, norm-restored), the source weights for the
challenge contexts, and a partial 8-source LOSO run that favoured flatter weights.

## What is fixed now

1. **Aggregation:** norm-restored transfer (the v4 model). Reason: on every held-out line
   so far, including H1, it has the best fid and reach and the worst mse, and the
   leaderboard's mse term has been 0.00 for all four entries, so it costs nothing there.
2. **Sources:** the six core screens, plus H1 and CD4 (Stim48hr), for the challenge
   contexts A to C.
3. **Temperature:** chosen by the existing pre-registered rule (`src/zsp/loso.py`, maximise
   the mean oriented proxy) run over all eight contexts as pseudo-targets, from the grid
   {0.05, 0.1, 0.2, 1e6} (1e6 = plain average). Not chosen by hand and not on H1 cell-level
   scores.

## When v5 is submitted

v5 is submitted if **both** hold:

a. On the H1 hold-out (`results/local_eval/h1_*`), adding CD4 as a source does not make
   norm-restored transfer worse: the paired mean change (scripts/15) is >= 0 on at least
   four of the six metrics, or no metric has a 95 % CI entirely below 0.
b. The submission file passes `vcc prep` with all official checks.

If (a) fails, v5 uses the six core sources plus H1 only, with the LOSO temperature as above.
Either way the outcome, and the leaderboard score when it comes, are reported in the README
with this file linked. Nothing here is edited afterwards; corrections go in a dated addendum.

## Addendum, 1 Oct 2026 (after the CD4 scores; nothing above was changed)

- Condition (a): adding CD4 to norm-restored transfer on the H1 hold-out, paired over 200
  knockdowns (`results/tables/round9_h1_cd4_paired.csv`): pds -0.009 [-0.034, 0.016],
  mse +0.0006 [0.0003, 0.0008], nmae +0.011 [0.001, 0.021], fid +0.0002 [-0.007, 0.007],
  reach +0.016 [-0.011, 0.042], jac +0.002 [0.000, 0.004] (positive = better). Five of six
  >= 0 and no CI entirely below 0: **passes**.
- Temperature by the LOSO rule over all eight contexts
  (`results/tables/loso_grid_h1_cd4_challenge.csv`): mean oriented proxy 0.168 (0.05),
  0.171 (0.1), 0.173 (0.2), **0.177 (1e6)**. Equal weights are chosen; v4 used 0.1.
- v5 is therefore norm-restored transfer over all eight sources with equal weights.
  Condition (b), `vcc prep`, is checked on the built file.

## Addendum, 2 Oct 2026: leaderboard result

Submitted as `norm-restored-8src-v5`. Validation leaderboard: rank 433 (v4: 524), overall
0.1369 (v4: 0.0987); pds 0.535, mse 0, nmae 0.101, fid -0.016, reach 0.207, jac -0.005, each
better than v4. Recorded in `results/tables/leaderboard.csv`.
