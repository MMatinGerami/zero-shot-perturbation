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
