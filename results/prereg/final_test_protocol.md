# Pre-registration: final test phase (hidden contexts D to F, released 22 Oct 2026)

Written 1 Oct 2026, three weeks before the test contexts exist.

## The prediction that counts

The final entry is the v5 model as committed today (`results/prereg/v5_submission_rule.md`):
norm-restored transfer over the eight public screens (K562, RPE1, HepG2, Jurkat, HCT116,
HEK293T, H1, CD4 Stim48hr), equal source weights, built by `scripts/05_make_submission.py`
with `--model norm_restored --extra-sources h1 cd4 --loso-table
loso_grid_h1_cd4_challenge.csv`, applied to the test contexts' control cells and target
list. Code and data are those of the commit that adds this file, unless a change is made
under the rule below before 22 Oct.

## What may change before 22 Oct, and how

A model change may replace v5 only if all three hold, and each is committed before 22 Oct:
1. it is written down as a rule first (as v5 was);
2. it beats v5 on the H1 hold-out (paired, scripts/15) on at least four of six metrics with
   no metric's CI entirely below 0;
3. it is not worse than v5 on the validation leaderboard.

## What may not happen after 22 Oct

- No hyper-parameter, source or model choice is made after the test contexts are released.
  The only test-phase work is building and validating the file (`vcc prep`) and submitting.
- Nothing is tuned on the test contexts' control cells beyond what the frozen model already
  does with them (basal profile for the norm restoration's source weighting, which is equal
  weights and so ignores it, and control cells for emitting predicted cells).

## What is reported afterwards

The final score and rank, whatever they are, go in the README next to the validation
entries, with this file and the commit hash of the submitted model linked. A failure is
reported the same way as a success.
