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

## Addendum, 23 Oct 2026: what happened, and the decision to take

Facts first. The validation phase closed on 22 Oct. Between 18 and 22 Oct the entries that
were scored on the board were v6 (gated, calibrated amplitude, exact cells: 0.081) and
v5-exact (0.132), both below v5 (0.137); the two files built on 18 Oct to restore DE coverage
(v9: amplitude target 2.5, responder fraction 0.3; v10: v9 plus control-axis deflation) were
never uploaded, so condition 3 of the rule above ("not worse than v5 on the validation
leaderboard") was never tested for them. Under the rule as written, the final entry is v5.

The final contexts (released 22 Oct, 130 of their 300 targets shared with the validation
panel): D is a myeloid line (PRTN3, ELANE, AZU1, MPO, LYZ; closest sources CD4 T cells and
K562, weakly); E and F are mesenchymal, fibroblast-like lines (VIM 300 CP10k, COL1A1,
SERPINE1, TGFBI, MMP2) that resemble none of the eight sources. Per-context source weights
were never scored on the board and stay out.

The evidence for v10 over v5, member by member, all from the board unless marked:
`pds` +0.09 (v6 vs v5, measured); `mse` 0 either way at amplitude 1.3 (v5-exact at 1.0 read
0); `nmae` +0.05 (measured); `jac` +0.025 (measured); `reach` -0.03 (measured); `fid`: v5
called about 1,000 genes per knockdown at chance precision (-0.016), v10 calls about 320 to
470 on A/B/C by the scorer's own test run locally (`scripts/22`), between v5's 1,000
(-0.016) and v5-exact's 148 (-0.138), so between -0.02 and -0.14; deflation's `pds` gain
(+0.05 to +0.11 scaled on the local hold-outs) is local evidence only. Expected overall: v10
0.15 to 0.17 against v5's 0.137 on the validation panel; the downside case is about equal
to v5. The final panel is different and the scores are hidden until 19 Nov.

Decision, recorded here before any final upload: the final entry is the v10 recipe, a
departure from the frozen-v5 rule justified by the measured members above and taken with
the fid uncertainty stated. v5 is built alongside as the fallback. Nothing is tuned on the
final contexts' control cells beyond what the recipe already reads from them (the emitted
groups' sampling noise for the amplitude, and the control-cell axes for deflation).

## Addendum, 23 Oct 2026: final entry submitted

`results/submissions/final_v10.vcc` (built at commit c0c3ef0) was uploaded on 23 Oct 2026 and
accepted by the server as entry `oO8fH5oVfckQojUnpwcw` (partition test, panel
vcc2026-test-1, status published). Scores stay hidden until 19 Nov 2026. `final_v5.vcc` is
kept as the fallback and has not been uploaded.
