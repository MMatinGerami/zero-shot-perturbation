# zero-shot-perturbation

[![CI](https://github.com/MMatinGerami/zero-shot-perturbation/actions/workflows/ci.yml/badge.svg)](https://github.com/MMatinGerami/zero-shot-perturbation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- cut a v0.1.0 release (none exists yet) and add the release and Zenodo badges -->

Predicting the transcriptional response to CRISPRi knockdowns in cell lines that have never been
perturbed, using only their unperturbed cells. My entry to the
[2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026)
(Arc Institute), with the code to reproduce it on public data.

**In one line:** a transfer baseline (average the same knockdown's measured response in other
cell lines, then restore the magnitude that averaging removes) reached rank 433 and overall score
0.137 on the validation leaderboard (October 2026); a noise ceiling and a transfer ceiling
measured from the data show that transfer models are already near what copying another line can
give (HepG2: 0.35 of 0.40), so the remaining gain has to be line-specific.

Since deep models do not yet beat simple linear baselines on this task
([Ahlmann-Eltze et al., 2025](https://doi.org/10.1038/s41592-025-02772-6)), I started from
transfer baselines rather than a foundation model.

![ceilings](results/figures/ceilings.png)

*The best model against two ceilings, per held-out line. Light grey: the noise ceiling, the
highest gene-wise correlation any prediction could reach against a response measured from 50 to
100 cells. Dark grey: the transfer ceiling, what a perfect copy of another line's response would
reach. Bar: norm-restored transfer. Medians over 200 perturbations.*

## Results at a glance

| Question | Answer |
|---|---|
| How far does the same knockdown transfer between lines? | After noise correction, about 0.47 correlation between HepG2 and Jurkat |
| How close is the model to that limit? | HepG2 0.35 against a transfer ceiling of 0.40; Jurkat reaches 0.24 against a centred noise ceiling of 0.72 (34%) |
| What helped on the leaderboard? | Restoring the magnitude after averaging sources: rank 559 to 524 (6 sources) to 433 (8 sources, equal weights), overall 0.079 to 0.137 |
| Does weighting sources by basal similarity help? | Not on the challenge-like line: equal weights beat similarity weights (ablation on the leaderboard, 0.137 against 0.107) |
| Can the model say which predictions to trust? | Not yet: none of five uncertainty signals changes any metric by more than 0.06 when keeping the 50% most-certain targets; kept as a negative result |
| Does transfer survive a change of study? | Partly: on HCT116 (X-Atlas, shallow knockdowns) direction transfers (fid 0.49) but gene-wise correlation falls to 0.09 |

Final test entry submitted 23 October 2026; scores are hidden until the organisers announce
results. Per-round results, the local benchmark and all tables are in
[docs/results.md](docs/results.md); every leaderboard entry is in
`results/tables/leaderboard.csv`.

## Quickstart

```bash
uv sync
make data        # about 16 GB from figshare and GEO
make contexts    # pseudobulk per screen
make benchmark   # leave-one-line-out, scored with cell-eval2
make figures
make test
```

## When to use this

- You enter the Virtual Cell Challenge and want a reproducible transfer baseline, a local
  leave-one-line-out benchmark scored with `cell-eval2`, and a fast pseudobulk evaluator that
  matches the scorer's `pds` and `mse` within a few percent.
- You want the noise-ceiling and transfer-ceiling code (`src/zsp/ceiling.py`, tested on
  simulated counts) to know how much of your error is fixable.
- You want pseudobulk contexts of six public CRISPRi screens (K562, RPE1, HepG2, Jurkat,
  HCT116, HEK293T) built from the raw releases, with "not measured" kept as NaN.

Not a trained deep model and not a winning entry: it is the baseline the deep models have to
beat, with the measurement of why.

## Data

Public CRISPRi Perturb-seq screens, reduced to one pseudobulk profile per knockdown:

- K562 and RPE1, [Replogle et al., 2022](https://doi.org/10.1016/j.cell.2022.05.013)
- HepG2 and Jurkat, [Nadig et al., 2025](https://doi.org/10.1038/s41588-025-02169-3)
- HCT116 and HEK293T (X-Atlas/Orion), [Huang et al., 2025](https://doi.org/10.1101/2025.06.11.659105)

## Limitations

- Few source lines: a target measured only in K562 is effectively a K562 prediction.
- The direction of the response is inherited from the sources, not adapted to the new line.
- Knockdown depth differs between screens, so sources are not equally informative.
- Local scores are raw `cell-eval2` metrics on public lines, not leaderboard scores.
- No validated uncertainty yet.

## Citation

```bibtex
@software{gerami2026zsp,
  author = {Gerami, Matin},
  title  = {Zero-shot perturbation-response prediction across cell contexts},
  year   = {2026},
  url    = {https://github.com/MMatinGerami/zero-shot-perturbation}
}
```

## License

MIT
