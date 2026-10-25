# WP11d — Likelihood-guided conditional pair particle Gibbs

WP11c's global conditional-base proposal was reversible but almost never
visited the localization-conditioned posterior. WP11d replaces it with an
**opt-in full-history pair particle-Gibbs update**. Other slots remain fixed;
the selected two slots may jointly change birth/death times, misses, clutter
assignments, and multi-frame associations. The existing local pair move
remains the default. This is still inference in the explicit localization-
level model, not an image-domain diffusion or Schrödinger-bridge result.

For each frame, the pair's unborn/alive/dead transition and finite-set
miss/detection/clutter factor are combined with the exact Gaussian predictive
log density of any assigned localization, conditional on that slot's **entire
earlier assigned history**. The product of these incremental Gaussian factors
equals the original collapsed Matérn marginal likelihood. At temperature
`beta`, the proposal is normalized over all compatible two-slot frame
assignments. Its importance weight is that local normalizer. A conditional
SMC sweep fixes one reference particle to the current complete path, uses
multinomial resampling, and selects one final particle. This is a
particle-Gibbs transition for the *exact* tempered pair conditional; it has
no Metropolis rejection, so actual state and count changes—not acceptance—
measure mixing. The construction follows the conditional-SMC principle in
[Andrieu, Doucet and Holenstein (2010)](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf).

## Validation and frozen pilot

- On a two-slot/two-frame enumerable oracle, summing *every* proposal,
  ancestor and final-selection outcome gives a transition matrix satisfying
  detailed balance at `beta = 0, 0.37, 1`.
- On the 269-state two-slot/three-frame oracle, adding all incremental
  weights exactly reconstructs existence prior + count factors + `beta`
  times the original collapsed localization log density. A three-slot
  integration test checks valid conditional support.
- Before the pilot, seeds 74–76, 512 exact-posterior starts, two sweeps,
  eight conditional path particles, the unchanged WP11c observations, and
  the local-pair comparator were frozen. A mean count-signature transition
  rate of at least 0.01 and all three count histograms within a 99% exact
  multinomial maximum-of-three envelope were required to pass. The latter
  envelope is 0.0877 in count-histogram total variation. The protocol and
  resumable per-run results are in
  [`results/wp11d_conditional_pair_pilot.json`](../results/wp11d_conditional_pair_pilot.json).

| Seed | Conditional PG count change | Local pair count change | PG exact-count TV |
|---|---:|---:|---:|
| 74 | 0.4072 | 0.1270 | 0.0539 |
| 75 | 0.4248 | 0.1465 | 0.0302 |
| 76 | 0.4033 | 0.1572 | 0.0561 |

The mean conditional-PG count change is **0.4118**, and all three
count-TV values lie below 0.0877. Thus the predeclared *small-case*
mixing/stationarity gate passes. Exact-posterior starts provide a sensitive
invariance check, not proof that a full annealed SMC run discovers all modes
from its base distribution. The count histogram is only one label-free
summary; the deterministic detailed-balance test and exact density identity
are the stronger correctness checks.

Cost is a serious boundary. The three conditional-PG pilot arms took
15.6, 17.8 and 19.2 seconds versus 1.3, 1.1 and 0.7 seconds for the local
controls in the same shared environment. These are diagnostic wall times,
not controlled benchmarks. Eight internal path particles require repeated
GP-predictive evaluations. No new eight-frame holdout was opened and no
four-slot posterior calibration is claimed.

The next bounded development test is on the **already exposed** difficult
WP8 observation, with a fixed compute budget and both state/count-change
and runtime diagnostics. Reduce redundant predictive evaluations before
attempting an 8,192-particle comparison. Only after an operational large-
case proposal and exact-oracle retention should an untouched synthetic
holdout be reserved and judged by the inherited seven-metric 0.10 gate.

```bash
python -m sbulm.wp11d_conditional_pair_pilot
pytest tests/test_conditional_pair_pg.py tests/test_wp11d_conditional_pair_pilot.py
```
