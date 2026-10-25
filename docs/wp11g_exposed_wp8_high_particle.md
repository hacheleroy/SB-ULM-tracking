# WP11g — Frozen four-path mixture on the exposed difficult WP8 case

WP11f passed small exact-target and four-slot development screens for a
state-independent 10% conditional particle-Gibbs (PG) update with four
internal paths. This next test froze that mixture **before** rerunning the
previously exposed, difficult WP8 `cluttered_staggered_crossing` observation.
It does not open a new synthetic holdout.

The observation has four slots over eight frames, detection cardinalities
`[1, 4, 4, 4, 3, 5, 1, 1]`, and SHA-256
`afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae`.
The exact WP8 checkpoint, including the original paired-move control runs
at seeds 100 and 200, was verified by file hash and model metadata and
reused without recomputation. The candidate kept the WP6 paired-mixture
probabilities, 8,192 particles, ESS target 0.80, eight rejuvenation sweeps
per stage, and eight final sweeps. Within its 25% paired-move component,
10% of selected proposals used four-path full-history conditional PG; the
remaining 90% used the original local pair move. Other moves were unchanged.
Both candidate seeds were run independently from fresh initial populations.

The frozen first-pair gate required **all seven** inherited label-free
pairwise differences to be at most 0.10. There was a maximum of two
candidate runs, each capped at 1,800 seconds of process CPU time; failure
after the second seed stopped the test before any new observation or seed.
BLAS/OMP thread counts were fixed to one. All controls, provenance, per-run
summaries, and metrics are in
[`results/wp11g_exposed_wp8_high_particle.json`](../results/wp11g_exposed_wp8_high_particle.json).

| Pairwise metric | Archived WP8 control | Four-path candidate | Limit |
|---|---:|---:|---:|
| Existence marginal | 0.1338 | 0.0573 | 0.10 |
| Missed-detection marginal | 0.1373 | 0.0349 | 0.10 |
| Expected clutter | 0.2013 | **0.1335** | 0.10 |
| Birth-time marginal | 0.0544 | 0.0328 | 0.10 |
| Death-time marginal | 0.0551 | 0.0396 | 0.10 |
| Alive-count histogram TV | 0.1511 | **0.1171** | 0.10 |
| Assigned-count histogram TV | 0.0752 | 0.0609 | 0.10 |

All seven discrepancies decrease relative to the archived control, but
**clutter and alive-count TV still fail**. Thus this exposed-case two-seed
gate is negative (`failed_early_before_new_holdout`). The result is
encouraging for movement across modes but cannot be called eight-frame
stability or posterior calibration. It does not reverse the original WP8
failure. No additional seed or previously untouched observation was run.

Both candidate runs used 20 tempering stages. Their process CPU times were
1,454 and 1,467 seconds, below the per-run cap; measured wall times were
1,756 and 3,487 seconds under very different shared-node contention.
Observed mean stage PG-selection fractions were 0.1003 and 0.1001. The
archived WP8 control wall times were about 431 and 433 seconds, but they
were measured in an earlier environment and **not interleaved**, so these
numbers do not yield a controlled speed ratio. At this particle count the
selective PG mixture is nevertheless computationally substantial.

The next bounded development step is a diagnosis of the remaining
clutter/alive-count disagreement on **these same saved observations and
seeds**, with attention to count histograms, count-signature modes, and
ancestry/move diagnostics. Any cheaper or more targeted proposal should
first be checked against the small exact oracle and a separate frozen
development protocol. A new synthetic holdout and in-vivo claims remain
closed until the larger stability and calibration boundary is resolved.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11g_exposed_wp8_high_particle --seed 100
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11g_exposed_wp8_high_particle --seed 200
```
