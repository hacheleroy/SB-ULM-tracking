# WP11k — Variance and calibration audit

WP11i and WP11j moved the small exact case yet each missed a different
two-seed eight-frame development guardrail. This audit uses the unchanged
WP11g sampler, with 10% four-path PG inside the paired mixture. It measures
absolute error where exact enumeration is feasible and independent-run spread
on fresh eight-frame observations. The implementation and this protocol are
committed before any posterior inference on these observations.

## Frozen design

The exact reference has three slots and four frames at times 0, 1, 2, and
3 ms. Detection arrays are `[[-0.009, 0.001]]`, empty,
`[[0.008, 0.0], [-0.007, 0.002]]`, and `[[-0.010, 0.001]]` in the existing
coordinate units. Localization standard deviation is 0.005; Matérn coordinate
scales are `(0.02, 0.01)` and temporal length scale is 0.003. Detection
probability is 0.65, expected clutter is 0.25 per frame, and the ROI is
`[-0.04, 0.04]` in both axes. Use the existing default birth/death prior.
The enumeration has **36,590 states**; the detections SHA-256 from the
repository's case metadata is
`879487937d467118e113a552e88412b828967dbfc2e3a42868d2658f861f9de9`.

The independent eight-frame cases use the existing staggered truth mask,
the existing trajectory/corruption generator, and these fixed inputs:

| Case | Detection probability | Clutter/frame | Separation | Trajectory seed | Corruption seed | Max clutter/frame | Detection SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Primary | 0.68 | 0.28 | 0.007 | 79 | 83 | 2 | `e7fc37d377e5b956377af6a64e41837f15e42d43e61a932af48400da9bffd922` |
| Stress | 0.62 | 0.50 | 0.004 | 89 | 97 | 3 | `7f6d4ebab4984dd5243684587b1a9da7e6babe71f4e2f79494395bf2b113b495` |

The primary frame cardinalities are `[2,2,4,4,3,2,1,1]` and the stress
cardinalities are `[1,1,2,2,3,1,2,0]`. Any mismatch aborts before sampling.

1. Enumerate and normalize the exact reference; draw 200 independent
   multinomial replicates at each particle count using baseline seed 18113.
   Report median and 95th percentile of alive-total histogram TV, maximum
   expected-clutter error, and joint per-frame alive/assigned count-signature
   TV. This is a descriptive finite-sample reference, not a tuned pass limit.
2. Exact case: seeds 820–825, first at 512, then at 2,048 particles. Report
   the six absolute errors, their median/range, the number above the
   corresponding independent-sample 95th percentile, maximum event-marginal
   error, and log-evidence error. Each run has a 250 or 450 process-CPU-second
   cap respectively; reference enumeration has a 600-second cap.
3. Primary eight-frame case: seeds 800–803, first 512, then 2,048 particles.
   Report all six pairwise differences per size, their median/range and
   four-seed maximum for the seven inherited label-free metrics. Compare the
   two sizes by median pairwise ratio. Caps are 300 and 750 CPU seconds.
4. Stress eight-frame case: seeds 810–813 at 512 particles only, with a
   300-CPU-second cap. Report the same spread as context, without extrapolating
   a particle-count trend from this single size.

Every sampler run uses a separate process with BLAS/OMP thread variables set
to one. A passive, RNG-free observer records post-stage joint low/mid/high
alive-total bands by clutter count at frame six (or the last frame for the
exact case), plus resampling joint TV. Compare final mode masses across seeds.
The exact-count signature ignores slot labels and preserves both per-frame
alive and assigned counts. Raw association samples need not be published.

This audit has **no candidate-selection threshold**. A cross-seed difference
alone does not establish absolute error on eight frames; the exact four-frame
case cannot establish calibration at eight frames. We will distinguish
repeat-to-repeat spread from exact-case systematic error in the report and
recommend the next experiment based on their measured sizes. No WP8 exposed
case, 8,192-particle run, or untouched holdout is part of this protocol.

Implementation: `src/sbulm/wp11k_variance_calibration.py`. Checkpoint and
per-run summaries: `results/wp11k_variance_calibration.json` after completion.
