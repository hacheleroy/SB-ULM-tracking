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
For the four-slot/eight-frame cases, low means at most 20 alive slot-frames,
mid means 21, and high means at least 22. For the three-slot/four-frame exact
case, the corresponding cutoffs are at most 7, exactly 8, and at least 9.
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

## Measured outcome

Exact enumeration took 3.5 process CPU seconds. All 12 exact-case sampler
runs completed within their caps. In each row below, the reference column is
the 95th percentile from 200 independent posterior draws at that particle
count. The sampler median and range come from six independent seeds; `0/6`
means no seed exceeded the reference percentile. These are descriptive
comparisons, not a hypothesis test with an adjusted familywise error rate.

| Particles | Diagnostic | IID 95th percentile | Sampler median | Sampler range | Above IID 95th percentile |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Alive-total TV | 0.06669 | 0.04426 | 0.02553–0.04928 | 0/6 |
| 512 | Maximum clutter error | 0.03307 | 0.01256 | 0.00711–0.03138 | 0/6 |
| 512 | Count-signature TV | 0.16350 | 0.14624 | 0.13174–0.15960 | 0/6 |
| 2,048 | Alive-total TV | 0.03324 | 0.02363 | 0.00812–0.03032 | 0/6 |
| 2,048 | Maximum clutter error | 0.01503 | 0.00795 | 0.00426–0.01207 | 0/6 |
| 2,048 | Count-signature TV | 0.08745 | 0.07606 | 0.06373–0.08073 | 0/6 |

The median maximum event-marginal error fell from 0.04533 to 0.01644 as
particles increased. The exact log evidence is 21.58021; estimated minus
exact log evidence ranged from -0.11778 to 0.21962 at 512 and from -0.10550
to 0.05363 at 2,048. The finite exact case shows no detected systematic
error beyond its IID baseline on the three selected diagnostics. Six runs do
not rule out a smaller bias or failures on different exact cases.

The primary eight-frame case produced the following six-pair medians and
four-seed worst pairs. These are *between-run differences*, not errors against
an exact posterior.

| Label-free difference | 512 median | 2,048 median | 512 worst | 2,048 worst |
| --- | ---: | ---: | ---: | ---: |
| Maximum existence marginal | 0.1660 | 0.1191 | 0.2461 | 0.1597 |
| Maximum missed marginal | 0.1553 | 0.0679 | 0.2109 | 0.1055 |
| Maximum clutter marginal | 0.3086 | 0.3098 | 0.5313 | 0.4072 |
| Maximum birth marginal | 0.1582 | 0.1052 | 0.1914 | 0.1597 |
| Maximum death marginal | 0.1104 | 0.0659 | 0.1719 | 0.1113 |
| Alive-count histogram TV | 0.3145 | 0.1738 | 0.4844 | 0.2686 |
| Assigned-count histogram TV | 0.3008 | 0.1880 | 0.5859 | 0.2813 |

Median pairwise alive-count TV decreased to 55% of its 512-particle value,
but median clutter difference was 100.4% of its earlier value. Mean process
CPU time grew from 48.9 to 185.8 seconds per run (3.80×). The final low-alive
band had masses 0.311–0.795 at 512 and 0.355–0.624 at 2,048; the worst final
joint-band/clutter TV fell from 0.492 to 0.326 but remained substantial. Near
beta 0.5, all primary seeds placed approximately all mass in the low band;
the wide split arose later in the tempering schedule. The maximum single
resampling joint TV was below 0.02 at 512 and below 0.01 at 2,048. Those
observations locate the divergence; they do not prove which transition or
likelihood feature caused it.

The four stress seeds at 512 had median/worst clutter differences of
0.0898/0.1211 and median/worst alive-count TVs of 0.0938/0.1328. Their
final low-band masses were 0.949–0.965, with worst joint-band/clutter TV
0.0176. Thus the primary case's persistent multimode uncertainty is not
universal across the two fresh observations. The stress case's frame six
has only one detection, which limits its clutter diagnostic's range; its
smaller spread should not be called a more demanding calibration result.

**Decision:** `audit_completed_no_candidate_gate`. Increasing particle count
alone reduced several primary-case differences but did not stabilize the
clutter marginal across four seeds. The exact reference is well behaved at
four frames; it cannot certify the eight-frame weights. The next study should
build an independently frozen, *enumerable or otherwise independently
calibrated* bridge that preserves the late competing alive/clutter modes,
then measure absolute mode-weight error and transition behavior there before
selecting another move or opening a holdout. Do not set a pass threshold from
the observed WP11k pairs. No 8,192-particle run or untouched holdout was used.
