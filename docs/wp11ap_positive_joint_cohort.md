# WP11ap — Prospective positive full-joint crowded cohort

The nonnegative full-joint method was developed only on exposed stopped
cases and committed in `27c4289280e2279a1a6655b5459cf863d492fe1d`.
WP11an remains stopped. This **fresh** protocol and its geometry-only
ledger are committed before any WP11ap posterior, sampler run, or
inspection of candidate mode weights. The WP8 holdout stays closed.

`results/wp11ap_geometry_ledger.json` has SHA-256
`eda79352fdf3fe529dd8cf54a60c297009a8f618d9c2660d647ef0f44016c2a2`.
The 600 unused pairs `(50011+29i, 50017+29i)`, `0<=i<600`, follow the
previous regime and geometry-only eligibility. The two decoys are placed
at frame six and the final frame using the prelisted `.010/.016/.022`
and `.004/.008/.012` mm offsets. Geometry alone yielded 13 eligible
18-detection bases and 15 eligible 19-detection bases, giving 117 and
135 eligible variants. Candidate order is by maximum frame cardinality,
seed index, then round-robin offset pair within each stratum. The first
balanced, numerically sound variant from each distinct base fills one
18-detection and two 19-detection quotas. The per-stratum screen maxima
are 30 and 72. Eligibility, exact order, hashes, and all stopping rules
are encoded in `src/sbulm/wp11ap_positive_joint_cohort.py`.

At every beta-one screen, paired-square remains the source of mode
weights. Its raw minimum must be at least `-1e-10`, clipping TV at most
`1e-9`, and imaginary residual and raw mass error at most `1e-8` each.
Against the nonnegative positive full joint, absolute log-partition
difference must be at most `1e-8`, joint TV at most `1e-7`, and maximum
cell difference at most `1e-8`. Positive float64 versus extended
precision must have log-partition difference at most `1e-10`, joint TV
at most `1e-10`, and maximum cell difference at most `1e-11`; both
positive joints must be nonnegative and normalized within `1e-12`.
Balanced means low/high alive bands at least `.18`, middle at least
`.05`, and two clutter categories at least `.15` at both frames.

Only after all three candidates are selected does validation repeat
those numerical comparisons at beta 0, .5, and 1, plus direct toy
enumeration and analytic-base log partition at beta zero within
`1e-8`. Beta-one validation must reproduce every screened alive and
clutter marginal within `1e-7`. The direct-fourth algorithm is excluded
from the reference gate because it was unstable on the exposed index
84. Screen and validation CPU caps are 900 and 3000 seconds per process.
Generator/ROI errors are geometry ineligibility; a numerical failure,
CPU cap, or validation failure stops the whole cohort without
replacement. Exhaustion gives no cohort. No partial cohort can enter
sampler inference. A complete validated manifest would pin all three
reference and observation hashes before the separately frozen WP11al
24-run rule could be used.

This protocol does not provide a formal floating-point interval bound
or independently validate the shared one-slot likelihood. Those
limits are recorded with the exact numerical evidence rather than
silently counted as calibration.

## Frozen beta-one screening outcome

The nine screens ran in their ledger order: `0, 36, 117, 135, 144,
189, 198, 207, 9`. All nine passed the predeclared numerical gate.
The first balanced 18-detection variant was index `36` (base `52`);
the two balanced 19-detection variants on distinct bases were index
`207` (base `424`) and index `9` (base `35`). The other six screens
failed the predeclared balance criterion and continued. No numerical
failure or CPU cap occurred. The frozen selector returns no next entry
and those three indices as the complete screened cohort.

| Index | Detections | Base | Sixth one-clutter | Final one-clutter | Screen outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 18 | 7 | 0.1321 | 0.2001 | Balance failed |
| 36 | 18 | 52 | 0.1606 | 0.4709 | Selected |
| 117 | 19 | 282 | 0.0667 | 0.1273 | Balance failed |
| 135 | 19 | 290 | 0.0716 | 0.5040 | Balance failed |
| 144 | 19 | 306 | 0.0131 | 0.2363 | Balance failed |
| 189 | 19 | 405 | 0.0530 | 0.2963 | Balance failed |
| 198 | 19 | 417 | 0.1348 | 0.2837 | Balance failed |
| 207 | 19 | 424 | 0.1740 | 0.4273 | Selected |
| 9 | 19 | 35 | 0.2321 | 0.2642 | Selected |

Across all nine records, the maximum paired/positive joint TV is
`4.18e-12` versus the frozen `1e-7` bound; maximum log-partition
difference is `1.44e-12` versus `1e-8`. The maximum positive
float64/extended joint TV is `2.41e-16` versus `1e-10`.
The minimum paired raw cell is `-5.91e-14` versus the permitted
`-1e-10`, and the maximum clipping TV is `1.31e-13` versus `1e-9`.
Total process CPU across screens is `2151.8` seconds. Each individual
screen remained below its `900`-second cap.

At the screening checkpoint these were candidates rather than
validated absolute references. The validation and manifest results
follow below. WP11al sampler inference and the WP8 holdout remain
closed.

## Three-case reference validation

All three screened candidates passed the frozen validation at beta
`0`, `.5`, and `1`. The paired-square and nonnegative full joints
passed every comparison, the positive float64 and extended joints
agreed, toy direct enumeration passed, and the beta-zero partition
matched the analytic base normalizer. The beta-one alive and clutter
marginals reproduced their saved screens exactly in this run. No
validation CPU cap was reached.

| Stratum | Index | Base | Reference SHA-256 | Validation CPU (s) |
| --- | ---: | ---: | --- | ---: |
| 18 | 36 | 52 | `2a4ed5be9e9ce3d41aca73dfe53f9e4044a489ea2394ac784ed4d5d49f6a8f17` | 343.2 |
| 19 | 207 | 424 | `6f1e229ee7a00c5b062b58a0f31b2f3b6b41655b40b00be9f132977b25179833` | 721.4 |
| 19 | 9 | 35 | `5ac3af551e941484a8999a57e73ba5789b73603a2106e69e0e8bf25926adce7a` | 758.7 |

Across all nine candidate/beta checks, maximum paired/positive
full-joint TV was `4.18e-12` (bound `1e-7`) and maximum
log-partition difference was `6.39e-13` (bound `1e-8`). Maximum
positive float64/extended joint TV was `2.60e-16` (bound `1e-10`).
The largest analytic-base log-partition error at beta zero was
`1.42e-14` (bound `1e-8`). The most negative paired raw joint cell
was `-1.07e-14` (permitted minimum `-1e-10`), and maximum clipping
TV was `2.51e-14` (bound `1e-9`).

`results/wp11ap_crowded_cohort_manifest.json` binds the fixed
geometry, all three observation hashes, screen hashes, reference
hashes, and beta-one mode marginals. Its SHA-256 is
`2e2e20acbb84524e5022d9eaa7ab40202420bad660214426dbcf0ecfca000e68`.
The references are numerical absolute targets under the shared
one-slot likelihood implementation; they do not supply a formal
floating-point interval proof or independent likelihood derivation.
The WP11al sampler rule must be bound to this manifest before its
24 jobs run. The WP8 holdout stays closed.
