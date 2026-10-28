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
