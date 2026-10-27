# WP11aj — Prospective crowded absolute reference cohort

WP11ai passed its four paired runs on one independently selected balanced
17-detection development case. This study targets **three new, distinct**
crowded observations: one with 18 detections and two with 19 detections.
The rule below and its geometry-only ledger are frozen before any WP11aj
posterior. No sampler result is used to select a case.

## Geometry and ordered selection

Generate 240 unused trajectory/corruption seed pairs
`(3611+13i, 3617+13i)` for `i=0..239`, four staggered slots,
detection probability 0.70, clutter rate 0.28, separation 0.007 mm,
and at most two injected clutter points per frame. Eligible bases have
18 or 19 detections; frame six contains 2–3 true detections and no
clutter, the final frame exactly two true and no clutter; frames three
through five total at least six points, frame seven at least one, and
no individual frame has more than four points. Generator errors are
ineligible and recorded.

In each of frame six and the final frame, move the y coordinate of the
largest nonnegative reference ID toward zero, then label it as clutter
for geometry bookkeeping. Frozen sixth-frame offsets are 0.010, 0.016,
0.022 mm; final-frame offsets are 0.004, 0.008, 0.012 mm. The inference
uses positions, not these bookkeeping labels. An out-of-ROI variant is
ineligible. Work first through the 18-detection stratum, then the
19-detection stratum. Within each, order eligible bases by smallest
maximum frame cardinality, then seed index; visit bases round-robin
for each sixth-then-final offset pair. Accept only the first eligible
variant per base, so that selected cases are distinct. Screen at most
12 variants for 18 detections and 27 for 19 detections; target one and
two references, respectively. No skipped or replacement entry is
allowed, and an incomplete quota yields no cohort.

The geometry-only ledger is `results/wp11aj_geometry_ledger.json`
(SHA-256 `cf3f078aa9bd07db20b1bc3ba02a71e5b331c7ce5c1ede473f94973c4f3ab675`).
It contains 10 eligible bases: six with 18 detections and four with
19. The first ordered base indices are 173 and 48, respectively.
This ledger and the executable rule must be committed before the first
exact screen.

## Exact balance and validation

At beta one, evaluate the exact four-slot squarefree joint of alive
total × frame-six clutter × final clutter. Alive low is <=20, middle
is 21, high is >=22. Accept a numerically sound screen only if low and
high masses are both >=0.18, middle >=0.05, and at least two clutter
categories have mass >=0.15 **in each** of frame six and the final
frame. An unbalanced, numerically sound screen advances in the frozen
order. A numerical failure or CPU cap stops the entire cohort, even
if some cases were previously accepted.

Every screen must have raw minimum cell >=-1e-10, relative imaginary
residual <=1e-8 and raw mass error <=1e-8, using one BLAS/OMP thread
and at most 600 process CPU seconds. For each of the three selected
cases, validate the toy by direct enumeration and compare direct-fourth
and paired-square parenthesizations at beta 0, 0.5 and 1. Require
absolute log-partition difference <=1e-8, joint TV <=1e-7 and maximum
cell difference <=1e-8, with the same raw-cell, imaginary and mass
limits for both calculations. The beta-zero log partition must match
the analytic base within 1e-8 and the beta-one alive bands must
reproduce their screen within 1e-7. Each validation has a 2,400
process-CPU-second cap. These independent numerical checks are
empirical cross-validation, not formal interval error bounds.

The final manifest will pin every detection, screen and reference hash.
Only after all three references validate will a sampler comparison
be frozen, using these exact joints. The previous WP11ah 17-detection
result remains an independent context case. The untouched WP8 holdout
stays closed.

## Outcome

The **first** ordered screen, 18-detection variant 63 from base 173,
returned a minimum raw joint cell of `-1.264757855e-10`, below the
frozen `-1e-10` limit. Its clipping TV was `3.923e-10` and process CPU
69.2 seconds. The posterior masses are recorded only as a diagnostic
in `results/wp11aj_screens/63.json` (SHA-256
`43bf298774b4eb044e708940089a62a2734205ee9600b3069924e96370bc113b`).
The exact numerical **stop** applies regardless of those masses. No
later variant was screened, no selected reference was validated, and
the WP11aj cohort failed. The WP8 holdout and all WP11aj sampler
inference remain closed. A future study must freeze an independent
numerical method and fresh observations before another posterior.

A diagnostic **after** this stop recomputed variant 63 by both subset
parenthesizations. The paired-square minimum raw cell was `-1.745e-13`
and clipping TV `1.05e-12`; the direct-fourth calculation reproduced
its failing `-1.265e-10` minimum. Their normalized joint TV was
`4.79e-9`, maximum cell difference `6.99e-10`, and log-partition
difference `8.36e-10`. The diagnostic is
`results/wp11aj_numeric_diagnostic.json` (SHA-256
`64480db0ce6a79429817b97f90d7c41066ef246373adebfb9c60b12641d41a18`).
This is evidence about numerical cancellation, **not** a reopened
WP11aj screen or validated reference.
