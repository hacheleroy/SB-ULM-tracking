# WP11ak — Fresh paired-primary crowded absolute cohort

WP11aj stopped at its **first** frozen screen because direct-fourth
squarefree convolution produced a raw joint cell below its `-1e-10`
limit. A diagnostic on that stopped observation found the paired-square
path's minimum at `-1.745e-13`; the two normalized joints agreed to
TV `4.79e-9`. WP11aj remains failed. This new protocol uses **unused
observations** and a different, independently frozen numerical budget;
it cannot retrospectively accept the WP11aj screen.

## Geometry and cohort rule before any WP11ak posterior

Use 240 fresh trajectory/corruption seed pairs `(8101+17i, 8111+17i)`
for `i=0..239` with four staggered slots, detection probability 0.70,
clutter rate 0.28, separation 0.007 mm and at most two injected clutter
points per frame. The geometry eligibility and two-frame y decoys are
as in WP11aj: 18 or 19 total detections, frame six 2–3 true and zero
clutter, final exactly two true and zero clutter, frames three through
five at least six detections, frame seven at least one, and at most four
per frame. Move the largest true ID's y coordinate toward zero in each
of frames six and eight by the frozen offset ladder 0.010/0.016/0.022
and 0.004/0.008/0.012 mm, respectively; relabel the two detections as
clutter for bookkeeping, never for posterior inference. Generator and
out-of-ROI errors are ineligible.

Search the 18-detection stratum first, then 19. Within each, order
bases by minimum maximum frame count and seed index, visiting bases
round-robin for every sixth-then-final offset pair. Only the first
passing variant of each base may be selected. Target **one distinct
18-detection and two distinct 19-detection cases**, with at most 18 and
35 numerically sound-or-unbalanced screens, respectively. The exact
balance condition remains low/high alive masses >=0.18, middle >=0.05,
and at least two clutter categories >=0.15 at both frame six and the
last frame. Low/middle/high mean alive total <=20, =21 and >=22.
Any numerical failure, CPU cap, or exhaustion stops the **whole**
cohort; do not substitute an entry or conduct sampler inference on a
partial cohort.

The geometry-only ledger is `results/wp11ak_geometry_ledger.json`
(SHA-256 `a34fbfcea6bb1230aed432f7da2a8ece8d1c94a346d7d615a49cb42117df8b47`).
It contains 12 eligible 18-detection and seven eligible 19-detection
bases; the first ordered base indices are 21 and 124. This code,
ledger, and numerical budget are committed before the first WP11ak
posterior.

## Independent numerical budget and validation

Use paired-square squarefree convolution as the **primary** absolute
joint. On **every** beta-one screen also compute direct-fourth from the
same one-slot coefficients. Paired minimum raw cell must be >=-1e-10
and clipping TV <=1e-9; direct minimum >=-1e-8 and clipping TV <=1e-7.
Both methods must have relative imaginary residual <=1e-8 and raw
mass error <=1e-8. Their normalized joint TV must be <=1e-7, maximum
cell difference <=1e-8, and absolute log-partition difference <=1e-8.
This two-calculation budget is set using the prior stopped observation
only as a *numerical method diagnostic*; the new seed ladder is unseen.
Use one BLAS/OMP thread and a 1,200 process CPU-second cap per screen.

For each selected case, validate the toy by direct enumeration and
repeat the paired/direct comparison at beta 0, 0.5 and 1 with the
same numerical budget. The paired beta-zero log partition must also
match the analytic base within 1e-8, and beta-one alive bands must
match the screen to 1e-7. The full validation has a 3,000 process
CPU-second cap per reference. These checks are empirical, not formal
interval bounds. The manifest will bind every detection, screen,
reference and ledger SHA-256 before any sampler comparison is frozen.
The WP8 holdout stays closed.

## Outcome

The frozen ladder screened two 18-detection variants and four
19-detection variants, in order. Variant 45 from base 67 met the exact
18-detection balance condition, but it remains **provisional** because
the full cohort never completed. Variants 81, 153 and 36 in the
19-detection stratum were numerically sound but failed balance.
Variant 81 missed sixth-frame clutter-one mass by only `0.000176565`:
`0.149823435` against the frozen `0.15` threshold.

The next ordered 19-detection variant 63 from base 116 met the
balance masses under the paired-primary calculation (alive bands
`0.39666 / 0.15685 / 0.44649`, frame-six clutter zero/one
`0.81239 / 0.18100`, final clutter zero/one `0.33628 / 0.62753`).
It **failed the numerical cross-check**: direct versus paired
log-partition difference `4.5414e-8` exceeded the frozen `1e-8` cap.
The normalized joint TV was `7.659e-8` (cap `1e-7`) and maximum cell
difference `9.606e-9` (cap `1e-8`), but passing those two checks does
not excuse the log-partition failure. Paired raw minimum was
`-6.947e-14`; direct raw minimum was `-4.186e-9`.

The screen records are in `results/wp11ak_screens/`. The first
numerical failure **stops the entire WP11ak cohort**. No later
variant, three-beta reference validation, sampler comparison, or WP8
holdout inference was run. This case cannot be selected by changing
the log-partition threshold after seeing it. The next absolute method
needs an independent, more stable partition cross-check and a fresh
prospective observation set.
