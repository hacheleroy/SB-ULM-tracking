# WP11x — Frozen frame-six decoy reference search

WP11w's three exact screens retained balanced low/high alive mass, but
each concentrated frame-six clutter in one count. This new **synthetic**
observation rule deliberately makes one frame-six localization a plausible
clutter decoy. The rule, seed ladder, eligibility, offset order, exact
acceptance and stop conditions are fixed here before evaluating any WP11x
posterior. No sampler move or untouched holdout run belongs to WP11x.

## Observation rule and ordered ladder

Use the WP8 four-slot, eight-frame staggered crossing with detection
probability 0.68, expected clutter 0.28 per frame, central separation
0.007 and a maximum of two baseline clutter points per frame. Use the
following trajectory/corruption seeds in order:

`(271,277), (281,283), (293,307), (311,313), (317,331), (337,347),
(349,353), (359,367), (373,379), (383,389), (397,401), (409,419),
(421,431), (433,439), (443,449), (457,461), (463,467), (479,487),
(491,499), (503,509), (521,523), (541,547), (557,563), (569,571)`.

Base eligibility uses only simulated labels and detection geometry:
17–19 total detections; frame six has two or three retained true
detections and no baseline clutter; frame eight has exactly two
detections; and frames three through five contain at least seven.
Record generator failures as ineligible and do not replace seeds.

For each eligible base, find the frame-six detection with the largest
nonnegative reference ID. Replace its **y** coordinate with
`y - sign(y) × δ`, where `sign(0)=+1`, for offsets δ=0.010, 0.016,
0.022 in that order, measured in the generator's millimetres. Keep x,
the cardinalities and all other frames unchanged. Mark the replaced
detection as clutter (`reference_id=-1`) so the original source is a
miss. This rule sees source labels and coordinates from the synthetic
generator but no posterior or sampler output. An offset outside the
fixed ROI is a geometry failure, not a reason to tune it.

Write the complete geometry ledger, including each eligible variant's
hash and cardinalities, and pin its SHA-256 **before any posterior
calculation**. Screen variants in seed order, then offset order, stopping
at the first accepted exact target. Evaluate at most the first nine
eligible variants; if fewer exist, evaluate all. A generator or geometry
failure is recorded and skipped. If none passes, stop without revising
the observation rule or numerical thresholds.

## Exact acceptance and validation

For beta one, compute the exact four-slot joint of alive total ×
frame-six clutter × final clutter. Accept only when low (alive ≤20)
and high (alive ≥22) bands each have mass ≥0.18, middle (alive 21)
has mass ≥0.05, at least two frame-six clutter counts each have mass
≥0.15, and at least two final clutter counts each have mass ≥0.15.
These criteria preserve the WP11w difficulty definition and use no
sampler results. Report every attempted candidate and its failed
criteria.

At every exact beta, require raw minimum cell ≥−1e-10, maximum
relative imaginary residual ≤1e-8 and raw mass error ≤1e-8. Stop on
numerical failure or a 450 CPU-second per-screen cap, one BLAS thread
and one process per candidate. A computational stop cannot select the
next variant. For a passing case, run separate toy exhaustive checks
at beta 0, 0.5 and 1 (log partition ≤1e-9, joint TV ≤1e-9, max
cell error ≤1e-10), then direct-fourth versus paired-square subset
checks on the selected target at all three betas (log partition
difference ≤1e-8, joint TV ≤1e-7, max cell difference ≤1e-8).
Require beta-zero log partition within 1e-8 of the model's base
normalizer and recomputed beta-one alive bands within 1e-7 of the
screen. Cap selected validation at 1,200 CPU seconds. These checks
are empirical numerical validation, not interval bounds. Pin the
validated reference before designing any later sampler test.

## Outcome

Pending geometry ledger and exact screens.
