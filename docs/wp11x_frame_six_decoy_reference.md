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
detections and no baseline clutter; frame eight has at least one
detection; and frames three through five contain at least six.
Record generator failures as ineligible and do not replace seeds.

The first, stricter geometry-only rule required exactly two last-frame
detections and at least seven across frames three through five. It
produced zero eligible bases in the fixed ladder. Before evaluating any
posterior, we preserved that failed feasibility ledger and broadened
those two **geometry** conditions as stated above. The seed list,
decoy transformation, offsets, exact balance criteria and numerical
limits did not change. This revision is committed before screening.

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
screen. Cap selected validation at 1,800 CPU seconds. These checks
are empirical numerical validation, not interval bounds. Pin the
validated reference before designing any later sampler test.

## Outcome

The first geometry-only ledger, with zero eligible bases, is retained
at SHA-256 `09a023d6c00e6578c65eff3e1a9437756da45855d287f7a116576b31c56b0190`.
The revised ledger is pinned at SHA-256
`d1eabae457ac7b4145828d4ee59566ca0811403ac616c4877b0b5f6858e35679`.
It has three eligible bases (seed pairs `(421,431)`, `(433,439)`,
`(491,499)`) and nine ordered decoy variants. One other base had a
generator ROI error; it was not replaced. No WP11x posterior has been
evaluated at this point.

The first two variants were screened in order; the second was the first
to pass all frozen criteria. The selected observation is pair
`(421,431)` with δ=`0.016` mm, cardinalities
`[1,2,2,2,2,3,3,2]`, and detection SHA-256
`dd3d3fbb55fd601298ee410539767df24950e2f886d814c30b98f54e89bc0c29`.
Later variants were not screened.

| Variant | Alive low / middle / high | Frame-six clutter 0 / 1 | Final clutter 0 / 1 | Result |
| --- | --- | --- | --- | --- |
| 0, δ=0.010 | 0.70795 / 0.11260 / 0.17945 | 0.76863 / 0.22501 | 0.60248 / 0.38729 | High band below 0.18 |
| 1, δ=0.016 | 0.71098 / 0.10885 / 0.18017 | 0.71847 / 0.27011 | 0.56225 / 0.42095 | First accepted |

The selected beta-one high-alive mass exceeds its threshold by only
`0.0001714`; this is a marginal pass and the comparison should retain
that limitation. The frame-six and final-frame clutter alternatives
both have substantial absolute mass. The original synthetic true
detection was relabeled as clutter in the decoy observation, as fixed
by the geometry rule; truth labels were not used by the exact inference.

The selected exact checkpoint was independently checked against the
enumerated toy at beta 0, 0.5 and 1 (worst toy joint TV `1.1e-15`).
On the selected case, direct and paired convolutions agreed at beta
one to `2.01e-11` joint TV, `3.52e-11` in log partition and
`3.25e-12` maximum cell difference; the other betas agreed more
closely. The beta-zero partition differed from the base normalizer by
`7.11e-15`. All raw-cell, imaginary and mass checks passed; beta-one
direct clipping TV was `1.93e-11`. These empirical numerical checks
do not constitute a formal interval proof.

The validated reference checkpoint is pinned by SHA-256
`d1834acda7022ba9d50d97a15115975909c6c938409969b8415e913dda62977c`.
No sampler move, high-particle run or untouched holdout was used.
