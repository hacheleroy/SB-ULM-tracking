# WP11w — Independently balanced eight-frame exact reference

WP11v had no sustained count/clutter loss under its frozen marker, but its
independent 13-detection target placed only 0.062 in the high-alive band.
This reference search fixes the selection rule **before computing any new
posterior**. It does not run a sampler, design a move, or open the WP8
holdout.

## Frozen candidate ladder

Generate the WP8 four-slot, eight-frame staggered crossing with detection
probability 0.68, expected clutter 0.28 per frame, central separation
0.007 and at most two clutter points per frame. Try trajectory/corruption
seed pairs, in order:

`(181,191), (193,197), (199,211), (223,227), (229,233), (239,241),
(251,257), (263,269)`.

Use only cardinalities to mark eligible cases: 17–19 total detections,
at least one in both frame six and frame eight, and at least seven across
frames three through five. Before evaluating any new posterior, record
the full ladder's cardinalities and detection hashes in a committed
checkpoint. If the generator raises an error for a pair, record the
error and mark that pair ineligible; do not replace it. This handling
was fixed after a geometry-only generator error at the last pair and
before any posterior computation. Evaluate **eligible** cases in seed
order and stop at the
first that satisfies every target-balance condition below. If none
qualifies, stop and report that this frozen ladder failed. Ineligible
cases never receive posterior evaluation.

## Exact target and selection rule

Use the beta-one four-slot squarefree subset reference from WP11v for
the joint of alive total × frame-six clutter × final-frame clutter.
Define low alive as ≤20, middle as 21, and high as ≥22. Accept the
first eligible case with both low and high mass ≥0.18, middle mass
≥0.05, at least two frame-six clutter counts each with mass ≥0.15,
and at least two final-frame clutter counts each with mass ≥0.15.
These thresholds measure the exact target's difficulty; they never use
sampler output. Record all attempted eligible cases and the exact
reasons for rejection. Do not change the ladder, bounds, or thresholds
after computing the first posterior.

For each beta-one screen, require raw minimum joint cell ≥−1e-10,
maximum relative imaginary residual ≤1e-8, and raw mass error ≤1e-8.
Stop the whole search on numerical failure; do not advance to the next
candidate as though it failed the balance rule. Allow up to 450 CPU
seconds per candidate, one process and one BLAS thread. A timeout is a
numerical/computational stop, not a license to choose a later case.

## Validation before any future inference

For the first passing candidate, independently validate the same
algorithm against complete enumeration on the separate four-slot,
three-frame toy at beta 0, 0.5 and 1 (log partition ≤1e-9, joint TV
≤1e-9, maximum cell error ≤1e-10). Recompute the selected target by
direct fourth power and paired square convolutions at beta 0, 0.5
and 1; require log partition difference ≤1e-8, joint TV ≤1e-7 and
maximum cell difference ≤1e-8, plus the raw-cell/imaginary/mass
envelope at every beta. Require the beta-zero partition to agree with
the model base normalizer within 1e-8 and the recomputed beta-one
alive-band masses to match the original screen within 1e-7. These are numerical cross-checks,
not a formal interval guarantee. Cap validation at 1,200 CPU seconds.
Pin the selected detection hash and validated reference checkpoint
before any later sampler protocol is designed or tested.

The WP11v/WP11t observations and all sampler traces are excluded from
candidate selection. No sampler or holdout run belongs to WP11w.

## Outcome

The geometry-only ledger was pinned before posterior computation at
SHA-256 `ba3d8aa8fb728d24b55cb966e6de0e53f8d0be0c3c2a17bdc2db1d82e12c15c4`.
Candidates 0, 2 and 4 are cardinality eligible. Candidate 7 raised the
ROI generator error and remains ineligible without replacement. The
exact balance screen is pending.
