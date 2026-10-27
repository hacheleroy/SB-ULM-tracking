# WP11ah — Fresh balanced crowded absolute reference

WP11ag passed its exposed 19-detection and independent 17-detection
transfer gates. Its transfer posterior had strongly concentrated
frame-six clutter, so it did not validate balanced crowded modes.
The older WP11ae screen had alive-band and frame-six alternatives in
some 17-detection cases, while the final-frame clutter distribution
remained too concentrated. This prospective search changes the
**observation geometry**, with no sampler output in its selection.

## Geometry and fixed order, before posterior screening

The code in `src/sbulm/wp11ah_balanced_crowded_reference.py` fixes 100
unused trajectory/corruption pairs `(2411+11i, 2417+11i)` for
`i=0..99`. It uses the four-slot staggered truth mask, detection
probability 0.68, clutter rate 0.28, central separation 0.007 mm,
and at most two injected clutter points per frame. A base is eligible
only with 17–18 detections, 2–3 true frame-six detections and zero
sixth clutter, exactly two true final-frame detections and zero final
clutter, at least six points in frames three through five and at
least one in frame seven. Generator errors are recorded and skipped.

At both frame six and the final frame, choose the detection with the
largest nonnegative synthetic reference ID. Move its y coordinate
toward zero by a fixed offset and relabel it as injected clutter;
all x coordinates, remaining positions and cardinalities stay fixed.
Sixth-frame offsets are 0.010, 0.016 and 0.022 mm, and final-frame
offsets are 0.004, 0.008 and 0.012 mm. Out-of-ROI variants are
recorded and skipped. The exact inference uses positions, not the
reference labels. Sort eligible bases by increasing total count and
then seed index, and screen variants round-robin across bases for
each sixth-then-final offset pair. Screen no more than 18 eligible
variants, in this exact order; never substitute a failed seed.

Only geometry was inspected during design. An initial ledger ordered
all nine offsets of the first base before the second; its SHA-256 is
`863bde060ec2edfaeb4a0b0cf73e696992aac7ff9b59b4f88e26582416eef268`.
The geometry-only round-robin revision gives all five eligible bases
an early screen, before any posterior is computed. The final pinned
ledger is `results/wp11ah_geometry_ledger.json` (SHA-256
`a5978ee5e8e98ed537159ecc98888dd19e04e5745e7afb8a79f3ec064822d2d3`);
the initial feasibility ledger is also retained. The first eligible
bases have 17, 17, 18, 18 and 18 detections. This code, rule and both
ledgers are committed before the first WP11ah posterior.

## Exact acceptance and stop rule

Compute the beta-one four-slot squarefree joint of alive total ×
frame-six clutter × final clutter. Define low alive as <=20, middle
as 21 and high as >=22. Accept the **first** candidate with both low
and high alive masses >=0.18, middle >=0.05, and at least two clutter
categories each >=0.15 in *both* frame six and the final frame.
An unbalanced numerically sound candidate advances; any numerical
failure stops the entire search. Exhaustion yields no reference.

Every screen must have minimum raw cell >=-1e-10, maximum relative
imaginary residual <=1e-8, raw mass error <=1e-8, one thread and
450 process CPU seconds at most. For the first accepted case, validate
the separate toy by direct enumeration at beta 0, 0.5 and 1; compare
direct-fourth and paired-square subset parenthesizations at all three
betas. Require absolute log-partition discrepancy <=1e-8, joint TV
<=1e-7 and maximum cell difference <=1e-8, with the same raw-cell,
imaginary and mass limits for each method. Beta-zero log partition
must agree with the analytic base within 1e-8; beta-one alive bands
must reproduce the original screen within 1e-7. Use a single-thread
1,800 process-CPU-second validation cap. A failure yields no validated
reference. These numerical checks are empirical cross-validation, not
a formal interval guarantee.

If one case passes, pin its checkpoint hash and freeze a matched
control versus unchanged WP11ag whole-slot move comparison before
sampler inference. The previously exposed 19-detection primary remains
a separate sentinel. The untouched WP8 holdout stays closed.

## Outcome

Exact screening pending the committed prospective freeze.
