# WP11ad — Fresh frame-six decoy absolute reference

The prior WP11ac independent search stopped at its third eligible
observation after a raw numerical cell crossed the frozen negative
threshold. That protocol remains failed. This is a **new** reference
design, frozen before examining any of its posteriors or running a
sampler. Its only purpose here is to supply an independent, calibrated
absolute comparison for the already specified whole-slot Gibbs move.

Use the sixteen new ordered seed pairs in
`src/sbulm/wp11ad_fresh_decoy_reference.py`, with the fixed staggered
four-slot generator, detection probability 0.68, clutter rate 0.28,
separation 0.007 mm and maximum two injected clutter points per frame.
Consider base cases with 17–18 total detections, two or three **true**
frame-six points and no frame-six clutter, at least one last-frame
point, and at least eight points across frames three through five.
Replace the largest frame-six reference ID with a clutter label and
move its y coordinate toward zero by 0.010, 0.016 or 0.022 mm. Reject
out-of-ROI variants. Traverse eligible variants in base-seed then
ascending-offset order, at most nine. The first exact posterior with
low and high alive-band masses each >=0.18, middle >=0.05, and at
least two clutter-count categories >=0.15 at both frame six and the
last frame is selected. An exact numerical failure stops the ladder;
unbalanced cases advance. Exhaustion leaves no reference.

The geometry-only ledger is
`results/wp11ad_geometry_ledger.json` (SHA-256
`9845b433dafa9ae498854794db30b34fd311e9f041fb6d59969a8a4f43d892b4`).
Its first nine eligible variants are 0–8, from bases 0, 2 and 7.
This ledger and the screening/validation code are committed before
posterior screening.

The reference uses the four-slot squarefree subset convolution at beta
0, 0.5 and 1.0. Direct and paired parenthesizations must agree within
1e-8 in log partition, 1e-7 in joint TV and 1e-8 in maximum cell.
Each must have minimum raw cell >=-1e-10, imaginary residual <=1e-8,
and raw mass error <=1e-8. Beta-zero log partition must match the
analytic base within 1e-8; the small directly enumerated toy must pass
its existing tighter checks. Single-thread CPU caps are 450 seconds
per screen and 1,800 seconds for selected validation. No sampler is run
before this succeeds and its result hash is pinned.

## Outcome

Variant 0, the first eligible 18-detection decoy, failed the frozen
numerical envelope: its minimum raw cell was -2.32e-8 (limit -1e-10),
with clipping TV 3.73e-8. It also lacked sixth-frame clutter balance.
The protocol requires stopping immediately on a numerical failure, so
variants 1–8 were not screened. No WP11ad absolute reference was
selected. The raw screen is `results/wp11ad_screens/0.json`.

WP11ae is a separate, lower-density fresh-seed protocol committed before
posterior screening; neither this numerical threshold nor a rejected
WP11ad observation is reused.
