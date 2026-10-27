# WP11ae — Fresh 16–17 detection decoy reference

The separate WP11ad ladder stopped on its first numerical screen.
WP11ae uses a new twenty-pair seed ladder and a lower observation count
to reduce subset-convolution cancellation; it does not change any
threshold from WP11ac or WP11ad. Its geometry-only code and ledger are
committed before examining a WP11ae posterior or running a sampler.

The observation rule in `src/sbulm/wp11ae_stable_decoy_reference.py`
uses the staggered four-slot truth mask, detection probability 0.68,
clutter rate 0.28, separation 0.007 mm and at most two injected clutter
points per frame. An eligible base has 16–17 detections, two or three
true frame-six points without frame-six clutter, at least one final
point, and at least six points in frames three through five. Replace
its largest frame-six reference ID with a clutter label, and move its
y coordinate toward zero by 0.010, 0.016 or 0.022 mm. Out-of-ROI
variants are ineligible. Traverse base seed then ascending offset and
screen at most nine eligible variants.

Select the first numerically valid joint with low and high alive-band
masses each >=0.18, middle >=0.05, and at least two clutter categories
with mass >=0.15 in both frame six and the last frame. On a numerical
failure, stop without substituting a later variant. If nine cases fail
balance, stop without a reference. The geometry ledger is
`results/wp11ae_geometry_ledger.json` (SHA-256
`21b2ced35088456bc8f6b6f31283e2d474f6357d468504fde2a096d5b9715afb`).
The first nine eligible variants are 0–8, from bases 6, 8 and 11.

Validate the first accepted joint using direct and paired squarefree
subset-convolution parenthesizations at beta 0, 0.5 and 1.0. The
absolute log-partition discrepancy must be <=1e-8, joint TV <=1e-7,
maximum cell discrepancy <=1e-8, raw mass error <=1e-8, imaginary
residual <=1e-8, and minimum raw cell >=-1e-10. The beta-zero partition
must agree with the analytic base within 1e-8; a directly enumerated
toy must pass its tighter prior checks. Single-thread process CPU caps
are 450 seconds per screen and 1,800 seconds for validation. A pass
will be pinned by SHA-256 before any sampler inference; failure stops.

## Outcome

All nine prespecified variants passed the exact numerical screen but
failed at least one balance condition. Variants 0–2 had insufficient
last-frame clutter ambiguity; variants 3–5 had insufficient high
alive-band and sixth-frame clutter-one mass; variants 6–8 had very
little high alive-band mass. Their screen checkpoints are archived in
`results/wp11ae_screens/`. The frozen ladder is exhausted and yields
**no** selected reference. Its thresholds were not revised.

A separate WP11af geometry-only selection now targets a rigorous
absolute *transfer check* without claiming balanced count modes.
The crowded 19-detection exposed case remains the count-mode sentinel.
