# WP11ac — Crowded count-band diagnosis and independent reference

## Retrospective diagnosis

The six pinned WP11ab primary candidate joints were decomposed against the
independent WP11t exact joint, without another sampler run. The exact
low/middle/high alive-band masses are 0.41116/0.13952/0.44932, where the
bands are alive slot-frame totals <=20, 21, and >=22. Alive-band TV is a
rigorous lower bound on full 3D joint TV after marginalization.

| Seed | Signed low / high band errors | Band TV / full-joint TV | Sixth clutter-one signed error |
| --- | --- | ---: | ---: |
| 1300 | +0.1504 / -0.1505 | 0.1505 / 0.1747 | -0.1001 |
| 1301 | +0.2470 / -0.2193 | 0.2470 / 0.2604 | -0.0913 |
| 1302 | +0.0078 / -0.0035 | 0.0078 / 0.1287 | -0.0494 |
| 1303 | -0.0420 / +0.0204 | 0.0420 / 0.1863 | -0.0933 |
| 1304 | +0.1201 / -0.1349 | 0.1349 / 0.2108 | -0.1621 |
| 1305 | +0.0229 / -0.0172 | 0.0229 / 0.1486 | +0.0781 |

Thus three seeds have band TV >=0.10, with 64–95% of their full-joint
TV bounded below by the band marginal. The other seeds retain sizable
joint error without the same band imbalance. The archived stage summaries
report frame-count-signature changes but do not track transitions between
the global alive bands; they cannot establish the precise onset or mechanism
of a mode switch. The proposed move addresses the observed global interval
barrier, while the independent comparison must check clutter as well.
The machine-readable retrospective diagnostic is
`results/wp11ac_count_band_diagnostic.json` (SHA-256
`c0e0e5c70d824d4545e0ef86177c0af7359b946ce9f8421bf0214517d5b907a5`).

## Geometry-only reference selection frozen before posterior inspection

Use sixteen ordered, previously unused trajectory/corruption seed pairs in
`wp11ac_independent_reference.py`. The generator and birth/death regime are
fixed at detection probability 0.68, expected clutter 0.28 per frame,
central separation 0.007 mm, maximum two injected clutter points per
frame, and the existing staggered four-slot truth mask. Eligibility uses
only cardinalities: 18–19 total detections, at least two in frame six,
one in the final frame, and eight across frames three through five.
Screen the **first five eligible** observations in seed order. The first
with low and high alive-band masses each >=0.18, middle >=0.05, and at
least two clutter-count categories >=0.15 at **both** frame six and the
last frame is the sole selection. A numerical failure stops selection;
an unbalanced observation advances to the next eligible entry. Exhaustion
stops without replacing seeds or testing a sampler.

The geometry-only ledger is `results/wp11ac_geometry_ledger.json`
(SHA-256 `8a9dd92829f62b6094913682950f8eb39eaaf7dfe48365e46890a0794d9f0623`).
Its first five eligible indices are 1, 2, 3, 6, and 8. This ledger and
the selection code are committed before any of those exact posteriors are
examined. No inference or move choice uses their posterior values.

For the selected observation, validate the four-slot exact joint with
the already tested squarefree subset convolution and two independent
parenthesizations at beta 0, 0.5 and 1. Require absolute log-partition
difference <=1e-8, joint TV <=1e-7, maximum cell difference <=1e-8,
raw mass error <=1e-8, imaginary residual <=1e-8, and minimum raw cell
>=-1e-10. The beta-zero partition must match the analytic base within
1e-8. The small directly enumerated toy must pass its existing tighter
checks. One-thread process CPU caps are 450 seconds per screen and
1,800 seconds for validation. A failed check stops before sampler inference.

## Next freeze

The intended distinct move is a whole-slot conditional Gibbs redraw:
sample one complete birth-to-death existence and assignment trajectory
from its exact tempered conditional given the other three slots. It can
cross an entire alive-band boundary in one step, and has no Metropolis
acceptance correction because the conditional is normalized exactly over
the finite one-slot support. Its mixture rate, beta switch, small exact
test, paired control seeds, absolute accuracy, replication and CPU rules
must be committed after validating the new reference and before any
candidate inference. Both the prior 19-detection primary and the newly
selected independent observation will be required; the untouched WP8
holdout remains closed.

## Outcome of the first independent ladder

The first two eligible observations (indices 1 and 2) passed the
numerical envelope but failed the sixth-frame clutter-balance rule:
their posterior probabilities of zero sixth-frame clutter were 0.9486
and 0.9649. Index 3 then failed the frozen numerical envelope with
minimum raw cell -4.66e-10 (limit -1e-10), despite a small total
clipping TV of 2.15e-9. The rule requires a stop on any numerical
failure. Indices 6 and 8 were therefore **not** screened, and no
reference or sampler comparison was selected from WP11ac.

This failure prompted a separate WP11ad geometry-only protocol with
fresh seeds and a predeclared frame-six decoy construction. It does not
change the WP11ac rule or reuse its rejected posterior.
