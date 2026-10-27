# WP11ab — Frozen late local-pair window comparison

WP11aa's four-particle suffix PG improved many exposed runs but missed the
absolute 19-detection limits, the 17-detection mode replication rule and
one cost limit on each case. This new **single move** retains the existing
local two-slot multiple-try kernel and spends its late-stage proposals
on the two frames with the predeclared clutter comparisons. Its design
uses those known development results; neither observation is a fresh
holdout. The implementation and gate were frozen at commit `3201c324`
before candidate inference.

## Fixed target-preserving move

At beta below 0.75, the sampler is bit-for-bit identical to the archived
WP11g control with matched seeds. At beta at least 0.75, half of its
**local paired proposals** draw their window start from frame six with
probability 0.75 or the final frame with probability 0.25; the other
half draw uniformly among all eight frames. Effective start probabilities
are 0.4375 at frame six, 0.1875 at the final frame and 0.0625 at each
other frame. The pair of slots remains uniformly selected, and the
one-/two-frame length rule, two-candidate multiple-try correction, 10%
four-particle full-path PG mixture, other move proportions, tempering,
particle count (2,048), and ordinary eight plus eight sweeps are unchanged.
There is no new conditional particle filter or extra sweep.

Every window probability depends only on beta and frame index. Given a
selected pair and window, the existing finite forward/reverse candidate
set and exact multiple-try Hastings correction are unchanged. A fixed
mixture of these invariant kernels preserves every tempered target. A
three-frame exact-posterior finite-sample check at beta 0.75, support
symmetry for each selected window, and identical below-switch transitions
must pass before inference. The exact small-case test is a regression
check, not evidence of eight-frame mixing.

## Pinned inputs and frozen decision

Reuse the exact 19-detection reference, the independently designed
17-detection decoy reference, and their **archived** unchanged control
arms at seeds `1300..1305` and `1500..1503`. The reference and control
hashes remain those in `wp11aa_tempered_suffix_pg.md`. Pin the full
WP11aa result at SHA-256
`620b5db1603af7626933000b70b38ba40b5239805ccf1a9ea247e5d55292a1ba`
for the cost comparator; WP11aa does not become the accuracy control.
Run all ten candidate seeds in separate one-thread processes, with a
500 process-CPU-second cap each. A cap or reference mismatch stops the
comparison without replacing a seed.

**Accuracy and replication:** retain every WP11aa primary and decoy
condition unchanged except its per-seed cost limit. The 19-detection
primary requires at least 4/6 paired full-joint TV wins; candidate
medians at most 0.12 full joint, 0.10 alive-band × sixth-clutter TV and
0.08 maximum alive-band mass error; each median at most 0.80 of its
control; and no more than +0.020 median regression on each of the four
secondary metrics. The 17-detection decoy requires at least 3/4 paired
full-joint wins, median full TV at most 0.12 and at most 0.90 of
control, band × sixth at most 0.10, maximum alive-band error at most
0.08, maximum cell error at most 0.04, no more than +0.020 secondary
regression, and at least 3/4 runs **simultaneously** meeting band ×
sixth ≤0.10, maximum alive-band error ≤0.08, and absolute clutter-one
mass error ≤0.06 at both sixth and final frames.

**Cost:** every candidate/control CPU ratio must be at most 1.25,
and the candidate median CPU time must be at most 0.80 of WP11aa's
median on **each** observation (WP11aa medians 346.72 and 333.95
seconds). These caps replace WP11aa's 1.50 per-seed rule. Both case
gates must pass. Record complete 3D joints, exact errors, per-seed
CPU times, evidence estimates and stage summaries. No threshold,
temperature switch, frame weight or seed may change after seeing a
candidate result. Even a pass is development evidence and requires a
new independently selected reference and replication rule before a
calibration or holdout claim. The untouched holdout stays closed.

Run `python -m sbulm.wp11ab_late_pair_window --case protocol` first,
then `--case run --observation primary_19 --seed 1300` and every other
fixed case/seed, then `--case aggregate`. Candidate checkpoints are
written under `results/wp11ab_runs/`; the joint comparison goes to
`results/wp11ab_late_pair_window.json`.

## Outcome

All ten fixed seeds finished within the 500-second CPU cap. The frozen
**combined gate failed**: the 17-detection case passed every criterion,
but the 19-detection case missed accuracy, paired replication and both
cost conditions. The move is not selected. The generated full comparison
is `results/wp11ab_late_pair_window.json` (SHA-256
`ab05dec063aa142e0504cf4a6c23a3abfb0db58730d569556f1e288251355c4c`),
with the ten per-seed stage and joint checkpoints under
`results/wp11ab_runs/`.

| Case | Paired full-joint wins | Control → candidate median full-joint TV | Median band × sixth TV | Median max alive-band error | Mode replication | Median candidate / WP11aa CPU | Maximum candidate / control CPU | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 19 detections | 3/6 (needs 4/6) | 0.19676 → 0.18050 (needs ≤0.12 and ≤0.80 of control) | 0.13707 (needs ≤0.10) | 0.08844 (needs ≤0.08) | — | 0.8334 (needs ≤0.80) | 1.3494 (needs ≤1.25) | Fail |
| 17 detections | 3/4 | 0.11072 → 0.09693 | 0.04942 | 0.03224 | 3/4 simultaneous modes | 0.7600 | 1.1872 | Pass |

The primary's three other guardrails passed, but its full-joint,
band × sixth and alive-band median ratios to control were about 0.917,
0.854 and 0.864, all above the required 0.80. The decoy passed its
absolute cell and secondary guardrails as well as its replicated mode
limits. Its fourth seed had full-joint TV 0.17787, so the 3/4 result
does not imply every replication was accurate.

Focusing short local windows helped some modes in the less crowded
geometry, but did not reliably transfer probability between the
19-detection alive-count bands. This is development evidence about this
specific move, not a calibrated sampler or a new independent case.
Keep the holdout closed. A next proposal should first separate the
primary's late count-band errors from its clutter errors and freeze a
move capable of changing the long-range count pattern; it needs a new
independently chosen absolute comparison before any calibration claim.
