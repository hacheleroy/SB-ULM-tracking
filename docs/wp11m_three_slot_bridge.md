# WP11m — Three-slot exact eight-frame bridge

WP11l established absolute mode-weight calibration on a deliberately balanced
two-slot/eight-frame problem. WP11k's four-slot/eight-frame primary case still
has large between-seed clutter differences. This bridge increases latent
cardinality to three while keeping a complete, independent enumerated
posterior. It retains two detections at frame six. This protocol is committed
before evaluating its posterior or sampling it.

## Frozen observation and model

Times are 0–7 ms, separated by 1 ms. The three exchangeable slots observe:

| Frame (zero based) | Detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | none |
| 2 | `(-0.007, 0.002)` |
| 3 | none |
| 4 | `(0.001, 0.000)` |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |
| 7 | none |

The `_case_metadata` observation SHA-256 is
`005b33179c58d2385cfe6e49e3ff3ddb264e2f6ab41b4f31a56974250416e5d6`.
Its exact association support has **8,375,839** labelled states. The model
uses localization standard deviation 0.005, Matérn coordinate scales
`(0.02, 0.01)`, length scale 0.003, detection probability 0.55, ROI
`[-0.04, 0.04]` in both axes, and the default birth/death prior. Vary only
the expected clutter per frame on the fixed grid
`[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0]`. These variants share an
observation; they are not independent synthetic replicates.

The label-invariant modes are **A**, at most 12 alive slot-frames with at
least one of the two sixth-frame detections counted as clutter; **B**, at
least 13 alive slot-frames with no sixth-frame clutter; and **other**.
Enumerate all grid variants. Choose the member maximizing
`min(P(A), P(B))`, breaking ties toward the lower rate. The bridge is valid
only if both exact masses are at least 0.10 and exact `P(B)` at beta 0.75
is at most half its beta-one value. If either check fails, record
`bridge_invalid_mode_or_late_criterion` and stop. Do not change the
observation, grid, cutoffs, or validity criteria after inspecting results.

The reference enumerates every labelled state, computing the collapsed
Matérn likelihood in 20,000-state batches. The rate reweighting identity
for the Poisson clutter count is checked directly at both ends of the grid;
all eight enumerated base normalizers are checked against the model's exact
base normalizer. At the chosen rate, save exact three-mode masses at beta
`[0, 0.25, 0.5, 0.75, 0.9, 1]`, the beta-one alive-total by sixth-frame
clutter distribution, marginal counts, and log evidence. Report exact mode
weights for **all** rates.

Use 500 IID multinomial draws from this exact posterior at each particle
count, seed 31313, to define descriptive median and 95th-percentile
finite-sample errors for maximum mode probability, three-mode TV, full
alive/clutter joint TV, alive-total TV, and mean sixth-frame clutter.
Then use the unchanged WP11g 10% four-path PG sampler in twelve separate
one-thread processes: seeds 920–925 at 512 particles, then the same seeds
at 2,048. The reference has a 3,600 process-CPU-second cap; sampler caps
are 450 and 1,200 seconds. A passive observer records stage mode masses,
resampling changes, and accepted A↔B transitions for PG and other paired
proposals. The checkpoint enforces this run order.

Report signed A/B errors, absolute three-mode and full-joint error ranges,
the number of seeds beyond each IID 95th percentile, log-evidence error,
and CPU cost. This is an **absolute diagnostic**, not a selection gate for
a new sampler move. Even a valid three-slot case has fewer detections than
WP11k's difficult four-slot observation, so its result cannot certify that
case. No 8,192-particle run or untouched holdout is included.

Runner: `src/sbulm/wp11m_three_slot_bridge.py`. Saved protocol, exact
reference, and independent runs: `results/wp11m_three_slot_bridge.json`.
