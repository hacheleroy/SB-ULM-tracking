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

## Exact reference and outcome

The first enumeration implementation materialized the whole support and
approached the container memory limit. It was stopped before producing any
posterior value or sampler run. The committed bounded-batch implementation
was checked against complete enumeration on a smaller case and then scored
all **8,375,839** frozen states in 1,066 process CPU seconds. Rate
reweighting at the grid endpoints and all eight exact base-normalizer
checks passed. The full predeclared model grid gave:

| Clutter/frame | Exact P(A) | Exact P(B) | Other |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.1707 | 0.2566 | 0.5727 |
| **0.35** | **0.2675** | **0.2147** | 0.5178 |
| 0.50 | 0.3462 | 0.1819 | 0.4719 |
| 0.75 | 0.4493 | 0.1409 | 0.4098 |
| 1.00 | 0.5285 | 0.1114 | 0.3601 |
| 1.50 | 0.6430 | 0.0728 | 0.2842 |
| 2.25 | 0.7531 | 0.0414 | 0.2054 |
| 3.00 | 0.8233 | 0.0251 | 0.1516 |

The frozen balance rule selected 0.35. Both masses exceed 0.10, and exact
P(B) rises from 0.05381 at beta 0.75 to 0.21473 at beta one, meeting the
late-mode criterion. The exact log evidence is 30.00985. This is a deliberately
conditioned sparse observation; its six detections are far fewer than the 19
in WP11k's difficult four-slot primary case.

The IID column below is the 95th percentile of 500 independent exact
posterior samples. The sampler columns summarize six independent seeds.
These are descriptive finite-sample comparisons, not a multiple-testing
adjusted calibration gate.

| Particles | Absolute metric | IID 95th | Sampler median | Sampler range | Above IID 95th |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Maximum three-mode error / three-mode TV | 0.048642 | 0.02928 | 0.01379–0.048638 | 0/6 |
| 512 | Full alive/clutter joint TV | 0.10538 | 0.08219 | 0.06716–0.09657 | 0/6 |
| 512 | Alive-total TV | 0.08313 | 0.05686 | 0.04266–0.07069 | 0/6 |
| 512 | Absolute clutter-mean error | 0.03960 | 0.02539 | 0.00835–0.03657 | 0/6 |
| 2,048 | Maximum three-mode error / three-mode TV | 0.024224 | 0.01112 | 0.00383–0.024002 | 0/6 |
| 2,048 | Full alive/clutter joint TV | 0.05258 | 0.04445 | 0.03632–0.04738 | 0/6 |
| 2,048 | Alive-total TV | 0.04076 | 0.02906 | 0.02079–0.03543 | 0/6 |
| 2,048 | Absolute clutter-mean error | 0.02107 | 0.00616 | 0.00434–0.02398 | 1/6 |

The near-boundary 512 maximum mode error is only about 0.0000038 below
the simulated 95th percentile; the corresponding 2,048 margin is about
0.00022. These reference quantiles themselves have Monte Carlo uncertainty.
At 512, mode A signed error ranged from -0.0468 to +0.0274, and B from
-0.0311 to +0.0353. At 2,048, A ranged from -0.0121 to +0.0240, and B
from -0.0170 to +0.0113. The mean signed A/B errors were -0.0051/-0.0012
and +0.0011/-0.0043 respectively. Estimated-minus-exact log evidence
ranged -0.3380 to +0.2572 at 512 and -0.2208 to +0.0098 at 2,048.
Mean process CPU was 27.6 versus 103.8 seconds per sampler run. The
observer recorded accepted A↔B changes in PG and other paired proposals;
these counts are movement diagnostics, not independent mode samples.

**Decision:** `three_slot_absolute_audit_complete_no_move_selected`. This
three-slot eight-frame exact posterior retains substantial late competing
alive/clutter modes. On its sparse observation, the unchanged sampler's
mode and full-joint errors were within the simulated IID 95th-percentile
envelopes on all six seeds at each size, with the margins above interpreted
cautiously. One sixth-frame clutter-mean error at 2,048 exceeded its
descriptive envelope. This does not resolve the four-slot/19-detection
instability of WP11k. The next diagnostic should increase observation
density or approach four slots **while retaining an independently validated
absolute reference**; it must freeze that reference and criteria before
considering another sampler move. No 8,192-particle run or untouched
holdout was used.
