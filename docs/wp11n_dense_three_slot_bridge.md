# WP11n — Denser three-slot exact bridge

WP11m retained two competing alive/clutter modes under an independent exact
reference, but its eight-frame observation had only six detections. WP11k's
four-slot primary case has 19. This experiment fills one early blank frame
of WP11m to increase observation density and temporal coupling while keeping
complete labelled-state enumeration feasible. Its seven-detection support
contains 20,768,284 states (2.48 times WP11m). This protocol, runner, and
criteria are committed before any posterior evaluation or sampler run.

## Frozen observation and model

The three exchangeable slots have eight frames at times 0–7 ms. The sequence
differs from WP11m only at frame 1, which now has `(-0.014, 0.001)`:

| Frame (zero based) | Detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | `(-0.014, 0.001)` |
| 2 | `(-0.007, 0.002)` |
| 3 | none |
| 4 | `(0.001, 0.000)` |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |
| 7 | none |

The `_case_metadata` observation SHA-256 is
`11a2f6f7c529f080ed0fdbb3eaf3da39c2b9170ce660e95f0b119de5d241734b`;
the cardinalities are `[1, 1, 1, 0, 1, 2, 1, 0]`. Use localization standard
deviation 0.005, Matérn coordinate scales `(0.02, 0.01)`, length scale
0.003, detection probability 0.55, ROI `[-0.04, 0.04]` in both axes, and
the default birth/death prior. Vary only expected clutter per frame over
`[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0]`. These are model variants
of one observation, not independent cases.

Define label-invariant mode **A** as at most 12 alive slot-frames with one or
two frame-six detections assigned to clutter; **B** as at least 13 alive
slot-frames with no frame-six clutter; **other** contains every remaining
state. Exactly enumerate all eight clutter variants. Select the rate that
maximizes `min(P(A), P(B))`, breaking ties toward the lower rate. The bridge
is valid only if both selected beta-one exact mode masses are at least 0.10
and exact `P(B)` at beta 0.75 is at most half its beta-one mass. Otherwise
save `bridge_invalid_mode_or_late_criterion` and stop. Do not change the
observation, rates, modes, or rule after inspecting exact or sampler results.

## Independent reference and fixed comparison

Enumerate the 20,768,284 labelled states in bounded batches, scoring the
collapsed Matérn likelihood and normalized birth/death/count base. Check
the clutter-rate reweighting identity directly at the grid endpoints and
all eight enumerated base normalizers against the model's analytic base
normalizers. Save exact mode masses for the full rate grid and, at the
selected rate, at beta `[0, 0.25, 0.5, 0.75, 0.9, 1]`. Save beta-one
alive-total by frame-six clutter, alive and clutter marginals, and exact log
evidence. Cross-check the bounded support enumerator against full
enumeration on a small case. The exact reference has a 4,200 process CPU
second cap. A cap or validity failure ends this frozen bridge; do not tune
the case after seeing results.

Draw 500 independent multinomial samples from the exact posterior for each
particle count (seed 32323). Report descriptive median and 95th-percentile
finite-sample errors for maximum three-mode probability, three-mode TV,
the full alive/clutter joint TV, alive-total TV, and mean frame-six clutter.
Run the **unchanged** WP11g 10% four-path PG sampler in separate one-thread
processes: 512 particles with seeds 940–945, then 2,048 with the same six
seeds. Per-run CPU caps are 450 and 1,200 seconds. A passive observer
records stagewise mode weights, resampling effects, and accepted A↔B
transitions for PG and other paired proposals; it must not alter the
sampler trajectory.

Report each seed's signed A/B errors, the six-seed error ranges, mean
signed errors, IID-95th exceedance counts, log-evidence errors, and CPU
costs. If **at least three of six** 2,048-particle seeds exceed the exact
IID 95th percentile for maximum three-mode error, label the result
`dense_three_slot_mode_error_signal_requires_replication`; otherwise label
it `dense_three_slot_absolute_audit_complete_no_move_selected`. This is a
predeclared diagnostic flag, not a new sampler-move selection gate. Neither
outcome justifies opening an 8,192-particle run or the untouched holdout.

Runner: `src/sbulm/wp11n_dense_three_slot_bridge.py`. Checkpoint:
`results/wp11n_dense_three_slot_bridge.json`. Even a valid seven-detection
bridge cannot certify the 19-detection four-slot primary observation.

## Exact reference and outcome

The bounded pass scored all **20,768,284** states in 2,575.4 process CPU
seconds, below the frozen cap. The direct Poisson clutter-rate reweighting
checks at both grid endpoints passed, as did the eight comparisons of the
enumerated base normalizer with the analytic model normalizer. The full
predeclared grid gave:

| Clutter/frame | Exact P(A) | Exact P(B) | Other |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.1549 | 0.3118 | 0.5333 |
| **0.35** | **0.2452** | **0.2697** | **0.4851** |
| 0.50 | 0.3199 | 0.2355 | 0.4447 |
| 0.75 | 0.4194 | 0.1908 | 0.3899 |
| 1.00 | 0.4971 | 0.1569 | 0.3460 |
| 1.50 | 0.6116 | 0.1094 | 0.2790 |
| 2.25 | 0.7246 | 0.0668 | 0.2086 |
| 3.00 | 0.7991 | 0.0422 | 0.1587 |

The frozen balance rule selected rate **0.35**. Both mode masses exceed
0.10. Exact P(B) rises from **0.083615** at beta 0.75 to **0.269732** at
beta one, satisfying the late-emergence criterion. Exact log evidence is
37.966625. The independent exact checkpoint was committed before the
sampler phase.

The IID q95 below comes from 500 independent draws of the same particle
count from the selected exact posterior. The sampler median and range
summarize six independent seeds; the range is for the absolute error.
These are descriptive, unadjusted finite-sample comparisons.

| Particles | Absolute metric | IID 95th | Sampler median | Sampler range | Above IID 95th |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Maximum three-mode error / three-mode TV | 0.04755 | 0.02095 | 0.01475–0.03623 | 0/6 |
| 512 | Full alive/clutter joint TV | 0.10192 | 0.08137 | 0.07101–0.09563 | 0/6 |
| 512 | Alive-total TV | 0.08135 | 0.05589 | 0.05003–0.06324 | 0/6 |
| 512 | Absolute clutter-mean error | 0.03716 | 0.01753 | 0.00581–0.04693 | 1/6 |
| 2,048 | Maximum three-mode error / three-mode TV | 0.02473 | 0.01069 | 0.00303–0.01475 | 0/6 |
| 2,048 | Full alive/clutter joint TV | 0.05168 | 0.04717 | 0.03978–0.05461 | 2/6 |
| 2,048 | Alive-total TV | 0.03973 | 0.02999 | 0.02518–0.04139 | 1/6 |
| 2,048 | Absolute clutter-mean error | 0.02046 | 0.00371 | 0.00103–0.02251 | 1/6 |

At 512 particles, signed A errors span -0.0362 to +0.0185 and B errors
-0.0217 to +0.0310, with mean signed errors -0.0030 and +0.0018.
At 2,048, A spans -0.0147 to +0.0028 and B +0.0008 to +0.0120,
with means -0.0079 and +0.0045. Estimated-minus-exact log evidence
spans -0.3830 to +0.2135 at 512 and -0.2213 to +0.1890 at 2,048.
Mean CPU was 34.2 and 137.3 seconds per run. The passive observer
recorded accepted A↔B transitions with both PG and other paired
proposals. These are movement counts, not independent posterior samples.

**Decision:** `dense_three_slot_absolute_audit_complete_no_move_selected`.
Zero of six high-particle mode errors exceeded the frozen exact IID q95,
so the predeclared mode-error signal did not fire. Two of six high-particle
full-joint TVs exceeded their descriptive IID q95 (seeds 942 and 944),
and one alive-total and one mean-clutter error exceeded their respective
q95. This is a warning about the **shape of the joint alive/clutter
distribution** even though the coarse A/B/other masses were close to the
exact reference. The 500-replicate q95 estimates and several metrics are
not a multiple-testing-adjusted gate; this observation alone does not
identify a sampler defect or a move to deploy.

WP11n adds one detection to WP11m but remains far below WP11k's
four-slot/19-detection case. The next diagnostic should freeze a new
independent absolute reference that stresses more of the joint
alive/clutter cells, or a tractable four-slot geometry. Predeclare its
cell-level or joint comparison and replicate rule before inference. No
8,192-particle run or untouched holdout was used.
