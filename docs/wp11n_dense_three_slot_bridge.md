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

Pending frozen enumeration and independent sampler runs.
