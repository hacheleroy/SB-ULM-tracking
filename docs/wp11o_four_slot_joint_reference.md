# WP11o — Paired four-slot absolute joint-cell references

WP11n found 2/6 high-particle full alive/clutter joint-TV values above a
descriptive exact IID q95, while all coarse mode errors stayed below their
reference. This experiment raises the latent cardinality to four slots and
predeclares a **cell-level** comparison on two distinct frozen observations.
Each observation has five frames and six detections, so this bridges toward
WP11k's four-slot/eight-frame, 19-detection case without claiming to
reproduce it. The protocol and runner are committed before exact posterior
evaluation or sampler inference.

## Frozen observations and model

Times are 0–4 ms, separated by 1 ms. The two observations have the same
cardinalities `[1, 1, 1, 1, 2]`, giving **24,193,016** labelled states
each. Coordinates are in the model's units:

| Frame | Primary detections | Replica detections |
| ---: | --- | --- |
| 0 | `(-0.019, 0.001)` | `(-0.017, -0.002)` |
| 1 | `(-0.011, 0.002)` | `(-0.010, -0.001)` |
| 2 | `(-0.004, 0.001)` | `(-0.001, 0.001)` |
| 3 | `(0.003, -0.001)` | `(0.005, 0.002)` |
| 4 | `(-0.003, 0.002)`, `(0.010, -0.001)` | `(-0.006, 0.001)`, `(0.013, 0.000)` |

The `_case_metadata` SHA-256 values are
`1fae28c9d57ea22d6d0a6a3457dec90d6acddae910541f583f9497d1effbd5c1`
for primary and
`ccc188850719d35edbd3fcfe018eab63f6ad5f5b34c2daabe17e267ec65f6d4f`
for replica. Use four exchangeable slots, localization standard deviation
0.005, Matérn coordinate scales `(0.02, 0.01)`, length scale 0.003,
detection probability 0.55, ROI `[-0.04, 0.04]` in both coordinates,
and the default birth/death prior. The only model parameter varied is
expected clutter per frame on `[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0]`.

Define a coarse, label-invariant **A** mode with at most nine alive
slot-frames and at least one final-frame clutter detection, and **B** with
at least ten alive slot-frames and no final-frame clutter. All remaining
states form **other**. Enumerate all eight clutter-rate variants for each
observation. Select the rate independently per observation by maximizing
`min(P(A), P(B))`, with ties going to the lower rate. Report the exact
mode masses at all rates and at tempered beta `[0, .25, .5, .75, .9, 1]`.
Report whether each selected A and B mass reaches 0.10; a weakly balanced
case is reported and still audited, without revising the observation or
rate grid. The primary output is the **full 21 by 3 joint table** of total
alive slot-frames (0–20) and final-frame clutter count (0–2), not only
the coarse modes.

## Exact reference and predeclared comparison

Score every labelled state in bounded batches and check the Poisson
clutter-rate reweighting identity directly at the grid endpoints. Check
all eight enumerated base normalizers against the model's analytic base
normalizer. Validate the bounded enumerator against full enumeration on a
small case. Save the exact posterior joint table, marginals, and log
evidence for both observations **before** any sampler run. The reference
process CPU cap is 6,000 seconds per observation; a cap or check failure
stops the study rather than changing the observation.

For each selected exact posterior and particle count, generate **1,000**
independent multinomial draws of the entire 63-cell joint table. Their
95th percentiles give descriptive per-run baselines for full joint TV,
maximum absolute signed cell error, maximum coarse-mode error, alive-total
TV, and mean clutter error. Seeds are 33331 and 33337 for primary and
replica. Generate another 1,000 independent groups of six draws at each
particle count. The q95 of the **maximum absolute six-run mean signed cell
error across all 63 cells** is a familywise cell-search reference. The
entire six-run mean cell table is reported, including its signed
exceedances of this threshold.

Run the unchanged WP11g 10% four-path PG sampler with six independent
one-thread seeds per observation and particle count. First complete both
exact references; then run primary seeds 1000–1005 at 512 and 2,048,
followed by replica seeds 1010–1015 at 512 and 2,048. Each invocation is a
separate process. Sampler process CPU caps are 600 and 1,600 seconds.
A passive observer records stage mode weights, resampling effects, and
accepted A↔B paired transitions without changing the random stream.

The frozen replication rule has two separate flags at 2,048 particles:

1. `replicated_joint_tail`: **both** observations have at least **2/6**
   full-joint-TV errors above their own exact IID per-run q95.
2. `replicated_cell_shift`: at least one identical `(alive total, clutter)`
   cell has a six-run mean signed error beyond its own observation's
   familywise group q95 in **both** observations with the **same sign**.

Report both flags, the per-seed joint and cell errors, all signed cell
mean residuals, exact cell masses, IID references, mode errors, and costs.
These are diagnostic signals; neither is a sampler-move selection gate.
Multiple descriptive metrics and a limited two-observation sample preclude
a broad claim of calibration on the eight-frame primary case. The two
designed observations are distinct, but they are not random draws from a
population of cases. No
8,192-particle run or untouched holdout is opened.

Runner: `src/sbulm/wp11o_four_slot_joint_reference.py`. Checkpoint:
`results/wp11o_four_slot_joint_reference.json`.

## Exact references and measured outcome

Pending the frozen exact and sampler runs.
