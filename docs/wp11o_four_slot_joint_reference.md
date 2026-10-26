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

Both bounded exact passes scored all **24,193,016** labelled states and
passed the endpoint Poisson rate-reweighting checks and all eight analytic
base-normalizer comparisons. Primary took 3,861.9 process CPU seconds;
replica took 3,538.3, both below the frozen 6,000-second cap. The
predeclared rate rule selected **1.0 for each**. The entire grid is:

| Clutter/frame | Primary P(A) | Primary P(B) | Replica P(A) | Replica P(B) |
| ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.0645 | 0.4250 | 0.0683 | 0.4261 |
| 0.35 | 0.1075 | 0.3950 | 0.1134 | 0.3944 |
| 0.50 | 0.1467 | 0.3681 | 0.1543 | 0.3662 |
| 0.75 | 0.2048 | 0.3288 | 0.2147 | 0.3252 |
| **1.00** | **0.2556** | **0.2953** | **0.2672** | **0.2904** |
| 1.50 | 0.3410 | 0.2410 | 0.3547 | 0.2347 |
| 2.25 | 0.4408 | 0.1819 | 0.4561 | 0.1749 |
| 3.00 | 0.5188 | 0.1400 | 0.5346 | 0.1331 |

Both selected cases have A and B above 0.10; other mass is 0.4491 and
0.4424. Exact P(B) rises from 0.06646 / 0.06245 at beta 0.75 to
0.29525 / 0.29039 at beta one. Exact log evidence is 32.30207 / 32.15707.
The top primary exact joint cells are `(alive, clutter)=(8,0):0.11409`,
`(9,0):0.11153`, and `(10,0):0.10256`; replica has the same top cell
coordinates with masses 0.11028, 0.10851, and 0.10022. The two-reference
checkpoint was committed before sampling.

The table summarizes six independent seeds per row. Ranges are absolute
errors; `exceed` counts per-run errors above that case's 1,000-draw exact
IID q95. These q95 comparisons are descriptive rather than a joint
multiple-testing gate.

| Case | Particles | Metric | IID q95 | Sampler median | Sampler range | Exceed |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Primary | 512 | Joint TV | 0.10705 | 0.08028 | 0.06776–0.08852 | 0/6 |
| Primary | 512 | Maximum cell error | 0.03438 | 0.02372 | 0.01574–0.02942 | 0/6 |
| Primary | 512 | Maximum mode error | 0.04654 | 0.01735 | 0.01334–0.03310 | 0/6 |
| Primary | 512 | Alive-total TV | 0.07945 | 0.05524 | 0.03627–0.07209 | 0/6 |
| Primary | 512 | Absolute clutter-mean error | 0.04478 | 0.01953 | 0.01578–0.03304 | 0/6 |
| Primary | 2,048 | Joint TV | 0.05364 | 0.04715 | 0.04306–0.04787 | 0/6 |
| Primary | 2,048 | Maximum cell error | 0.01738 | 0.01043 | 0.00957–0.01648 | 0/6 |
| Primary | 2,048 | Maximum mode error | 0.02354 | 0.01224 | 0.00473–0.01963 | 0/6 |
| Primary | 2,048 | Alive-total TV | 0.03899 | 0.03104 | 0.02053–0.04373 | 1/6 |
| Primary | 2,048 | Absolute clutter-mean error | 0.02378 | 0.00497 | 0.00114–0.01676 | 0/6 |
| Replica | 512 | Joint TV | 0.10684 | 0.08146 | 0.06896–0.08607 | 0/6 |
| Replica | 512 | Maximum cell error | 0.03388 | 0.01964 | 0.01595–0.02283 | 0/6 |
| Replica | 512 | Maximum mode error | 0.04922 | 0.02423 | 0.01657–0.02697 | 0/6 |
| Replica | 512 | Alive-total TV | 0.07920 | 0.05548 | 0.03618–0.06587 | 0/6 |
| Replica | 512 | Absolute clutter-mean error | 0.04566 | 0.01734 | 0.00855–0.02660 | 0/6 |
| Replica | 2,048 | Joint TV | 0.05394 | 0.03963 | 0.02930–0.04960 | 0/6 |
| Replica | 2,048 | Maximum cell error | 0.01716 | 0.01139 | 0.00831–0.01672 | 0/6 |
| Replica | 2,048 | Maximum mode error | 0.02383 | 0.01070 | 0.00388–0.02330 | 0/6 |
| Replica | 2,048 | Alive-total TV | 0.03990 | 0.02850 | 0.01864–0.03395 | 0/6 |
| Replica | 2,048 | Absolute clutter-mean error | 0.02270 | 0.00951 | 0.00367–0.02123 | 0/6 |

At 2,048, the largest absolute six-seed mean cell error is **0.00476**
for primary at `(7,1)` and **0.00541** for replica at `(10,0)`. Their
familywise six-draw q95 values are **0.00718** and **0.00684**. Neither
case has a flagged cell, let alone the same signed cell in both. The
predeclared `replicated_joint_tail` and `replicated_cell_shift` flags are
both **false**; status is
`four_slot_joint_audit_complete_no_replicated_signal`. Mean sampler CPU
seconds were 26.3 / 97.3 (primary 512 / 2,048) and 25.3 / 89.9
(replica 512 / 2,048). The single primary 2,048 alive-total TV
exceedance is retained in the table. Replica 512 log-evidence errors
were all negative (-0.5374 to -0.0655), a descriptive feature outside
the predeclared joint-cell flags.

**Interpretation:** these two designed four-slot/five-frame cases do not
replicate WP11n's single-case joint-TV tail, and their coarse modes and
cell means show no flagged absolute error under the frozen reference.
This does not resolve the four-slot/eight-frame, 19-detection WP11k
instability: the present observations contain only six detections and
shorter temporal histories. No new sampler move was selected; no
8,192-particle run or untouched holdout was opened. A further move
would need a separately frozen development geometry that preserves
the eight-frame mode difficulty, with a credible independent absolute
reference or validated approximation where full enumeration is too large.
