# WP11s — Frozen middle-frame ten-detection absolute audit

WP11r established two independently checked exact four-slot/eight-frame
references at eight and nine detections. Its nine-detection late-frame case
had two of six 2,048-particle full-joint TV errors beyond its exact-IID q95,
but the nested eight-detection case had none. This experiment increases
**middle-frame** density while preserving the difficult two-detection frame
six and last frame. It is a designed bridge toward WP11k's 19-detection
four-slot observation, not calibration of that observation.

## Observations and exact-support budget frozen before posterior evaluation

Times are 0–7 ms in 1 ms steps. Both cases have the WP11r nine-detection
geometry at frames 0, 1, 3, 5, 6 and 7:

| Frame | Shared detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | `(-0.014, 0.001)` |
| 3 | none |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |
| 7 | `(-0.011, 0.002)`, `(0.018, -0.001)` |

The **early-middle** case adds `(0.005, -0.001)` at frame 2 next to
`(-0.007, 0.002)`, and retains only `(0.001, 0.000)` at frame 4.
The **middle** case retains only `(-0.007, 0.002)` at frame 2 and adds
`(-0.011, 0.002)` at frame 4 next to `(0.001, 0.000)`.
Both have ten detections; they share the nine-point WP11r base and are
**dependent designed cases**, not independent population replicates.
They were chosen for positions along the competing trajectory hypotheses.
No result-dependent alteration of coordinates, observations, thresholds,
modes or seeds belongs to this experiment.

| Case | Cardinalities | SHA-256 of detection metadata | Canonical states | Labelled states |
| --- | --- | --- | ---: | ---: |
| Early-middle | `[1,1,2,0,1,2,1,2]` | `6b460f3d90d695f03e6dd836a8439e263604d6b36700dfcae64e9afdda5b6676` | 1,679,234,508 | 30,414,095,341 |
| Middle | `[1,1,1,0,2,2,1,2]` | `961a3ee11c17233630ca7e23dab61ccbfcad933cdde1def1ee9a41cf65e13bd9` | 1,878,017,514 | 34,201,578,205 |

The support sizes are independently computed from per-frame
`sum_k binomial(active,k) * n!/(n-k)!`, multiplied across frames and
summed over 91,390 nondecreasing interval-mask multisets. Labelled
counts weight each multiset by the number of distinct slot permutations.

Use four exchangeable slots, localization standard deviation 0.005,
Matérn coordinate scales `(0.02,0.01)`, length scale 0.003, detection
probability 0.55, ROI `[-0.04,0.04]` in each coordinate, and the default
birth/death prior. Vary expected clutter/frame over
`[0.05,0.1,0.15,0.2,0.35,0.5,0.75,1,1.5,2.25,3]`.

## Frozen exact method, validity gate and comparison

Sum the posterior over every compatible assignment for each
nondecreasing four-slot interval-mask tuple, weighted by slot-permutation
multiplicity. Cache the one-slot GP likelihood patterns and compare
factorized scores against the direct collapsed likelihood on fixed blocks.
Group contributions by their integer clutter count and recover each rate
by the exact finite polynomial in that rate; an independent small-array
test compares this reweighting against direct log-score sums at both grid
endpoints.
The inherited small-case test checks the quotient against full labelled
enumeration. Check the frozen support sizes, direct Poisson endpoint-rate
reweighting and **all eleven** analytic HMM base normalizers per case.
Each exact reference has a 5,000 CPU-second cap.

Keep the WP11q modes: A has at most 15 alive slot-frames and at least
one frame-six (index 5) clutter point; B has at least 16 alive slot-frames
and no frame-six clutter. For each case choose a clutter rate maximizing
`min(exact P(A), exact P(B))`, ties to the lower rate. In **both** cases,
exact selected A and B must each be at least 0.10, and exact B at beta
0.5 must not exceed half its beta-one probability on the grid
`[0,.25,.5,.75,.9,1]`. If either fails, save both references and stop
all sampler inference. Commit both exact checkpoints before any sampler
run.

The primary absolute table is alive total (0–32) × added-middle clutter
(0–2) × frame-six clutter (0–2) × final-frame clutter (0–2), **891
cells** per case. Save its collapsed 3D table with middle clutter summed,
modes, marginals, beta curves, all eleven-rate grids, and log evidence.
For each case and particle count, simulate 1,000 exact-posterior IID
multinomial draws (seeds 34501 and 34507). Compare each run's 4D joint TV,
maximum mode error and maximum 4D cell error to the corresponding
descriptive IID q95. Also report 3D joint TV, alive-total TV and all
three clutter-mean errors. Simulate 1,000 exact groups of six draws for
familywise q95 thresholds on both maximum six-run mean signed 4D cell
error and maximum six-run mean signed **collapsed 3D** cell error.

Only if both exact difficulty gates pass, run the **unchanged** WP11g
10% four-path PG mixture with a passive mode observer, separate
one-thread processes: early-middle seeds 1180–1185 at 512 and then 2,048;
middle seeds 1190–1195 at 512 and then 2,048. Process CPU caps are 600
and 1,600 seconds. At 2,048, freeze three paired descriptive flags:

1. `persistent_mode_tail`: at least **2/6** runs beyond case-specific
   exact-IID maximum-mode q95 in **both** cases.
2. `persistent_joint_tail`: at least **2/6** runs beyond case-specific
   exact-IID full 4D joint-TV q95 in **both** cases.
3. `shared_cell_shift`: the **same** collapsed 3D
   `(alive, frame-six clutter, final clutter)` cell has a mean signed
   residual beyond its case-specific familywise six-run q95 in both
   cases **with the same sign**. Report 4D flagged cells separately.

The two cases are dependent. The exact IID reference describes ordinary
multinomial noise, whereas SMC particles are correlated; q95
exceedances are diagnostic, not calibrated hypothesis tests. The flags
do not select a move or authorize 8,192 particles or the untouched
holdout. Ten detections cannot establish WP11k 19-detection calibration.

Runner: `src/sbulm/wp11s_middle_frame_density_audit.py`. Checkpoint:
`results/wp11s_middle_frame_density_audit.json`. The protocol, code and
tests precede both exact posterior evaluations.

## Paired exact checkpoints

Both exact sums passed the frozen canonical/labelled count checks,
direct collapsed GP and endpoint-rate checks, and all eleven analytic
HMM base normalizers. The early-middle result was committed first;
the **paired exact result was committed before any sampler inference**.
Each case passed the predeclared A/B mass and beta-0.5 difficulty gate.

| Case | Selected clutter/frame | Exact P(A) | Exact P(B) | Other | P(B), beta 0.5 | Log evidence | Exact CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early-middle | 1.5 | 0.340521 | 0.256287 | 0.403192 | 0.00002567 | 49.649772 | 1,088.5 |
| Middle | 1.5 | 0.225934 | 0.268918 | 0.505147 | 0.00002774 | 50.042000 | 1,214.6 |

The checkpoint stores all 891 exact joint cells, rate grids, beta
curves, marginals, IID baselines and normalizer-validated evidence for
each designed observation.

## Absolute sampler comparison

All 24 unchanged runs completed within the frozen per-process CPU caps.
Entries below give the six-seed median absolute error, the case-specific
exact-IID q95 in parentheses, and the count above that q95. The full
per-seed joint tables, signed errors, mode traces and CPU times are in
the result checkpoint.

| Case | Particles | Max mode error | Full 4D joint TV | Collapsed 3D joint TV | Added-middle clutter-mean error | Final clutter-mean error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early-middle | 512 | 0.03040 (0.04840), 1/6 | 0.23711 (0.25452), 0/6 | 0.16082 (0.18064), 0/6 | 0.01367 (0.04823), 2/6 | 0.03027 (0.05096), 2/6 |
| Early-middle | 2,048 | 0.01657 (0.02407), **1/6** | 0.12462 (0.13200), **1/6** | 0.08425 (0.09166), 1/6 | 0.01569 (0.02509), 1/6 | 0.01458 (0.02606), 2/6 |
| Middle | 512 | 0.02472 (0.04750), 1/6 | 0.21190 (0.23243), 0/6 | 0.14593 (0.17117), 0/6 | 0.01991 (0.04439), 0/6 | 0.01953 (0.05508), 1/6 |
| Middle | 2,048 | 0.00792 (0.02371), **1/6** | 0.10928 (0.12155), **0/6** | 0.07523 (0.08724), 0/6 | 0.00659 (0.02236), 0/6 | 0.02183 (0.02529), 2/6 |

For 2,048 particles, the maximum absolute **six-run mean signed 4D
cell** residual is 0.00294 against familywise q95 0.00379 in
early-middle, and 0.00266 against 0.00483 in middle. After collapsing
added-middle clutter, the corresponding maximum 3D cell residuals
are 0.00325 against q95 0.00472 and 0.00350 against 0.00564.
No 4D or collapsed 3D cell is familywise flagged in either case.
The full 4D TV range is 0.11949–0.13207 in early-middle and
0.10469–0.12048 in middle. Mean CPU seconds per run are 34.8/134.3
(512/2,048) and 37.9/136.4.

All three frozen flags—`persistent_mode_tail`,
`persistent_joint_tail` and `shared_cell_shift`—are **false**;
status is `middle_density_no_persistent_flag`. Both cases have 2/6
last-frame clutter-mean q95 exceedances at 2,048, but the flagged
errors are positive in both early-middle runs and have opposite signs
in the middle case. This secondary marginal warning is not a coherent
shared-cell shift and was not a move-selection criterion.

These two ten-detection cases add middle-frame ambiguity and retain
absolute exact references, yet remain far sparser than WP11k's
19-detection primary. They share nine detections; their paired flags
are descriptive and do not constitute independent replication or a
calibrated SMC test. No new move was selected, no 8,192-particle run
was made, and the untouched holdout remains closed. An alternative
absolute approximation for the 19-detection geometry now warrants an
explicit error budget, first validated against these exact middle-density
cases, before attributing a large-case discrepancy to a sampler move.
