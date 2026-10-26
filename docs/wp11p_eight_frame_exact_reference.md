# WP11p — Exact four-slot/eight-frame late-mode bridge

WP11o's paired five-frame references did not reproduce a joint-cell error.
WP11k's four-slot/eight-frame primary has 19 detections and persistent
between-seed clutter differences but lacks an absolute reference. This
experiment freezes a four-slot/eight-frame case with **six detections** and
two competing frame-six alive/clutter modes. It uses an exact symmetry sum
over all 654,814,613 labelled states, implemented by scoring 33,074,574
canonical mask-assignment states. This is a bridge toward WP11k, not an
absolute reference for its 19-detection observation.

## Frozen observation and validation

Times are 0–7 ms in 1 ms steps; coordinates are in the model's units.

| Frame | Detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | `(-0.014, 0.001)` |
| 2 | `(-0.007, 0.002)` |
| 3 | none |
| 4 | none |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |
| 7 | none |

The cardinalities are `[1,1,1,0,0,2,1,0]` and the case-metadata
detection SHA-256 is
`6daec6150a3d24bcd13e222a5134ecc61e51454cd567d5874db69e4ad9fe61a9`.
Use four exchangeable slots, localization standard deviation 0.005,
Matérn coordinate scales `(0.02, 0.01)`, time length 0.003, detection
probability 0.55, ROI `[-0.04, 0.04]` in each coordinate, and the default
birth/death prior. Vary only expected clutter per frame over
`[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0]`.

Define mode **A** as total alive slot-frames at most 16 and at least one
clutter detection in frame six (index 5); **B** as at least 17 alive
slot-frames and no clutter at frame six. Other states form **other**.
For each rate, calculate exact P(A), P(B) and evidence; select the rate
maximizing `min(P(A), P(B))`, ties to the smaller rate. The frozen difficulty
check requires both selected masses at least 0.10 and exact P(B) at beta
0.75 at most half its beta-one mass, on beta grid `[0,.25,.5,.75,.9,1]`.
If either check fails, report an invalid bridge and **stop without sampler
runs**. Do not revise the observation, rate grid, or modes after looking
at the posterior.

## Exact reference and comparison rule

Enumerate nondecreasing tuples of four interval masks, with all compatible
frame assignments. A tuple with repeated interval masks contributes each
state `4! / product(factorial(mask multiplicity))` times: the labelled
model is exchangeable under all slot permutations. Verify the total
canonical and weighted labelled counts; compare this weighted sum with
ordinary full labelled enumeration on a small case. Check the count/prior
formula and direct Poisson-rate reweighting on deterministic blocks, and
check all eight weighted base normalizers against the independent analytic
HMM base normalizer. A failed check or the 10,000 CPU-second cap ends this
reference without changing the design. Save exact full **33×3** alive-total
by frame-six-clutter cells, mode masses, marginals, tempered modes and log
evidence before any sampler run.

For each 512 and 2,048 particle size, obtain 1,000 exact-posterior
multinomial IID draws (seed 34349). Compare each sampler run's maximum
three-mode absolute error, full joint TV, maximum absolute cell error,
alive-total TV, and clutter-mean error to that metric's descriptive IID
q95. Generate another 1,000 IID groups of six runs to obtain a
familywise q95 for the maximum absolute **six-run mean signed cell** error
across all 99 cells. Report full signed residuals and per-seed errors.

Only after the exact reference passes, run the unchanged WP11g 10% PG
four-path paired sampler, in separate one-thread processes: six independent
seeds 1100–1105 at 512 particles, then the same six seeds at 2,048.
Process CPU caps are 600 and 1,600 seconds. A passive observer records
mode weights, resampling, and accepted A↔B crossings without changing RNG.

The predeclared **diagnostic** flags at 2,048 particles are:

1. `mode_error_tail`: at least 3/6 maximum mode errors exceed the exact
   IID per-run q95.
2. `joint_tail`: at least 3/6 full-joint TVs exceed the exact IID q95.
3. `cell_shift`: the largest six-run mean signed cell residual exceeds the
   familywise q95, and a flagged cell has the same error sign in the ordered
   first and last three seed groups.

Multiple descriptive flags and a single designed observation do not form
a calibrated sampler-move gate. Neither outcome selects a move or opens an
8,192-particle run or untouched holdout. Crucially, six detections are much
less than WP11k's 19; a valid bridge preserves the eight-frame temporal and
four-slot mode geometry but cannot certify that original case.

Runner: `src/sbulm/wp11p_eight_frame_exact_reference.py`. Checkpoint:
`results/wp11p_eight_frame_exact_reference.json`. This protocol and runner
are committed before exact posterior evaluation or sampler inference.

## Frozen outcome

The weighted exact pass scored **33,074,574** canonical states representing
**654,814,613** labelled states in 1,474.7 CPU seconds. The count and all
eight analytic base-normalizer checks passed, as did the direct endpoint
clutter-rate checks. The frozen balance rule chose rate **0.2**. Its exact
P(A), P(B), and other masses were **0.19729, 0.05631, and 0.74639**.
P(B) at beta 0.75 was **0.02455**, below half its beta-one mass; the
predeclared **P(B) ≥ 0.10** check failed. The full rate grid, 99-cell
distribution, tempered curve, evidence and IID baselines are saved in the
checkpoint. This geometry is an exact eight-frame reference but **does not
retain sufficiently balanced late modes** under its frozen definitions.
Status: `frozen_eight_frame_bridge_invalid_stop`. No sampler was run, and
the observation or threshold was not revised after seeing this result.
