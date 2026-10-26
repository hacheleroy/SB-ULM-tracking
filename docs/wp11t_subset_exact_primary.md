# WP11t — Subset-convolution absolute reference for the exposed 19-detection case

WP11s supplied independently checked exact four-slot/eight-frame joint
posteriors at ten detections. The next reference uses a different exact
combinatorial calculation on the **already exposed** WP11k primary with 19
detections. This avoids extrapolating a ten-detection Monte Carlo error
estimate to 19 detections. A feasibility prototype established that a
subset calculation runs on this geometry and gave provisional evidence
and mode weights before this protocol was committed. Therefore the
primary exact posterior itself is **not blinded**. The validation
thresholds, archival comparisons and **new sampler seeds** below are
frozen before the formal checked reference and fresh sampler runs.

## Fixed model and calculation

Use WP11k's unchanged `wp11k_primary` regime (trajectory seed 79,
corruption seed 83, detection probability 0.68 and clutter rate 0.28).
Its eight-frame cardinalities are `[2,2,4,4,3,2,1,1]`, and its
case-metadata detection SHA-256 is
`e7fc37d377e5b956377af6a64e41837f15e42d43e61a932af48400da9bffd922`.
The reference is not the untouched WP8 holdout.

For one slot, enumerate each contiguous existence interval and every
assigned-detection subset containing at most one point per frame.
Compute its prior, detection/miss factor and analytically collapsed
single-slot Matérn likelihood. Divide by clutter intensity for each
assigned detection, and group the resulting weights by detection subset
and alive interval length. There are at most (2^{19}) subset indices;
only the valid one-slot patterns have a nonzero coefficient.

Multiply the one-slot polynomial four times in the **squarefree** algebra:
products sharing a detection vanish. Ranked subset zeta transforms,
polynomial multiplication by subset cardinality and Möbius inversion
compute the four-slot disjoint-assignment coefficients. The unused
points receive their exact Poisson clutter factor. A 33-point Fourier
inversion of an alive-length marker recovers alive totals 0–32. For
any assigned union, its frame-six and final-frame clutter counts are
known directly. The primary absolute joint has **33 × 3 × 2 = 198 cells**.
This is exact in real arithmetic; numerical validation remains necessary.

## Frozen numerical error budget and stop rule

First reproduce **both** WP11s ten-detection exact references at their
already selected rate 1.5. Require absolute log-evidence error at most
`1e-9`, full 4D joint TV at most `1e-9` and maximum absolute cell error
at most `1e-10` in each case. Verify the primary one-slot GP lookup
against direct collapsed scores on 32 fixed base draws.

Next compute the primary 198-cell joint twice in extended precision:
(1) a direct fourth power in ranked-zeta space and (2) two separate
pairwise subset convolutions. Require absolute log-evidence difference
at most `1e-8`, normalized joint TV at most `1e-7`, maximum absolute
cell difference at most `1e-8`, minimum raw cell probability at least
`-1e-10`, maximum relative imaginary Fourier residual at most `1e-8`,
and absolute raw total-mass error at most `1e-8`. Clip any smaller
negative roundoff cells and renormalize; record the resulting TV
correction. These tolerances are an **operational numerical validation
envelope**, not an interval-arithmetic proof of a strict error bound.
The process CPU cap is 1,200 seconds. If any check fails, checkpoint the
failure and **stop before fresh sampler inference**.

## Frozen absolute comparison

Report exact alive histogram, three bands (low ≤20 alive slot-frames,
mid =21, high ≥22) by frame-six clutter, full 198-cell joint, last-frame
clutter, and log evidence. Retrospectively compare archived WP11k seeds
800–803 at 512 and 2,048 using their saved alive histograms and final
band-by-clutter tables; these runs were seen before this reference and
cannot serve as a fresh selection gate.

For descriptive finite-sample scale, simulate 1,000 exact multinomial
IID draws at 512 and 2,048 particles with seed 34531. Compare full 3D
joint TV, band-by-frame-six-clutter TV, maximum band-mass error,
alive-total TV, maximum 3D cell error and frame-six/final clutter-mean
errors to their case-specific q95. Draw 1,000 exact groups of six at
2,048 for a familywise q95 on the maximum absolute mean signed 3D cell.

Only after the validated exact checkpoint is committed, run the
**unchanged WP11g 10% four-path PG mixture** at 2,048 particles with
six new seeds `1200..1205`, in separate single-thread processes with
750 CPU-second caps. A passive observer records the inherited WP11k
band trace without changing the RNG stream. Retain each final 198-cell
table and evidence estimate. At the last seed, report three frozen
**descriptive** flags: mode tail and full-joint tail if at least 2/6
values exceed their respective exact-IID q95; signed-cell shift if a
six-run mean 3D cell exceeds the familywise q95 and has the same sign
in both ordered three-seed halves.

Correlated SMC particles do not follow the IID multinomial law, and the
six seeds do not create a calibrated hypothesis test. Even a flagged
result would diagnose an absolute discrepancy, **not** select a move.
No 8,192-particle run or untouched holdout belongs to this step.

Runner: `src/sbulm/wp11t_subset_exact_primary.py`. Result checkpoint:
`results/wp11t_subset_exact_primary.json`. The WP11s and WP11k source
checkpoints are pinned by SHA-256 in the runner.

## Checked exact result

The formal computation passed every frozen numerical check. Independent
WP11s validations reproduced the early-middle and middle exact log
evidences to `0` and `7.1e-15`, with full 4D joint TVs `7.0e-16` and
`5.7e-16`. The primary one-slot lookup agreed with direct collapsed GP
scores. Direct fourth power and paired subset convolutions gave primary
normalized joint TV `9.82e-12`, maximum cell difference `8.83e-13`,
and log-evidence difference `5.23e-12`. The largest roundoff clipping
correction was TV `1.67e-12`; the most negative raw cell was
`-4.44e-13`. Both computations finished under the 1,200 CPU-second
reference cap (170.1 and 233.9 CPU seconds). These are cross-checks,
not a formal interval proof, but comfortably meet the frozen envelope.
The validated reference was committed **before** the six fresh WP11g
sampler runs.

| Absolute quantity | Exact value |
| --- | ---: |
| Log evidence | 105.2099672551 |
| Low alive band (≤20) | 0.411160 |
| Mid alive band (=21) | 0.139522 |
| High alive band (≥22) | 0.449318 |
| Frame-six clutter 0 / 1 / 2 | 0.055831 / 0.496129 / 0.448040 |
| Final-frame clutter 0 / 1 | 0.345129 / 0.654871 |

The full 198-cell joint, alive histogram, exact IID baselines and
numerical diagnostics are saved in the checkpoint.

## Retrospective and fresh absolute comparison

The archived WP11k seeds 800–803 were previously compared only with
one another. Against the new exact reference, **4/4** at 512 and
**4/4** at 2,048 exceed their exact-IID q95 for maximum band-mass
error and alive-total TV. At 2,048, the four-seed median band-by-clutter
TV is 0.19770, median maximum band error 0.16843, and median
alive-total TV 0.17776. These are retrospective diagnostics; the archive
does not contain the full final 198-cell tables.

All six **new**, unchanged WP11g runs completed within their 750
CPU-second caps (mean 198.7 CPU seconds). The table reports six-seed
median absolute error, exact-IID q95 in parentheses and exceedances.

| Diagnostic at 2,048 particles | Median (IID q95) | Above q95 |
| --- | ---: | ---: |
| Full alive × frame-six clutter × last clutter TV | 0.21032 (0.07099) | **6/6** |
| Low/mid/high × frame-six clutter TV | 0.17862 (0.03303) | **6/6** |
| Maximum low/mid/high mass error | 0.17582 (0.02383) | **6/6** |
| Alive-total TV | 0.17814 (0.04185) | **6/6** |
| Maximum 3D cell error | 0.03640 (0.01491) | **6/6** |
| Frame-six clutter-mean error | 0.09302 (0.02604) | **6/6** |
| Final clutter-mean error | 0.05908 (0.02157) | **5/6** |

Full 3D TVs range from 0.10108 to 0.30577. The exact low/high band
masses are 0.411/0.449, whereas the fresh runs range from 0.380–0.709
low and 0.182–0.465 high. Every fresh log-evidence estimate is below
the exact value by 1.155–2.970 nats. The largest absolute six-run mean
signed 3D cell residual is **0.02597**, versus the frozen familywise
exact-IID q95 **0.00621**, at alive total 17, frame-six clutter 2,
final clutter 1. Seventeen cells exceed that familywise threshold;
sixteen have the same sign in both ordered three-seed halves. The
corresponding per-seed signed tables and passive mode traces are saved.

The frozen `mode_tail`, `joint_tail` and `cell_shift` diagnostic flags
are all **true**; status is `absolute_primary_diagnostic_signal`.
This establishes a substantial absolute discrepancy on this exposed
19-detection synthetic case under the independently checked model,
including a recurrent count/clutter shift. The numerical validation
error is many orders smaller than the observed sampler errors. The
exact-IID q95 remains a descriptive yardstick for correlated SMC
particles, so the exceedance counts are not a calibrated frequentist
p-value. This single observation does not establish how often the
failure occurs across acquisitions or which sampler transition causes
it. No new move, 8,192 run or untouched holdout was selected.

The next proposal should target late count/clutter mode movement, freeze
its cost and absolute-error criteria against this exposed exact case
before new inference, and retain independent replication and the
untouched holdout for later validation.
