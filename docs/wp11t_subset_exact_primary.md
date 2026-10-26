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
