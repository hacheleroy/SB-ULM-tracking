# WP11r — Eight- and nine-detection exact density ladder

WP11q provided a validated four-slot/eight-frame reference at seven
detections, but its unchanged sampler had no persistent absolute flag.
This new protocol freezes two **nested, designed** denser observations
before evaluating either posterior. Adding detections at the final frame
makes the complete eight-frame trajectory relevant and lets a three-way
joint table expose errors that collapse away when final-frame clutter is
ignored. The cases share their first seven frames and are **not independent
draws** from a population.

## Frozen observations and model

Times are 0–7 ms in 1 ms steps. Both cases use frames 0–6 from WP11q:

| Frame | Detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | `(-0.014, 0.001)` |
| 2 | `(-0.007, 0.002)` |
| 3 | none |
| 4 | `(0.001, 0.000)` |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |

At frame 7, **eight** has `(0.018, -0.001)`; **nine** has
`(-0.011, 0.002)` and `(0.018, -0.001)`. Their cardinalities and
case-metadata SHA-256 values are:

| Case | Cardinalities | Detection SHA-256 | Canonical states | Labelled states |
| --- | --- | --- | ---: | ---: |
| Eight | `[1,1,1,0,1,2,1,1]` | `af1d17a353977b9caa06dcc09b937dab715931028692832379d2270ab0f982f3` | 272,486,255 | 5,245,666,773 |
| Nine | `[1,1,1,0,1,2,1,2]` | `2cb50f0b1c824dbee74e1b80e5220bc722deced9e6d9f2eec6a597e818b12edb` | 555,457,390 | 10,325,709,693 |

Use four exchangeable slots, localization standard deviation 0.005,
Matérn coordinate scales `(0.02,0.01)`, time length 0.003, detection
probability 0.55, ROI `[-0.04,0.04]` in each coordinate, and the default
birth/death prior. Vary expected clutter per frame over the same frozen
WP11q grid `[0.05,0.1,0.15,0.2,0.35,0.5,0.75,1,1.5,2.25,3]`.

Keep the WP11q coarse modes: **A** has at most 15 alive slot-frames and
at least one frame-six (index 5) clutter detection; **B** has at least 16
alive slot-frames and no frame-six clutter. Select a clutter rate for
each observation by maximizing `min(exact P(A), exact P(B))`, ties to the
lower rate. For **both** cases, exact selected A and B must each be at
least 0.10, and exact P(B) at beta 0.5 must be at most half beta-one
P(B), on `[0,.25,.5,.75,.9,1]`. If either fails, save both attempted
references and **stop all sampler inference**. No post-result revision
of observations, rates or modes belongs to this protocol.

## Exact reference and predeclared comparison

The exact sum uses nondecreasing four-slot interval-mask tuples and all
compatible assignments, weighted by the corresponding slot-permutation
multiplicity. Cache every single-slot GP detection pattern and verify
its factorized score against the direct collapsed likelihood on fixed
blocks. The inherited small-case test compares the weighted sum and
alive distribution against ordinary complete labelled enumeration.
Check the frozen canonical/labelled state counts, direct endpoint rate
reweighting and all eleven analytic HMM base normalizers for **each**
observation. The exact CPU cap is 4,000 seconds per case. Complete and
commit both independent exact references **before** a sampler run.

The primary exact distribution is the full alive total (0–32) by
frame-six clutter (0–2) by final-frame clutter table: **198 cells** for
eight and **297 cells** for nine. Save modes, the 2D collapsed joint,
marginals, beta curves, the eleven-rate grids and log evidences. For
each case and particle count, simulate 1,000 exact-posterior IID
multinomial draws (seeds 34361, 34367). Compare per-run 3D joint TV,
maximum mode and 3D cell errors to their own descriptive q95. Also
report 2D collapsed joint TV, alive-total TV and both clutter-mean
errors. Simulate 1,000 exact groups of six draws for a familywise q95
on the **maximum absolute six-run mean signed 3D cell** residual.

If both references pass, run the unchanged WP11g 10% four-path PG
mixture in separate one-thread processes: eight-case seeds 1160–1165
at 512, then 2,048 particles; nine-case seeds 1170–1175 at 512, then
2,048. Process CPU caps are 600 and 1,600 seconds. A passive observer
records A/B modes, resampling and accepted crossings without changing
the random stream.

The frozen **density-persistence** flags at 2,048 are:

1. `persistent_mode_tail` if both cases have at least **2/6** maximum
   mode errors above their exact IID q95.
2. `persistent_joint_tail` if both have at least **2/6** full 3D joint
   TVs above their exact IID q95.
3. `shared_cell_shift` if the **same** `(alive, frame-six clutter,
   last-frame clutter)` cell has a six-run mean signed error beyond the
   familywise q95 in both cases with the same sign.

These nested cases are dependent. The IID comparison is a descriptive
finite-sample yardstick for correlated SMC particles, not a calibrated
hypothesis test or move-selection gate. Neither outcome opens an 8,192
run or untouched holdout, and nine detections still cannot establish
calibration on WP11k's 19-detection case.

Runner: `src/sbulm/wp11r_eight_frame_density_ladder.py`. Checkpoint:
`results/wp11r_eight_frame_density_ladder.json`. The code and full rule
are committed before exact posterior evaluation or sampler inference.
