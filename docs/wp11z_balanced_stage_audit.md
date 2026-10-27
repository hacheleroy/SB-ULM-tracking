# WP11z — Frozen stage attribution on the balanced decoy observation

WP11y failed its frozen absolute replication screen on the selected
17-detection WP11x case: only one of four seeds met all mode limits.
This audit asks **when** alive-count and frame-six/final-frame clutter
discrepancies appear and whether weighting, resampling or rejuvenation
first exposes them. The rule below is committed before any WP11z sampler
replay or stage-level exact calculation. It is a diagnostic on four
**already exposed** seeds, not a new candidate or a selection gate.

## Inputs and replay

Pin the WP11y comparison by byte SHA-256
`3af1e36390c5700d150291e5493608009f319498f44f9ec4164d89654c795f50`
and require its failed status. Pin the WP11x exact reference by SHA-256
`d1834acda7022ba9d50d97a15115975909c6c938409969b8415e913dda62977c`,
the WP11x selected detection SHA-256
`dd3d3fbb55fd601298ee410539767df24950e2f886d814c30b98f54e89bc0c29`,
and cardinalities `[1,2,2,2,2,3,3,2]`. Its exact joint is alive total
× frame-six clutter × final clutter, with shape 33 × 4 × 3.

Replay **all** WP11y seeds `1500..1503`, unchanged WP11g 10% four-path PG
controls and 2,048 particles, one BLAS/OMP thread, each in a separate
process capped at 750 CPU seconds. A passive observer consumes no RNG and
records, at every adaptive stage, prior unweighted, likelihood weighted,
resampled and post-rejuvenation joints. It also records beta, ESS, unique
parents and states. Before any stage exact calculations, require each
replay's final empirical joint, number of stages and log evidence to
match its WP11y checkpoint. A mismatch or cap stops the audit; no seed
is replaced. Existing WP11y results are used only for this replay check,
not for adjusting the onset rule.

## Exact stage targets and fixed comparison

After the four replays, calculate the direct squarefree four-slot exact
joint at beta zero and **every observed adaptive beta** in each seed.
Use a separate one-thread process per seed with a 1,000 CPU-second cap.
Every exact beta must have minimum raw cell ≥−1e-10, relative imaginary
residual ≤1e-8 and raw mass error ≤1e-8. At beta zero and one, compare
the recomputed joint and log partition to the pinned WP11x checkpoint:
TV ≤1e-7 and absolute log-partition difference ≤1e-8. A numerical
failure or cap stops the audit, without interpolating an uncomputed beta.

At stage t, compare the **prior unweighted** joint to the exact target
at the previous beta. Compare the weighted, resampled and rejuvenated
joints to the exact target at the stage's beta. Record full 396-cell TV,
maximum absolute low (≤20), middle (21), high (≥22) alive-band mass
error, absolute and signed clutter-count-one mass error for frame six
and the final frame, and band × sixth and band × final clutter TV.
Signed full-joint TV changes across operations and diversity are
descriptive; weighting also changes the target.

For each observed beta and metric, take 1,000 exact multinomial
histograms of 2,048 particles, using seed `56789 + 128 × sampler_seed
+ stage_number`, and compute its 95th percentile. The descriptive
marker is `max(floor, 2 × q95)` with these **predeclared** floors:

| Metric | Floor |
| --- | ---: |
| Maximum alive-band mass error | 0.06 |
| Absolute sixth clutter-one mass error | 0.06 |
| Absolute final clutter-one mass error | 0.06 |
| Band × sixth clutter TV | 0.08 |
| Band × final clutter TV | 0.08 |

For each seed and metric, mark the first pair of **consecutive
post-rejuvenation stages** above its stage-specific limit; report the
first stage's beta and the earliest operation that crossed its limit.
If the prior joint already exceeds the previous-beta limit, mark it
`carried_from_previous_stage`. Also report peak post-rejuvenation errors
and weighted exceedances cleared by rejuvenation when no sustained
onset is found. An IID q95 is a scale, not a calibrated SMC significance
test. The observed adaptive betas and correlated particles prevent that
interpretation. This audit cannot license a new sampler move by itself.

Runner: `python -m sbulm.wp11z_balanced_stage_audit --case sampler --seed
1500` and the other fixed seeds, then `--case exact --seed` for each
and finally `--case aggregate`. Detailed replay and exact checkpoints
remain locally reproducible in `results/wp11z_runs/` and
`results/wp11z_exact/`; the concise scored stage summary goes to
`results/wp11z_balanced_stage_summary.json`. The untouched holdout,
high-particle run and earlier 19-detection case remain closed.

## Outcome

Pending frozen replay and exact stage calculation.
