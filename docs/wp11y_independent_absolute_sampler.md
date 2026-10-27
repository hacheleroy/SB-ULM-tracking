# WP11y — Frozen independent absolute sampler comparison

WP11x selected and independently checked a synthetic four-slot,
eight-frame, **17-detection** reference. Its beta-one alive low/middle/high
masses are 0.71098/0.10885/0.18017, so the high band's acceptance margin
is only 0.00017. This comparison is frozen before **any WP11y sampler
inference**. It tests the unchanged sampler's final distribution against
that reference; it neither revisits candidate tuning nor claims calibration
on the earlier exposed 19-detection WP11k primary.

## Fixed input and run schedule

Require the byte SHA-256
`d1834acda7022ba9d50d97a15115975909c6c938409969b8415e913dda62977c`
of `results/wp11x_frame_six_exact_reference.json`, its validated status,
the pinned WP11x geometry ledger, selected variant 1, detection SHA-256
`dd3d3fbb55fd601298ee410539767df24950e2f886d814c30b98f54e89bc0c29`,
and cardinalities `[1,2,2,2,2,3,3,2]`. The selected exact beta-one joint
is 33 alive totals × 4 frame-six clutter counts × 3 final clutter counts.
Check its raw-cell, imaginary-residual and mass numerical envelope before
sampling. Use truth labels only to reproduce the WP11x synthetic geometry;
the sampler receives detections and model parameters, not labels.

Run the **unchanged WP11g 10% four-path PG paired mixture** at 2,048 outer
particles, with seeds `1500,1501,1502,1503`. Each seed is a separate
one-thread process; set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to `1` and cap at 750 process CPU
seconds. Run all four seeds regardless of early results. A seed that fails
or exceeds the cap ends this comparison as incomplete; do not replace it.
No additional sweeps, alternative kernel, 8,192-particle run or untouched
WP8 holdout belongs to this step.

## Predeclared comparison and replication rule

At beta one, histogram the final equal-weight particles into the same
alive total × sixth clutter × final clutter cells. Report each empirical
joint, the full-joint total variation (TV), the three alive-band errors
(low ≤20, middle 21, high ≥22), band × sixth-clutter TV, maximum absolute
alive-band mass error, maximum single-cell error, all sixth and final
clutter marginal errors, their signed mean errors, log-evidence error,
runtime and number of tempering stages. Compare **absolute** masses with
the frozen reference; a pairwise agreement of sampler runs is insufficient.

All of the following fixed limits must hold for the descriptive
17-detection baseline adequacy screen:

| Criterion | Limit |
| --- | ---: |
| Median full 396-cell joint TV | ≤0.12 |
| Median band × sixth-clutter TV | ≤0.10 |
| Median maximum absolute alive-band mass error | ≤0.08 |
| Median maximum single-cell error | ≤0.04 |
| Replicated mode condition | At least 3 of 4 seeds |

For a seed to meet the replicated mode condition, its band × sixth TV
must be ≤0.10, its maximum alive-band mass error ≤0.08, and its absolute
mass error for **clutter count one** in both frame six and the final frame
must each be ≤0.06. The first three median limits reuse WP11u's frozen
absolute targets. The single-cell and specific clutter-count limits make
the joint-cell and decoy comparison explicit, with margins above simple
2,048-draw multinomial fluctuation. These are practical development
limits, not a formal calibration test.

For scale only, draw 1,000 independent multinomial histograms of 2,048
from the exact joint with RNG seed `45671`; report the 95th percentile
of the corresponding errors. Correlated SMC particles do not have the
IID distribution, so these percentiles cannot override a frozen limit.
If any criterion fails, stop and report it without changing seeds,
thresholds or the observation. A passing screen supports this one
17-detection observation only; the WP11k 19-detection failure and
untouched holdout remain unresolved.

Runner: `python -m sbulm.wp11y_independent_absolute_sampler --case sampler --seed 1500`
(and each other fixed seed); then `--case aggregate`. Individual run
checkpoints go to `results/wp11y_runs/`; the comparison goes to
`results/wp11y_independent_absolute_sampler.json`.

## Outcome

Pending sampler inference. The protocol and runner are committed first.
