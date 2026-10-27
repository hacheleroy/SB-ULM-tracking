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

The protocol and runner were committed at `503ae8eea2abfcb5abc9429e999a4502511a5668`
**before** any of the four sampler processes started. All four completed
within the 750 CPU-second cap. The final per-seed empirical joints and
errors are in `results/wp11y_independent_absolute_sampler.json`, byte
SHA-256 `3af1e36390c5700d150291e5493608009f319498f44f9ec4164d89654c795f50`.
The detailed stage diagnostics and repeated configs can be reproduced
with the fixed runner and seeds.

| Seed | CPU s | Stages | Full TV | Band × sixth TV | Max alive-band error | Sixth clutter-1 error | Last clutter-1 error | Replicated mode? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1500 | 231.3 | 19 | 0.11557 | 0.09512 | 0.03324 | +0.07071 | −0.01373 | No |
| 1501 | 225.5 | 19 | 0.10587 | 0.04646 | 0.02730 | −0.01376 | +0.06293 | No |
| 1502 | 234.8 | 20 | 0.09556 | 0.04388 | 0.01412 | +0.04337 | −0.02154 | Yes |
| 1503 | 228.2 | 19 | 0.16715 | 0.11931 | 0.11813 | +0.04972 | −0.00738 | No |

The median full TV `0.11072`, band × sixth TV `0.07079`, maximum
alive-band error `0.03027` and maximum cell error `0.01931` satisfy
their individual limits. **The frozen screen fails** because only one
of four seeds satisfies every replicated mode condition, short of the
required three. Seed 1500 exceeds the 0.06 sixth clutter-1 limit,
1501 exceeds the 0.06 final clutter-1 limit, and 1503 exceeds both
the band × sixth TV and alive-band error limits. The 1,000-draw exact-IID
95th percentiles are 0.07860 full TV and 0.02964 band × sixth TV;
all four SMC full TVs exceed the descriptive full-TV percentile. The
run log-evidence errors range from −0.702 to −0.415.

This is an independent absolute discrepancy on the selected synthetic
17-detection observation, with a marginal exact high-alive acceptance
margin. It does not establish the sampler's error on the 19-detection
WP11k case, select a new move or open the untouched holdout. A next
diagnostic can freeze stage-level attribution for the count and late
clutter errors on this balanced reference before proposing a new kernel.
