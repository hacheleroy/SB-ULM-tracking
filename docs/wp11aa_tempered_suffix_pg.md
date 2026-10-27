# WP11aa — Frozen tempered suffix PG development comparison

WP11z found sustained count and late-clutter discrepancies beginning
near β=0.83–0.89 in three of four 17-detection replays. WP11u's
frame-six-through-eight pair suffix particle-Gibbs (PG) update was
tested only at β=1 on the exposed 19-detection case. This is **one**
new schedule for that existing target-preserving kernel. The design,
two exposed-case comparisons and stop rule are committed before any
WP11aa candidate inference. Both observations and their previous
control results are already known; a pass is development evidence,
not independent calibration.

## Fixed move and target check

Keep the unchanged WP11g 10% four-path pair PG and the same local pair
move, outer particle count (2,048), adaptive tempering and sweeps.
At each rejuvenation with β≥0.75, replace 40% of **paired proposals**
by the four-particle conditional pair suffix PG of WP11u, retaining
50% local pair and 10% full-path PG. The suffix conditions on frames
one through five and redraws frames six through eight. Before β=0.75,
the candidate is bit-for-bit identical to control for a matched seed.
The switch depends only on temperature. In particular, the final stage
receives the control's ordinary eight plus eight sweeps; it does not
inherit WP11u's additional final 24 sweeps.

The fixed-prefix conditional SMC kernel uses the exact birth/death,
finite-set count and tempered collapsed GP factors. As a Gibbs update
conditional on other slots and the prefix, it preserves each tempered
target. Extend WP11u's exhaustive 165-state two-slot transition check
to β=0.75, alongside 0, 0.37 and 1; require stochastic rows and
stationarity at absolute 1e-12 and relative 1e-11. Test identical
control/candidate transitions below the switch and nonzero suffix
selection above it. A check failure stops all inference.

## Frozen inputs and runs

The 19-detection primary exact reference is
`results/wp11t_subset_exact_primary.json`, byte SHA-256
`52f13725b9f7dd8851b44ee97025afebf98f042791b111cf508dcbd5c0c0c14c`.
Reuse the six unchanged 2,048-particle WP11u control arms at seeds
`1300..1305`, pinned by byte SHA-256
`cc3a478913938537e04a498e87c65506ae0da42eb1fcfe90ce7feb380aee9962`.
The 17-detection decoy exact reference is
`results/wp11x_frame_six_exact_reference.json`, byte SHA-256
`d1834acda7022ba9d50d97a15115975909c6c938409969b8415e913dda62977c`.
Reuse WP11y's four unchanged controls at seeds `1500..1503`, byte
SHA-256 `3af1e36390c5700d150291e5493608009f319498f44f9ec4164d89654c795f50`.
The stage audit used to place the switch is pinned at SHA-256
`805ba60c16bdb768d1236cba6d6cb48491639555874b79033bb4eba987640ab9`.

Run **all ten candidate seeds**, paired against their saved controls,
regardless of an early result. Run each in a separate process with all
four BLAS/OMP thread variables set to one and a 750 process-CPU-second
cap. A cap or reference/oracle failure stops the comparison without
replacing a seed. Archive the final alive total × frame-six clutter ×
last clutter joint, signed band/clutter errors, per-stage selection and
movement rates, log evidence, runtime and costs. The primary is a
33 × 3 × 2 joint and the decoy a 33 × 4 × 3 joint. Compute exact-IID
scales only for description from the existing checkpoints; they do not
override the gates.

## Predeclared two-case gate

The **primary 19-detection** gate retains WP11u's absolute and relative
limits. Every condition must hold: each candidate/control CPU ratio
≤1.50; at least four of six paired full-joint TV wins; candidate
medians ≤0.12 full-joint TV, ≤0.10 alive-band × frame-six TV and
≤0.08 maximum absolute alive-band mass error; candidate/control median
ratio ≤0.80 on each of those three; and no candidate median regression
>0.020 in alive-total TV, maximum cell error, absolute sixth-clutter
mean error or absolute last-clutter mean error.

The **decoy 17-detection** gate requires every candidate/control CPU
ratio ≤1.50, at least three of four paired full-joint TV wins, candidate
median full-joint TV ≤0.12 and ≤0.90 of the control median, median
band × sixth TV ≤0.10, median maximum alive-band mass error ≤0.08,
and median maximum cell error ≤0.04. At least three of four seeds must
simultaneously have band × sixth TV ≤0.10, maximum alive-band error
≤0.08 and absolute clutter-count-one mass errors ≤0.06 at **both**
frame six and the final frame. No median regression >0.020 in the same
four secondary metrics used for the primary. These retain WP11y's
replication guardrail and add a paired full-joint improvement rule.

**Both** case gates must pass. A failure stops this candidate. No
post-result change to the beta switch, suffix fraction, thresholds,
seeds or observation can rescue it. Even a pass on these two exposed
development cases requires a new independently selected observation
and frozen replication rule before a calibration or holdout claim.
There is no 8,192-particle or untouched WP8 holdout run here.

Runner: `python -m sbulm.wp11aa_tempered_suffix_pg --case run
--observation primary_19 --seed 1300` and each other prespecified
case/seed; then `--case aggregate`. The candidate checkpoints go to
`results/wp11aa_runs/`; the concise comparison goes to
`results/wp11aa_tempered_suffix_pg.json`.

## Outcome

All ten prespecified candidate seeds completed below the 750-second
process CPU cap. Their final joints and paired, absolute comparison
are archived in `results/wp11aa_tempered_suffix_pg.json` (SHA-256
`620b5db1603af7626933000b70b38ba40b5239805ccf1a9ea247e5d55292a1ba`).
The frozen combined gate **failed**; the suffix schedule is not selected.

| Case | Paired full-joint TV wins | Control → candidate median full-joint TV | Candidate median band × sixth TV | Candidate median maximum alive-band error | CPU ratios above 1.50 | Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 19 detections | 5/6 | 0.19676 → 0.15087 | 0.13639 | 0.07983 | seed 1301: 1.646 | fail |
| 17 detections | 4/4 | 0.11072 → 0.08959 | 0.04296 | 0.01418 | seed 1502: 1.526 | fail |

The primary misses the absolute median full-joint TV ≤0.12 and
band × sixth TV ≤0.10, as well as the latter's relative ≤0.80 rule:
0.13639 / 0.16060 = 0.849. Its paired-win and secondary-regression
conditions pass. The decoy meets its median accuracy, paired-win and
secondary-regression conditions, but only seeds 1501 and 1502 satisfy
all four per-run mode limits; three were required. Seed 1500 has a
frame-six clutter-one signed error +0.06339 against ±0.06. Seed 1503
has maximum alive-band error 0.08395 against 0.08. The suffix was
selected in about 39–41% of paired proposals at active temperatures,
with first active β between 0.753 and 0.786, as specified.

For scale only, all six primary full-joint TVs exceed the existing
2,048-sample exact-IID q95 of 0.07099. Three of four decoy runs exceed
its existing 1,000-replicate, 2,048-sample exact-IID q95 of 0.07860;
seed 1502 is 0.07309.
Correlated SMC outputs do not make those IID percentiles significance
tests or substitutes for the frozen gate.

The result shows useful movement but insufficient absolute reliability
and cost on these exposed cases. Do not tune this beta switch, fraction
or gate to rescue the comparison. A subsequent proposal should freeze
one cheaper, more selective count/late-clutter move and its replication
and cost rules independently before another test. The 19-detection
case remains uncalibrated, and the untouched holdout remains closed.
