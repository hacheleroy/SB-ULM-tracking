# SB–ULM tracking

This repository is the reproducible starting point for studying **posterior
sampling of uncertain microbubble trajectories in ultrasound localization
microscopy (ULM)**.

The first scientific question is deliberately narrow:

> When two bubbles cross, can a posterior over data associations preserve the
> competing coherent trajectory hypotheses, instead of committing early to one
> deterministic assignment?

The baseline benchmark operates on noisy localizations with a fixed number of
bubbles and detections. This isolates the central association ambiguity from
missed detections, false positives, births/deaths, and acoustic-model errors.
WP3 now reintroduces the first two while deliberately retaining persistent
latent bubbles. WP4 adds births and deaths through explicit exchangeable slots
with a contiguous existence mask while preserving an exact small-case oracle.
WP5 replaces interval-product initialization by an exact sequential HMM base
sampler and calibrates a blocked slot proposal through three slots and six
frames. A likelihood-informed paired-slot move improves four-slot cross-run
agreement. A subsequent exact oracle also passes for a sparse four-slot,
four-frame case. Six independent four-slot/eight-frame runs fail the
stability gate at 4,096 particles but pass all seven cross-run checks at
8,192. This is a reproducible result on one synthetic case, not an exact
eight-frame calibration claim.

## Current contents

- A common representation for trajectories and unordered detections
- A controlled two-bubble crossing simulator
- A separation-by-noise difficulty map identifying intrinsically ambiguous regimes
- Position-only and constant-velocity Hungarian baselines
- An exactly enumerable association posterior for small fixed-cardinality
  problems, scored with a linear-Gaussian constant-velocity model
- A velocity-calibrated Matérn-3/2 target in physical ULM units, with a full
  7-by-7 separation/noise regime map and speed-scale sensitivity analysis
- An exact Gaussian-localization posterior using the same frozen physical
  prior, plus an order-invariant differentiable set potential and analytical
  gradient used by the TSBM integration
- A tested adapter to the public TSBM discrete-potential interface, an exact
  matched Brownian/Sinkhorn reference target, and discrete plus continuous
  sampler-calibration metrics
- A collapsed annealed-SMC sampler for the original Matérn localization
  posterior, with adaptive ESS tempering, association rejuvenation, exact
  conditional path draws, and a complete exact-target calibration gate
- An explicit null-association extension for missed detections and uniform
  Poisson clutter, with reversible partial-injection moves, exact small-case
  enumeration, and a separate WP3 calibration gate
- An explicit birth/death extension with three association states (absent,
  alive-but-missed, and assigned), a birth--survival--death prior, reversible
  interval and slot-permutation moves, and a separate WP4 exact-target gate
- An exact forward-filtering/backward-sampling base initializer and a
  conditional-base blocked slot move, with a WP5 exact gate beyond two slots
  and a documented four-slot stability boundary
- A joint two-slot, one-/two-frame multiple-try proposal that scores candidate
  births, misses, clutter, and associations with the exact tempered posterior,
  with a separate exact-oracle and four-slot stability comparison
- A fixed-sampler convergence audit with a sparse four-slot exact oracle,
  higher-particle eight-frame replicates, and label-invariant count diagnostics
- Metrics for identity accuracy, identity switches, posterior mass, entropy,
  and credible-set coverage
- A tested exporter to the public Smooth-SB notebook format and distributional
  metrics for comparing its sampled index paths with the exact posterior
- A flexible reader for compressed ULM track JSON files, including the common
  layouts expected from `tracks.json.xz` archives
- Unit tests and a reproducible demonstration experiment

The enumerable posterior is a **gold-standard small-problem benchmark**, not
the final Schrödinger-bridge method. It gives exact mode weights under a fully
specified model and therefore provides a target against which approximate
samplers can be calibrated. The package now exposes both the original
localization-noise/constant-velocity posterior and the discrete Matérn-GP target
corresponding to the public Smooth-SB notebook. Keeping these targets separate
prevents a difference in priors from being misdiagnosed as sampler error.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m sbulm.experiment --output-dir results
python -m sbulm.difficulty --output-dir results --workers 4
python -m sbulm.smooth_sb --output results/sbulm_crossing_smooth_sb.npz
python -m sbulm.physical_sweep --output-dir results --workers 6
python -m sbulm.wp2_likelihood --output-dir results --n-seeds 64 --workers 6
python -m sbulm.smc_calibration --output-dir results --workers 6
python -m sbulm.wp3_null_calibration --output-dir results --workers 6
python -m sbulm.wp4_existence_calibration --output-dir results --workers 6
python -m sbulm.wp5_scaling_calibration --output-dir results
python -m sbulm.wp6_paired_scaling --output-dir results
python -m sbulm.wp7_convergence_audit --bridge-only
python -m sbulm.wp7_convergence_audit --large-only --large-particles 4096
```

Public-TSBM experiments use `experiments/run_tsbm_wp2.py` with an external
checkout pinned as described in `docs/tsbm_wp2_integration.md`.

The experiment writes:

- `results/crossing_summary.json`: aggregate and illustrative metrics
- `results/crossing_example.png`: truth, deterministic tracks, and leading
  posterior hypotheses
- `results/difficulty_map.csv/.json/.png`: quantitative regime map over central
  bubble separation and localization noise
- `results/sbulm_crossing_smooth_sb.npz`: a `(T, N, D)` crossing input whose
  `arr_0` key can be loaded by the upstream one-by-one-matching notebook
- `results/smooth_sb_smoke_summary.json`: first external `Mi=32` validation,
  including target alignment, five-seed sample diagnostics, and runtime
- `results/physical_matern_calibration.json` and
  `results/physical_matern_calibration_map.csv/.png`: physical 1 kHz regime map
  comparing the upstream-default and velocity-calibrated Matérn targets
- `results/physical_matern_speed_sensitivity.csv/.png`: ULMShare-based
  5th/median/95th-percentile speed-scale sensitivity
- `results/wp2_likelihood_map.csv/.png` and
  `results/wp2_credible_coverage.csv/.png`: paired hard-observation versus
  Gaussian-likelihood results on the frozen physical map
- `results/wp2_localization_likelihood.json`: complete WP2 configuration and
  aggregate metrics
- `results/tsbm_wp2_calibration.json/.png`: consolidated public-TSBM gate,
  including finite-sample references and forward/backward checkpoint comparisons
- `results/smc_matern_calibration.json/.png` and
  `results/smc_matern_calibration_map.csv/.png`: Option-A SMC gate on three
  frozen regimes and 196 separation/noise/data-seed cases
- `results/wp3_null_calibration.json/.png` and
  `results/wp3_null_calibration_map.csv/.png`: exact-target gate and
  detection-probability/clutter-rate map for explicit null associations
- `results/wp4_existence_calibration.json/.png`: four-regime exact-target gate
  for explicit persistent, birth, death, and turnover slot configurations
- `results/wp4_evidence_escalation.json`: retained 16,384-particle diagnostic
  for the single-run evidence failure
- `results/wp5_scaling_calibration.json/.png`: exact three-slot/five- and
  six-frame calibration plus the four-slot/eight-frame stability stress test
- `results/wp6_paired_scaling.json/.png`: frozen paired-versus-blocked
  exact-oracle and four-slot stability comparison
- `results/wp6_paired_heldout.json`: an untuned second triplet of four-slot
  independent runs
- `results/wp7_convergence_audit.json`: exact four-slot bridge and
  independent eight-frame particle-scaling diagnostics

The NPZ stores hidden identities and latent trajectories under separate keys
for auditing, but the upstream method receives only the unordered observations
in `arr_0`. See `docs/smooth_sb_integration.md` for the pinned upstream version,
environment constraints, and comparison protocol.

The physical calibration exposed the previous method boundary: tuning the
motion prior is not enough when noisy localizations are imposed as exact
intermediate marginals. WP2 now shows that an explicit Gaussian localization
likelihood removes this severe overconfidence on the fixed-cardinality map.
At 95% nominal mass, reference-path inclusion rises from 35.52% to 99.01%
across 3,136 paired cases. See `docs/physical_calibration.md` and
`docs/wp2_localization_likelihood.md`.

The subsequent public-TSBM integration is technically functional but does not
pass its calibration gate. It finds the important ambiguous association modes
without recovering their probabilities, and it learns the wrong endpoint
coupling in the easy crossing. It is retained as a negative baseline. See
`docs/tsbm_wp2_integration.md`.

Option A now supplies the calibrated alternative. Collapsed annealed SMC
passes all nine primary discrete/continuous/evidence gates. Across the complete
196-case map, TV, JS, and finite-sample-calibrated credible-set recall pass in
every case, and the minimum exact posterior mass covered is 98.80%. This
validates the fixed-cardinality localization posterior, but not yet a
variable-cardinality or IQ/RF model. See `docs/smc_matern_calibration.md`.

WP3 now adds the first controlled departure from fixed observation
cardinality. A persistent latent bubble can be assigned to a detection or to
the explicit null state `-1`; any unassigned detection is explained by a
uniform Poisson clutter process. The four-regime exact-target gate passes in
12/12 runs. On the 36-case stress map, TV, JS, null/clutter marginals, evidence,
and exact-mass coverage pass in every case; one finite-sample run misses one
low-mass member of the exact 95% credible set, so the aggregate all-check count
is 35/36. This validates missed detections and false positives for two
persistent bubbles, not births/deaths or unknown cardinality. See
`docs/wp3_null_association.md`.

WP4 distinguishes non-existence from a missed detection. Each of two
exchangeable slots is either unborn, alive, or permanently dead; while alive,
it may be assigned to one localization or missed. The sampler changes complete
birth/death intervals, reassigns individual frames, and globally permutes slot
labels. Exact enumeration remains feasible on four-frame calibration cases,
so existence, birth-time, death-time, clutter, path, and evidence diagnostics
are all measured against the true finite posterior. All posterior-distribution,
event-marginal, and path checks pass in 12/12 runs. The strict combined count is
11/12 because one finite sample misses its mass-coverage threshold by
\(7.6\times10^{-6}\) and the same run has an unstable log-evidence estimate;
even a 16,384-particle escalation does not pass the evidence threshold. The
sampler is therefore calibrated for posterior inference on this benchmark, not
for single-run marginal-likelihood comparison. See
`docs/wp4_birth_death_slots.md`.

WP5 removes the need to enumerate every combination of slot intervals during
SMC initialization. The exact sequential base sampler uses a joint
unborn--alive--dead HMM, while a blocked move redraws one slot's complete
existence interval and temporal associations with an exact Hastings
correction. All six blocked-hybrid runs pass exact targets with three slots and
five or six frames. A four-slot/eight-frame case runs without enumerating its
1,874,161 interval combinations, but independent posterior runs disagree by as
much as 0.2495 on expected clutter. The calibrated ceiling is therefore three
slots/six frames; four-slot mode weights are not yet validated. See
`docs/wp5_scalable_slots.md`.

The likelihood-informed paired-slot move passes all six of its three-slot
exact-target runs and reduces four-slot expected-clutter disagreement from
0.2495 to 0.0645. It nevertheless misses the frozen cross-run gate by a
small margin (alive-count TV 0.10010 versus 0.10), while costing about 1.9
times as much per four-slot run. Three untouched new seeds pass within their
triplet, so the result is promising but not a robust four-slot calibration.
At WP6, the exact-oracle ceiling was three slots/six frames. See
`docs/wp6_paired_moves.md`.

The WP7 audit extends exact-target checking to one sparse four-slot/four-frame
case: three independent 2,048-particle runs pass. It does not make the
four-slot/eight-frame problem calibrated. At 4,096 particles, the original
three-seed group passes all seven stability checks, but the held-out triplet
and combined six seeds fail on clutter and label-invariant count histograms.
At 8,192 particles, all seven metrics pass over both independent triplets
and their combined six runs (worst disagreement 0.0557). The eight-frame
result is therefore stable under this fixed test at substantial compute
cost, but its mode weights have not been checked against an exact oracle
and may not transfer to other corruption regimes. See
`docs/wp7_convergence_audit.md`.

## Data strategy

The project does not vendor third-party ULM code or data.

1. **Synthetic benchmark:** generated locally and fully controlled.
2. **PALA:** simulation ground truth and established localization benchmarks.
3. **ULMShare:** lightweight precomputed `tracks.json.xz` files first; selected
   raw acquisitions only when an image- or RF-level likelihood is needed.
4. **Open-3DULM and other pipelines:** external comparators and format adapters.

See `docs/resources.md` for the audited resources and licensing constraints.

## Planned method progression

1. Fixed cardinality, noisy localizations, enumerable posterior
2. Smooth Schrödinger Bridge reproduction on the same benchmark
3. Exact Gaussian localization posterior and differentiable potential
4. Insert the validated potential into TSBM and compare sampled mode weights
   — completed, current public configuration does not pass
5. Calibration-first collapsed SMC on the Matérn localization posterior
   — completed and passed; TSBM retained as a negative baseline
6. Explicit null associations for missed detections and false positives
   — completed and calibrated for persistent latent particles
7. Births/deaths and genuinely variable latent cardinality
   — completed for two explicit exchangeable slots on enumerable cases
8. Scale the discrete representation beyond two slots and four frames
   — exact three-slot/six-frame gate passed; the initial four-slot/eight-frame
   2,048-particle stability gate failed
9. Test likelihood-informed collective multi-slot moves — exact three-slot
   gate passed and four-slot disagreement improved, but the 2,048-particle
   frozen stability gate still failed narrowly
10. Audit four-slot convergence — sparse four-frame oracle passed;
    six-run eight-frame stability fails at 4,096 but passes at 8,192
11. Validate across new synthetic crossing and corruption regimes before
    selected in-vivo ULMShare trajectories
12. Only then: IQ/RF likelihoods and acquisition-level inference

## Scientific checkpoints

Three choices are intentionally not hidden in implementation details:

- latent representation: ordered particles, unordered point cloud, or
  variable-mass measure;
- observation model: noisy localizations versus IQ/RF data;
- first-paper claim: validated fixed-cardinality association posterior versus
  an immediate variable-cardinality extension.

The current frozen choice is a finite set of exchangeable labelled slots with
unordered noisy localizations and a time-varying existence mask. Labels are a
computational representation, not a claim that bubbles are intrinsically
ordered; exact checks now include three slots over six frames and one sparse
four-slot/four-frame case, but not the four-slot/eight-frame target.

