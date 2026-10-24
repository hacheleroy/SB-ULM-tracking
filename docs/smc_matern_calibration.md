# Option A — Calibrated SMC for the Matérn ULM posterior

Benchmark date: 2026-10-19.

## Decision and scope

Option A implements a calibration-first posterior sampler for the original
fixed-cardinality Matérn localization model. It does not use the public TSBM
network and it is not itself a diffusion model. Its role is to establish a
trusted posterior-sampling baseline before adding missed detections, births,
deaths, or real-data complications. The public TSBM result remains a negative
baseline for its current configuration.

The method follows the SMC-sampler construction of
[Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
The temperature increments are selected by an effective-sample-size criterion,
in the spirit of the adaptive construction described by
[Zhou, Johansen, and Aston (2016)](https://doi.org/10.1080/10618600.2015.1060885).

## Target distribution

Let (a) be a complete sequence of detection permutations, (x) the labelled
latent bubble paths, and (y) the unordered noisy localizations. Frame zero
fixes arbitrary global labels. The joint target is:

\[
p(a,x\mid y)\propto p(y\mid x,a)p(x)p(a)
\]

Each coordinate of each bubble has the frozen Matérn-3/2 prior used in WP2,
and each assigned localization has independent Gaussian error. Conditional on
an association, the latent path can be integrated analytically:

\[
p(y\mid a)=
\prod_{i,d}\mathcal N\!\left(y_{a,i,d};0,
K_d+\sigma_{\mathrm{loc}}^2I\right)
\]

The SMC sampler therefore operates on the collapsed discrete target:

\[
p(a\mid y)\propto p(y\mid a)p(a)
\]

After an association has been sampled, the continuous path is drawn exactly
from its conjugate conditional distribution:

\[
\begin{aligned}
\mu_{a,d}
  &=K_d(K_d+\sigma_{\mathrm{loc}}^2I)^{-1}y_{a,d}\\
\Sigma_d
  &=K_d-K_d(K_d+\sigma_{\mathrm{loc}}^2I)^{-1}K_d
\end{aligned}
\]

Thus every returned item is a joint posterior draw ((a,x)), not merely a
hard track or a sample of detection indices.

## SMC construction

The intermediate targets are:

\[
\pi_\beta(a)\propto p(y\mid a)^\beta p_0(a),
\qquad 0=\beta_0<\cdots<\beta_L=1
\]

where (p_0) is uniform over frame-wise permutations. The implementation:

1. samples initial associations from (p_0);
2. chooses each new temperature by bisection so that the incremental ESS is
   approximately 80% of the particle count;
3. applies systematic resampling at every temperature;
4. rejuvenates particles with symmetric Metropolis moves that swap labels in
   one frame or over a contiguous frame segment;
5. performs eight additional invariant Metropolis sweeps at (eta=1) to
   reduce genealogical duplicates from the final resampling;
6. draws the continuous Matérn path exactly conditional on each final
   association.

The inference routine never calls the exact enumerator. Enumeration is used
only by the calibration harness on the two-particle, nine-frame benchmark.

## Validation protocol

The primary gate uses 8,192 particles, eight Metropolis sweeps per temperature,
eight additional final sweeps, and three independent SMC seeds in each frozen
regime:

- easy: 20 µm central separation and 2.5 µm localization-noise SD;
- ambiguous: 0 µm separation and 5 µm noise;
- high noise: 0 µm separation and 20 µm noise.

The gate requires:

- association TV no larger than
  \(\max(0.05,3q_{0.95}^{\mathrm{exact\ MC}})\);
- association JS no larger than
  \(\max(0.01,3q_{0.95}^{\mathrm{exact\ MC}})\);
- 95% credible-set recall and exact mass covered no worse than the fifth
  percentile of equally sized exact posterior samples, with an absolute 95%
  mass floor;
- continuous sliced-Wasserstein distance no larger than the maximum of 1 µm
  and three times an independent exact-sample baseline;
- absolute log-evidence error no larger than 0.10 nat.

The exact finite-sample reference is important for high-entropy targets. With
4,096 draws from a posterior spread over nearly all 256 associations, requiring
every low-mass state to appear is not a valid deterministic gate even for an
exact sampler. The stricter all-states-observed result is still reported as a
diagnostic.

## Results

All nine primary runs pass every discrete, continuous, support, and evidence
check.

| Regime | Mean association TV | Mean path SW (µm) | Mean absolute log-evidence error (nat) | Gate |
|---|---:|---:|---:|---|
| Easy | 0.000069 | 0.0391 | 0.0305 | Pass, 3/3 |
| Ambiguous | 0.01136 | 0.1000 | 0.00532 | Pass, 3/3 |
| High noise | 0.06228 | 0.1708 | 0.00456 | Pass, 3/3 |

The largest primary-gate ratios to their respective thresholds are 0.308 for
association TV, 0.192 for continuous sliced-Wasserstein distance, and 0.490
for absolute log-evidence error.

The stress test uses 4,096 particles on 196 cases: seven separations, seven
noise levels, and four independently simulated data seeds per cell.

- TV passes in 196/196 cases; the largest threshold ratio is 0.366.
- JS passes in 196/196 cases; the largest threshold ratio is 0.357.
- finite-sample-calibrated credible-set recall passes in 196/196 cases.
- every case covers more than 98.80% of exact posterior mass.
- all members of the exact 95% credible set are observed in 192/196 cases.
- the fifth-percentile mass-support check passes in 183/196 cases. A test
  defined at the fifth percentile is expected to reject about 5% of exact
  finite samples; this cellwise count is retained as a diagnostic rather than
  being misreported as a systematic target-distribution failure.
- the adaptive schedule uses between one and ten temperatures across the map.

The final rejuvenation was introduced after a controlled ablation: eight
additional target-invariant sweeps resolved every substantive support deficit
among the initial borderline cases. A (10^{-12}) tolerance handles only
floating-point comparisons at otherwise identical probability masses.

## Interpretation and boundary

This is a positive result for the narrow first-paper claim: on enumerable
fixed-cardinality localization problems, the implementation samples the
specified multimodal posterior with calibrated mode weights and continuous
path geometry. It also supplies a defensible reference against which learned
controls, Schrödinger bridges, or direct posterior networks can be compared.

It does **not** yet establish scalability to many simultaneous bubbles. The
state still contains a permutation per frame, the initialization is uniform,
and the Metropolis kernels are designed for small fixed cardinality. It also
does not cover false positives, missed detections, births/deaths, IQ/RF data,
or in-vivo model mismatch.

The next scientific decision is therefore no longer whether the localization
posterior can be sampled at all. It is how to represent variable cardinality:
explicit discrete birth/death and association moves, a padded ordered state,
or a variable-mass point-process formulation. That representation should be
chosen before extending this sampler to WP3.

## Reproduction

```bash
python -m sbulm.smc_calibration \
  --output-dir results \
  --n-particles 8192 \
  --n-smc-repeats 3 \
  --map-n-particles 4096 \
  --map-n-data-seeds 4 \
  --mh-sweeps 8 \
  --final-mh-sweeps 8 \
  --workers 6
```

The compact outputs are `smc_matern_calibration.json/.png` and
`smc_matern_calibration_map.csv/.png`. The JSON includes every primary run,
every map evaluation, each adaptive temperature, ESS, acceptance rate,
finite-sample reference, and gate decision.
