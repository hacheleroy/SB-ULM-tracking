# WP2 — Gaussian localization likelihood

Benchmark date: 2026-10-16.

## Question

WP1 showed that a physically calibrated motion prior is not sufficient when
noisy localizations are imposed as exact path states. WP2 changes only this
observation model. The physical crossing, Matérn-3/2 prior, 1 kHz frame rate,
6.84 mm/s speed scale, 256 association hypotheses, and evaluation diagnostics
remain fixed.

The comparison is therefore:

1. **hard observations:** each observed localization is treated as the latent
   position itself;
2. **Gaussian likelihood:** the latent path follows the same Matérn prior and
   each localization is a noisy observation with known standard deviation
   \(\sigma_{\mathrm{loc}}\).

The prior coordinate standard deviation is frozen at 12.488 µm in each spatial
dimension, independently of the simulated separation and noise realization.
Its velocity-calibrated temporal length scale is 4.472 ms. This prevents either
method from adapting its prior to the observations it is asked to explain.

## Exact enumerable posterior

For one particle and one coordinate, let the latent path obey:

\[
x_d \sim \mathcal N(0,K_d)
\]

and let the observed localizations obey:

\[
y_d\mid x_d \sim \mathcal N(x_d,\sigma_{\mathrm{loc}}^2 I)
\]

The latent path can be integrated analytically:

\[
y_d \sim \mathcal N(0,K_d+\sigma_{\mathrm{loc}}^2 I)
\]

Every complete association is scored with this Gaussian marginal and all 256
weights are normalized exactly. This provides a gold-standard posterior for
the fixed two-bubble benchmark; no approximate sampler is involved in this
WP2 acceptance test.

## Differentiable unordered observation potential

For later insertion into a controlled diffusion or Schrödinger-bridge method,
the package also implements the framewise set likelihood:

\[
p(Y_t\mid X_t)
=\frac{1}{N!}\sum_{\pi\in S_N}
\prod_{i=1}^{N}
\mathcal N\!\left(y_{t,\pi(i)};x_{t,i},
\sigma_{\mathrm{loc}}^2 I\right)
\]

and its potential:

\[
U_t(X_t)=-\log p(Y_t\mid X_t)
\]

The sum over permutations makes the potential invariant to the arbitrary order
of detections and latent particles. It is evaluated with `logsumexp`, and its
analytical spatial gradient is implemented and checked against finite
differences. Exact factorial enumeration is appropriate for this two-particle
calibration benchmark but is not proposed as a scalable tracking algorithm.

## Frozen physical map

The full experiment contains:

- 7 central separations from 0 to 20 µm;
- 7 localization-noise standard deviations from 1 to 20 µm;
- 64 noise and detection-order seeds per cell;
- 3,136 paired comparisons in total.

The same simulated observations and same fixed prior are used in both members
of every pair. The only difference is whether localization uncertainty is
ignored or represented in the likelihood.

At 95% nominal credible mass, reference-path inclusion over all 3,136 cases is
35.52% for hard observations and 99.01% for the Gaussian likelihood. The latter
is conservative on this frozen simulator rather than perfectly calibrated, but
it removes the severe false certainty of the hard formulation. All 49 cells
reach at least 95% mean inclusion under the likelihood; only 9 of 49 do so under
hard observations.

| Regime | Hard inclusion | Likelihood inclusion | Hard set size | Likelihood set size |
|---|---:|---:|---:|---:|
| Exact crossing, 5 µm noise | 0.266 | 0.969 | 1.53 | 7.53 |
| 10 µm separation, 5 µm noise | 0.391 | 1.000 | 1.47 | 5.92 |
| 20 µm separation, 2.5 µm noise | 1.000 | 1.000 | 1.02 | 1.08 |
| Exact crossing, 20 µm noise | 0.000 | 0.984 | 1.08 | 189.73 |

At 5 µm noise, mean reference inclusion across separation levels rises from
37.95% to 99.11%, while the mean 95% set size rises from 1.41 to 5.96. At 20 µm
noise, the likelihood correctly becomes broad: its mean set size is 181.66 of
256 hypotheses. This is not a loss of calibration; it records that the data no
longer identify a narrow set of associations.

The likelihood modestly improves aggregate MAP association accuracy, from
0.672 to 0.697, but MAP accuracy is not the main criterion. At very high noise,
the likelihood can return a less accurate MAP while providing a much more
honest distribution over alternatives.

## Interpretation and limits

These results validate the observation-model change on the controlled
benchmark. They do **not** yet validate a neural posterior sampler or a TSBM
training procedure. The implemented potential supplies the mathematical and
tested software interface needed for that next comparison.

The inclusion curve is also not a universal proof of Bayesian calibration. The
reference trajectories are deterministic crossing paths, not random draws from
the stationary Matérn prior. It should therefore be read as
**reference-path inclusion on the frozen simulator**. A later generative
calibration experiment should draw latent paths and observations from the
stated model and test nominal frequentist coverage separately.

The localization variance is known in this synthetic experiment. On real ULM
data it must be estimated, predicted per detection, or subjected to sensitivity
analysis. Missed detections, clutter, births, and deaths also remain outside
this fixed-cardinality gate.

## Decision

WP2 passes the exact-posterior gate: an explicit localization likelihood
eliminates the dominant overconfidence failure found in WP1 while preserving
the same physical prior and exact association diagnostics. The potential has
since been implemented in the public TSBM framework and evaluated against a
separate exact Brownian-reference target. That approximate sampler does not
pass the discrete or continuous calibration gate, so variable cardinality
remains paused. See `tsbm_wp2_integration.md`.

## Reproduce

```bash
python -m sbulm.wp2_likelihood \
  --output-dir results \
  --n-seeds 64 \
  --workers 6
```

The command writes the full CSV and JSON summaries plus the regime-map and
credible-inclusion figures.
