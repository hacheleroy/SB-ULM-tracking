# Controlled-crossing benchmark results

These results validate the software and freeze the calibration benchmark. They
do not yet constitute a comparison with a Schrödinger-bridge method.

## Central benchmark

- 32 independent noise/permutation seeds
- 9 frames and 2 bubbles on orthogonal vessel branches
- central bubble separation: 0.05 arbitrary position units
- localization-noise standard deviation: 0.075 units
- 256 complete association hypotheses per sequence
- uniform prior over associations
- each hypothesis scored with the same linear-Gaussian constant-velocity model

| Method/diagnostic | Mean |
|---|---:|
| Position-only Hungarian association accuracy | 0.469 |
| Position-only Hungarian trajectory RMSE | 0.426 |
| Constant-velocity Hungarian association accuracy | 0.898 |
| Constant-velocity Hungarian trajectory RMSE | 0.101 |
| Posterior MAP association accuracy | 0.953 |
| Posterior mass of the exact true association | 0.557 |
| True association contained in 95% credible set | 32/32 |
| Mean 95% credible-set size | 2.03 hypotheses |

The nearest-neighbour tracker usually loses identity near the crossing. The
constant-velocity tracker performs well on average but returns only one answer.
The enumerable posterior improves the MAP decision and, more importantly,
usually retains two coherent hypotheses instead of hiding the ambiguity.

## Separation-by-noise difficulty map

The second experiment evaluates 49 cells: seven central separations from 0 to
0.20 and seven localization-noise levels from 0.01 to 0.20. Each cell contains
16 independent seeds. The main regimes are:

| Example regime | CV accuracy | Posterior MAP accuracy | True-mode mass | Mean 95% set size |
|---|---:|---:|---:|---:|
| Exact crossing, noise 0.01 | 0.922 | 0.906 | 0.474 | 2.00 |
| Separation 0.05, noise 0.075 | 0.883 | 0.945 | 0.519 | 2.00 |
| Separation 0.10, noise 0.05 | 0.930 | 0.984 | 0.805 | 1.75 |
| Separation 0.20, noise 0.025 | 1.000 | 1.000 | 1.000 | 1.00 |
| Exact crossing, noise 0.20 | 0.773 | 0.914 | 0.327 | 5.44 |

This map separates three situations. With large separation and low noise, one
association dominates. At a tight crossing, two paths can remain genuinely
competitive even with precise localizations. At high noise, posterior mass is
spread across a larger set of paths, so reporting only a MAP trajectory is
particularly incomplete.

The true association was in the nominal 95% set in every cell except five
cells at noise 0.15, where coverage was 15/16. With only 16 seeds per cell this
is not evidence of systematic miscalibration; the larger final benchmark must
report confidence intervals and use more replicates.

## Consequence for Smooth SB

An approximate method must be tested in all three regimes. Matching the MAP
trajectory in the easy regime is insufficient. The decisive quantities are the
probabilities assigned to competing association paths, invalid/non-bijective
sample frequency, total-variation distance, and Jensen--Shannon divergence from
the enumerable target. The exported upstream-compatible dataset and these
metrics are implemented in `sbulm.smooth_sb`.

An initial external smoke test exposed an essential distinction. For the same
seed-0 observations, the localization-noise/constant-velocity posterior assigns
mass 0.523 to the reference-consistent path and 0.476 to the smoother competing
path. The exact discrete Matérn-GP target corresponding to the Smooth-SB
notebook settings instead assigns 0.99976 to the smoother path. Smooth SB's
valid samples agree with the latter target, not the former: their conditional
total-variation distance is 0.00024. Thus the apparent mode collapse disappears
once the targets are aligned.

At the published `Mi=32` basis, five sampling seeds produced 10,000 path sets.
The smoother path had mean frequency 0.9978 and the mean non-bijective rate was
0.0022 (range 0.0005--0.0045). The mean total-variation distance from the exact
Matérn target was therefore 0.0022; conditional on bijective samples it was
0.00024. A reduced `Mi=8` smoke test had produced a much larger 0.0515 invalid
rate, showing that this diagnostic is sensitive to numerical discretization.

This first result is numerically encouraging, but it is not yet an ULM-method
validation. It shows that Smooth SB samples its own strong smoothness target on
the controlled case. The next scientific step is to calibrate that target from
plausible bubble dynamics and determine whether its implied mode weights are
appropriate for ULM.

## Physical Matérn calibration

The next step has now been executed in millimetres and seconds. The benchmark
uses a 1 kHz acquisition rate and a 6.84 mm/s reference speed, the median of the
acquisition-level mean velocities reported for the 99 official ULMShare
acquisitions. For a Matérn-3/2 prior, the temporal length scale is chosen so its
RMS derivative matches that speed scale. Across the 7-by-7 map, the resulting
length scales range from 4.47 to 8.32 ms, compared with a 12 ms physical
equivalent for the upstream default.

Calibration helps in the low-noise exact-crossing regime: at 1 µm localization
noise, reference-consistent mass increases from 0.316 to 0.455. It does not,
however, make the hard-marginal target a calibrated localization posterior. At
5 µm separation and 5 µm noise, the reference-consistent path has mean mass
0.292 and appears in the nominal 95% set in only 5 of 16 runs. Increasing the
speed scale across the ULMShare 5th--95th percentile range broadens the
posterior but does not materially recover reference-path mass.

This failure is mechanistically informative: the one-by-one Smooth-SB target
treats noisy localizations as exact intermediate marginals, so it can prefer an
identity switch that happens to trace a smoother path. This finding motivated
the differentiable localization likelihood/potential evaluated in WP2, with
the physical map retained as the frozen acceptance test. The calibration
derivation is in `physical_calibration.md`.

## WP2 Gaussian localization likelihood

That acceptance test is now complete. The same fixed Matérn-3/2 prior was used
for hard observations and for a model in which each localization is a Gaussian
measurement of a latent path. Across 49 physical regimes and 64 paired seeds
per regime, the reference-consistent association appears in the nominal 95%
set in 35.52% of hard-observation cases and 99.01% of Gaussian-likelihood
cases. At 5 µm noise, mean inclusion across separations improves from 37.95%
to 99.11%, while the credible set expands from 1.41 to 5.96 hypotheses.

The central result is not merely a better MAP estimate. The explicit
likelihood stops the posterior from becoming spuriously narrow when
localizations are noisy. At high noise it correctly returns a broad set of
association hypotheses. The full model, equations, limitations, and
reproduction command are in `wp2_localization_likelihood.md`.
