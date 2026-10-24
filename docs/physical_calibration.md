# Physical calibration of the Matérn motion target

Audit and benchmark date: 2026-10-16.

## Empirical scale used in this benchmark

The lightweight ULMShare repository was audited at commit
`6679f16190c5583a57be506133ebbcb2235ddf45`. For the 99 acquisitions listed in
its official `metadata_summary.csv`, the corresponding `metrics.json` files
report the following acquisition-level mean velocities:

| Statistic | Mean velocity (mm/s) |
|---|---:|
| Minimum | 5.810 |
| 5th percentile | 6.219 |
| 25th percentile | 6.585 |
| Median | 6.840 |
| Mean | 7.656 |
| 75th percentile | 7.345 |
| 95th percentile | 11.622 |
| Maximum | 17.630 |

These are means of reconstructed tracks within acquisitions, not direct
samples from a population-level RMS-speed distribution. The median is therefore
used only as a transparent design scale, with the 5th and 95th percentiles used
for sensitivity analysis. It is not claimed to be a universal physiological
prior.

The reference acquisition rate is 1,000 Hz. This matches the open PALA in-vivo
rat-brain example audited at commit
`f3e7813767d747139e67a0d3f1a7c60b745a698a`. At 6.84 mm/s, a bubble moves
approximately 6.84 µm per frame. The controlled nine-frame crossing is expressed
in millimetres and seconds, with a half-span of 27.36 µm over 8 ms.

## Calibration equation

For independent coordinates of a stationary Matérn-3/2 process with marginal
standard deviations \(\sigma_d\) and temporal length scale \(\ell\):

\[
\mathbb{E}\!\left[\lVert x'(t)\rVert^2\right]
= \frac{3}{\ell^2}\sum_d\sigma_d^2
\]

Matching an RMS speed scale \(v_{\mathrm{RMS}}\) therefore gives:

\[
\ell
= \frac{\sqrt{3\sum_d\sigma_d^2}}{v_{\mathrm{RMS}}}
\]

The implementation deliberately retains the upstream notebook's empirical
variance rule, \(\sigma_d=\operatorname{SD}(D_{:,:,d})\), so the only conceptual
change is to express time and speed in physical units. For Matérn-5/2, the
coefficient 3 is replaced by \(5/3\).

Matérn-3/2 has a finite first-derivative variance but not a finite
second-derivative variance. It can therefore be calibrated from speed, but not
simultaneously from acceleration. An acceleration-calibrated model requires at
least Matérn-5/2 or a phase-space/state-space motion prior.

## Full regime map

The physical sweep contains seven central separations from 0 to 20 µm and seven
localization-noise standard deviations from 1 to 20 µm. Each of the 49 cells is
averaged over 16 independent noise and detection-order seeds. Every cell is
evaluated twice on the same observations:

1. the upstream target with times rescaled to \([0,2]\) and \(\ell=3\);
2. the velocity-calibrated target using physical frame times and the 6.84 mm/s
   design scale.

The calibrated length scale ranges from 4.47 to 8.32 ms across the map. The
upstream default corresponds to 12 ms on an 8 ms sequence, so calibration does
make the prior less rigid. At an exact crossing with 1 µm localization noise,
the mean mass of the reference-consistent association rises from 0.316 to
0.455, and it remains in the 95% credible set in all 16 runs.

The central conclusion is nevertheless negative and useful. At 5 µm separation
and 5 µm localization noise, the calibrated target assigns only 0.292 mean mass
to the reference-consistent path, with 95% credible-set coverage in 5 of 16
runs. At 5 µm noise, coverage across the separation grid is only 0.188--0.562.
The posterior is generally low-entropy in these failures: it is often confident
in a smoother but incorrect identity switch.

Changing the assumed speed from the ULMShare 5th percentile (6.219 mm/s) to the
95th percentile (11.622 mm/s) shortens the mean length scale from 5.26 to
2.81 ms and increases the 95% credible-set size from 1.31 to 3.00. It does not
materially recover the true-path mass in the selected 5 µm/5 µm regime
(0.182 versus 0.190). The failure is therefore not resolved by reasonable
speed tuning alone.

## Decision

WP1 passes its target-definition checkpoint but exposes a model mismatch:
Smooth SB's one-by-one formulation treats every noisy localization as a hard
intermediate marginal. It has no likelihood term stating that an observed point
is a noisy measurement of a latent bubble position. Consequently, changing
identities can be rewarded whenever it creates a smoother path through noisy
points.

This result motivated WP2: introduce a differentiable Gaussian localization
likelihood or path potential while keeping the same physical map and exact
association diagnostics. That experiment is now complete. It confirms that
prior calibration alone should not be presented as a calibrated ULM posterior:
on 3,136 paired cases, 95% reference-path inclusion rises from 35.52% under
hard observations to 99.01% under the Gaussian likelihood. See
`wp2_localization_likelihood.md` for the derivation, results, and limitations.

## Reproduce

```bash
python -m sbulm.physical_sweep \
  --output-dir results \
  --n-seeds 16 \
  --sensitivity-seeds 32 \
  --workers 6
```

The command writes the complete JSON and CSV data plus two figures. The source
data repositories are not vendored.
