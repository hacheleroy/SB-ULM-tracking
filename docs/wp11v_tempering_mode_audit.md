# WP11v — Independent tempering-stage count/clutter audit

WP11u improved final samples on the exposed WP11k primary but failed its
frozen absolute gate. This diagnostic uses a new synthetic four-slot,
eight-frame observation to identify the first sustained count/clutter
discrepancy during the **unchanged WP11g 10% four-path PG sampler**. It does
not test or select another move.

## Observation and reference frozen before inference

Use the WP8 crossing generator with staggered truth mask, central separation
0.006, detection probability 0.50, expected clutter 0.35 per frame, and
maximum two clutter points per frame. Try trajectory/corruption seed pairs
in this order: `(101,103)`, `(107,109)`, `(113,127)`, `(131,137)`,
`(139,149)`, `(151,157)`, `(163,167)`, `(173,179)`. Select the **first**
case with 11–14 total detections, at least one detection in both frame six
and frame eight, and at least four detections across frames three through
five. This rule sees only cardinalities. If none qualifies, stop and report
that the prespecified ladder failed. Pin the selected detection hash and
cardinalities before any posterior computation.

The first qualifying case was the third pair `(113,127)`: cardinalities
`[2,2,2,3,1,1,0,2]`, 13 detections, detection SHA-256
`400975d533de5fa9ba5833f0ade92e20045dd7397701637cd8db4534c531e1c4`.
This selection used no posterior or sampler results.

For the selected observation, evaluate the tempered target

\[
\pi_\beta(a) \propto p_{\rm BD}(a)\,p_{\rm count}(a)
                  \exp\{\beta\,\ell_{\rm GP}(a)\}, \qquad 0\le\beta\le1.
\]

Reuse squarefree four-slot subset convolution with a beta-scaled one-slot
collapsed GP term and Fourier extraction of alive totals. The joint is
alive total × frame-six clutter × last-frame clutter. Validate against
complete enumeration on a separate four-slot, three-frame toy at beta 0,
0.5, and 1: log partition error ≤1e-9, joint TV ≤1e-9, maximum cell
error ≤1e-10. At beta zero the selected-case log partition must equal
the model's base normalizer within 1e-8. On the selected case, compare
direct fourth power and pairwise square convolutions at beta 0, 0.5 and
1: log partition difference ≤1e-8, joint TV ≤1e-7 and maximum cell
difference ≤1e-8. Every computed beta must have raw minimum cell at least
-1e-10, relative imaginary residual ≤1e-8 and raw mass error ≤1e-8;
clip smaller negative roundoff and record its TV. These are numerical
cross-checks, not a formal interval bound. If any check fails, stop before
sampler inference.

## Fixed stage comparison

Run unchanged WP11g controls with 2,048 outer particles, seeds
`1400..1403`, one BLAS thread, one process per seed and a 750 CPU-second
cap. A passive observer records at every adaptive stage: the prior
unweighted joint, the likelihood-weighted joint, the resampled joint,
the rejuvenated joint, beta, ESS and parent/state diversity. It consumes
no sampler RNG. Calculate exact tempered joints at each **observed** beta
after sampling with the frozen reference algorithm. Compare the prior
joint to exact at the previous beta and the other three joints to exact
at the current beta. Report full-joint TV, low ≤20 / mid 21 / high ≥22
alive-band × frame-six-clutter TV, maximum band-mass error and both
clutter-mean errors.

For a descriptive onset marker, set the band-by-sixth TV threshold at
`max(0.05, 2 × exact-IID q95)` at each beta using 1,000 multinomial draws
of 2,048 from that exact joint (seed 34621 plus the beta's fixed ordinal
index in the sorted union of all stage betas). Mark a seed's first pair
of consecutive **post-rejuvenation** stages above threshold. Report its
beta and the operation at which the discrepancy first appears in that
pair. Also report medians of signed changes in TV across weighting,
resampling and rejuvenation at their respective targets. These are
descriptive diagnostics; correlated SMC particles and adaptive beta
choices do not make IID q95 a calibrated significance test. Do not
change thresholds after seeing traces or choose a move from this audit.

No 8,192-particle run or untouched WP8 holdout is part of WP11v.

The exact checkpoint was validated before sampler inference and pinned by
SHA-256 `4577d3c6214c4493b5b59b0b8302936f5af04bc37e0ed1c16c0cb8f1b4ce06d6`.
Toy enumeration agreed at beta 0, 0.5 and 1 to at most `1.1e-15` TV.
On the selected case, direct and paired subset calculations agreed to
at most `2.4e-14` TV and `1.5e-14` in log partition. At beta zero the
computed partition matched the model's base normalizer to `7e-15`.
The exact beta-one alive bands are 0.89574 / 0.04226 / 0.06200 and
frame-six clutter 0 / 1 masses are 0.95124 / 0.04876. Thus the
prespecified geometry did **not** retain balanced low/high alive bands;
the audit can locate discrepancy on this independent case but does not
replicate WP11t's competing high-alive mode. This observation is kept
unchanged rather than selecting another after seeing its posterior.

## Outcome

All four fixed runs completed within the 750 CPU-second cap. They had
16–17 stages and consumed 196.7–205.3 CPU seconds each. The exact joint
was evaluated at all 65 distinct observed betas. Across those evaluations,
the largest relative imaginary residual was `2.8e-19`, raw mass error
was zero at the reported precision, and largest clipping TV was
`4.0e-15`; no numerical envelope failed. Every recorded joint was
normalized, and each beta sequence increased to one.

| Seed | Stages | First sustained onset | Peak post-rejuvenation band × sixth TV | Final full TV | Final band × sixth TV | Final maximum alive-band error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1400 | 17 | None | 0.0401 | 0.0917 | 0.0401 | 0.0393 |
| 1401 | 17 | None | 0.0273 | 0.0694 | 0.0088 | 0.0066 |
| 1402 | 16 | None | 0.0415 | 0.0764 | 0.0145 | 0.0124 |
| 1403 | 17 | None | 0.0303 | 0.0673 | 0.0087 | 0.0042 |

The descriptive onset limit was `0.05` at every observed beta. No
post-rejuvenation stage exceeded it. Seed 1402 had a transient weighted
joint above the limit at stage nine (`beta≈0.651`); resampling brought it
below the limit, and it did not meet the two-stage onset rule. Median
signed changes in full-joint TV across all 67 stages were `+0.01350`
for weighting, `+0.00291` for resampling, and `−0.01370` for
rejuvenation. Weighting compares the prior-beta sample with the
current-beta target, so this signed change also includes movement of
the target. Final full-joint TVs remain `0.0673–0.0917`, and the largest
final frame-eight clutter-mean error is `0.0275` detections.

This independent case does not show sustained count/clutter band loss
under the predeclared marker. Its beta-one low/high alive masses are
strongly unbalanced, so it cannot settle the WP11t competing-mode
problem. The four seeds are descriptive runs, not an independent
replication of WP11t or a reason to choose an earlier-stage move. A next
development gate needs an independently validated, more balanced
eight-frame mode reference and a comparison frozen before inference.
The untouched holdout remains closed.

The committed concise outcome is `results/wp11v_tempering_summary.json`.
The runner reproduces the detailed stage traces in `results/wp11v_runs/`
and exact stage comparisons in `results/wp11v_tempering_mode_audit.json`
using the pinned reference and seeds. Those larger generated traces were
kept out of this PR.
