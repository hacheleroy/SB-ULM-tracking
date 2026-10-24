# WP2 — Public TSBM integration and calibration gate

Benchmark date: 2026-10-19.

## Scope

This checkpoint inserts the differentiable unordered Gaussian localization
potential into the [public implementation of Twisted Schrödinger Bridge
Matching](https://github.com/maxencenoble/twisted-sb-matching) ([paper](https://arxiv.org/abs/2607.16987),
TSBM). The external repository is pinned to commit
`c9dfec715266a2fd45e1ab195eb3b388fa6e8ee8` and remains external to this
project. Its license is MIT.

The public method uses a Brownian reference process, whereas the exact WP2
posterior uses a Matérn-3/2 prior. Comparing their samples directly would
confound sampler error with a change of prior. The calibration therefore uses a
separate, exactly enumerable target matched to TSBM:

- the same nine-frame physical ULM crossing at 1 kHz;
- two labelled latent particles and unordered localization sets;
- a Brownian diffusion whose one-frame two-dimensional RMS displacement is
  6.84 µm;
- exchangeable initial and final permutation marginals;
- the Gaussian localization likelihood at the seven interior frames.

This is a sampler-calibration target, not a replacement for the Matérn WP2
model.

## Exact matched target

For each interior frame, the potential is the negative log likelihood of the
unordered set:

\[
U_k(X_k)=-\log\left[
\frac{1}{N!}\sum_{\pi\in S_N}\prod_{i=1}^{N}
\mathcal N(y_{k,\pi(i)};x_{k,i},\sigma_{\mathrm{loc}}^2I)
\right]
\]

The discrete TSBM reference weights paths by
\(\exp[-\sum_k V_k/\sigma^2]\). The adapter therefore supplies
\(V_k=\sigma^2U_k\); omitting this factor would silently change the assumed
localization variance.

For two particles and nine frames, all 512 labelled association paths can be
enumerated. Conditional on each endpoint pair, the Brownian bridge and Gaussian
observations can be integrated analytically. The resulting endpoint kernel is
then scaled in the log domain by Sinkhorn iterations so that both prescribed
permutation marginals are exactly uniform. This provides:

- exact association probabilities;
- exact endpoint-coupling probabilities;
- exact continuous latent-path samples;
- finite-sample references for the empirical metrics.

Global particle labels are arbitrary. Results are therefore reported both on
the 512 labelled hypotheses and after quotienting this symmetry to 256
canonical hypotheses.

## Software validation

The TSBM adapter implements the upstream discrete-cost interface without
modifying the external checkout. Its values and autograd derivatives agree with
the independent NumPy implementation at machine precision. Unit tests also
cover the Brownian covariance, endpoint Sinkhorn constraints, exact continuous
sampling, global-label quotient, finite-sample baseline, and path-to-association
projection.

The runner records the upstream commit, every training parameter, both endpoint
marginals, leading mode frequencies, discrete TV/JS distances, continuous
sliced-Wasserstein distance, and localization residuals. Forward and backward
fields can be evaluated without retraining.

TSBM returns continuous trajectories rather than an explicit association
variable. Association probabilities are therefore not estimated by simply
assigning every point to its nearest localization. At each interior frame, the
code computes the conditional probability of every detection permutation under
the Gaussian likelihood and averages these probabilities across sampled paths.
This Rao--Blackwellized estimator is used for TV and JS. Hard nearest-assignment
projections are retained only to test whether the sampler visits the credible
modes and covers their support.

## Acceptance gate

The consolidated evaluation applies the following fixed operational gate in
both the easy and ambiguous regimes:

1. Rao--Blackwellized canonical TV below
   \(\max(0.05,3q_{0.95}^{\mathrm{MC}})\);
2. Rao--Blackwellized canonical JS below
   \(\max(0.01,3q_{0.95}^{\mathrm{MC}})\);
3. endpoint-joint TV at most 0.05;
4. continuous sliced-Wasserstein distance below
   \(\max(1\,\mu\mathrm m,3d_{\mathrm{MC}})\);
5. all hypotheses in the exact 95% credible set observed;
6. sampled support covering at least 95% of exact posterior mass.

The last two checks prevent a deceptively good average distance obtained after
dropping modes. The first four test their weights and continuous geometry.

## Results

| Regime and checkpoint | RB canonical TV | MC 95% TV | Endpoint-joint TV | Path SW (µm) | MC path SW (µm) | Gate |
|---|---:|---:|---:|---:|---:|---|
| Ambiguous crossing, best TV checkpoint | 0.201 | 0.021 | 0.095 | 3.692 | 0.134 | Fail |
| Easy crossing, best TV checkpoint | 0.878 | 0.001 | 0.852 | 15.357 | 0.059 | Fail |

In the ambiguous case, the sampler finds every member of the exact 95%
credible set and its empirical support covers more than 99% of exact mass. It
therefore discovers the relevant modes, but assigns them incorrect weights.
The best Rao--Blackwellized canonical TV is about ten times the
95th-percentile Monte-Carlo reference. Its median interior localization
residual is about 10.4 µm, versus 5.5 µm for exact target samples.

The easy case is a stronger warning. Its exact posterior assigns 99.86% of the
endpoint coupling to the off-diagonal permutation pair corresponding to the
coherent crossing. The fresh pretrained sampler assigns only 14.65% there;
subsequent forward/backward updates reduce this fraction further. The failure
is therefore not hidden by a visually plausible path or by correct individual
endpoint marginals, both of which remain close to 50/50.

These results establish a **no-go for downstream use of the current public
TSBM configuration** on this ULM target. They do not establish that TSBM cannot
be adapted successfully. Training was performed on CPU with a reduced budget
relative to the paper's longest runs, and the upstream Gaussian-spline
reciprocal approximation may be restrictive for an unordered mixture
potential. The observed monotone movement toward the wrong easy-case coupling,
however, means that simply adding missed detections or real data would be
scientifically unjustified.

## Interpretation

The experiment separates what works from what does not:

- **Passed:** exact potential/gradient parity, public-code integration,
  exchangeable boundary marginals, leading-mode discovery, checkpointed
  forward/backward evaluation.
- **Failed:** endpoint coupling, posterior mode weights, and continuous path
  distribution.
- **Still mismatched to WP2:** the public implementation uses a Brownian
  reference rather than the velocity-calibrated Matérn prior.

Plausible method-level causes are the unimodal Gaussian reciprocal
approximation for a non-convex unordered likelihood, local optimization of the
endpoint coupling, and the Brownian reference itself. These are hypotheses,
not conclusions about the TSBM theory.

## Decision boundary

At this checkpoint, adding missed detections was paused and the next step was a
genuine project choice:

1. **Calibration-first posterior sampling (recommended):** implement a
   sequential/annealed Monte Carlo sampler directly for the original Matérn
   localization posterior, validate it against the existing exact target, and
   retain TSBM as a negative baseline.
2. **TSBM method development:** replace or enrich the Gaussian reciprocal
   family, introduce a mode-aware coupling initialization, and eventually add
   a phase-space Matérn reference before returning to ULM complexity.

The first route advances the ULM inference question. The second is a more
fundamental Schrödinger-bridge methods project.

**Subsequent decision:** Option A was selected. The collapsed adaptive-SMC
implementation now passes its fixed-cardinality exact-target gate, including
association weights, continuous paths, and log evidence. The public TSBM
configuration remains in the benchmark as a negative baseline; no further
TSBM training was used to obtain the positive SMC result. See
`smc_matern_calibration.md`.

## Reproduction

Install this package with the `tsbm` extra, clone the public TSBM repository,
and check out the pinned commit. The optional extra contains the dependencies
imported by the audited public training path; the upstream environment file
remains the authoritative fallback if that repository changes. A single run is
launched with:

```bash
PYTHONPATH=src:/path/to/twisted-sb-matching \
python experiments/run_tsbm_wp2.py \
  --tsbm-root /path/to/twisted-sb-matching \
  --output-dir results/tsbm_work \
  --regime ambiguous
```

The full runner exposes the network, spline, control-variate, checkpoint-resume,
and sampling-direction settings used in the calibration. The compact JSON
retains these configurations, the pinned upstream commit, checkpoint lineage
without machine-local paths, and all gate metrics. Compact accepted outputs
are `results/tsbm_wp2_calibration.json` and
`results/tsbm_wp2_calibration.png`; raw checkpoints and samples are deliberately
excluded from version control.
