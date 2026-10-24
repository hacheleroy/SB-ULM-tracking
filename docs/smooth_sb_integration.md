# Smooth-SB integration contract

Audit date: 2026-10-16.

## Upstream version and scope

The external reference is
[WanliHongC/Smooth_SB](https://github.com/WanliHongC/Smooth_SB), pinned to commit
`2e0388a56e06dbd922f27012ba0441277adc0731` (2025-02-10). The relevant entry
point is `Tutorial_one_by_one_matching.ipynb`. At that revision, its loader
expects a NumPy archive in `G_data/`, reads the key `arr_0`, and interprets the
array as `(T+1, N, D)`: observations at all acquisition times, a constant number
of particles per time, and at most five spatial dimensions.

The repository has no visible license file. Its source is therefore not copied,
modified, or distributed in this project. The implementation should be run in
a separate checkout until reuse permission or a license is obtained.

## Frozen data interface

Generate the controlled ULM input with:

```bash
python -m sbulm.smooth_sb \
  --output results/sbulm_crossing_smooth_sb.npz \
  --seed 0 \
  --central-separation 0.05 \
  --localization-std 0.075
```

The archive contains:

| Key | Shape | Role |
|---|---|---|
| `arr_0` | `(9, 2, 2)` | unordered noisy observations consumed upstream |
| `times` | `(9,)` | acquisition-time audit metadata |
| `reference_ids` | `(9, 2)` | hidden identity of each row; evaluation only |
| `truth_positions` | `(9, 2, 2)` | latent paths; evaluation only |
| `truth_particle_ids` | `(2,)` | latent identifiers; evaluation only |

Rows in `arr_0` are shuffled independently at every frame. The hidden keys must
not be passed to the inference algorithm.

## Reproduction environment

The upstream repository pins POT 0.9.1, SciPy 1.10.1, scikit-learn 1.3.0,
NumPy 1.24.4, and Matplotlib 3.7.2. Its README names `requirements.txt`, whereas
the audited file is named `requirements.txt.txt`. Because these versions are
older than the main project's environment, use an isolated Python 3.10 or 3.11
environment rather than weakening this package's dependencies.

The one-by-one notebook also imports `pandas` through `evaluation_metric.py`,
but pandas is absent from the audited requirements file. The smoke test added
`pandas==2.0.3` explicitly. Notebook-kernel startup was unavailable in the
execution sandbox, so its code cells were executed sequentially in one Python
process without changing the mathematical code.

```bash
git clone https://github.com/WanliHongC/Smooth_SB.git
cd Smooth_SB
git checkout 2e0388a56e06dbd922f27012ba0441277adc0731
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt.txt
python -m pip install pandas==2.0.3
```

Copy the generated NPZ into upstream `G_data/`, add its stem to `G_data_list`
in the one-by-one-matching notebook, and retain the notebook's published
parameters for reported results. Reduced precomputation samples or iterations
may be used only for a clearly labelled smoke test.

## Required output and comparison

The upstream notebook constructs `D_sample` with shape `(S, T+1, N)`, where
each integer selects one row of `arr_0` at the corresponding frame. Save this
array as an NPZ or NPY result without reordering it. It can then be compared to
the exact enumerated posterior:

```python
from sbulm.smooth_sb import compare_smooth_sb_samples

comparison = compare_smooth_sb_samples(D_sample, exact_posterior)
print(comparison)
```

The comparison reports:

- mass assigned to invalid or non-bijective paths;
- total-variation distance from the exact path distribution;
- Jensen--Shannon divergence;
- sampling frequency of the exact MAP path;
- size of the sampled valid support.

Finite-sample error must be estimated by repeated sampling or bootstrap. The
first acceptance test is recovery of the two leading modes and their weights
in the central ambiguous regime; visually smooth interpolations alone do not
pass the decision gate.

## First external validation run

The exported seed-0 crossing was run with `nu=1.5`, length scale 3, scale
multiplier 1, 200 message-passing iterations, 20 precomputation samples, and
the published `Mi=32` basis. The same converged messages were sampled with five
random seeds, producing 2,000 path sets per seed. A representative complete run
took 24.9 seconds in the audit environment: 0.23 seconds for precomputation,
14.16 seconds for message passing, and 9.10 seconds for sampling and plotting.
These timings are indicative only because no hardware benchmark was performed.

Two exact distributions were enumerated on the same 256-path support:

| Quantity | Reference-consistent path | Smoother competing path |
|---|---:|---:|
| Localization-noise/constant-velocity posterior | 0.52344 | 0.47636 |
| Matérn-GP target matching Smooth-SB settings | 0.00024 | 0.99976 |
| Mean raw Smooth-SB sample frequency | 0.00000 | 0.99780 |

The remaining mean frequency, 0.00220, consists of non-bijective path sets; the
five seed-specific rates range from 0.00050 to 0.00450. Against the Matérn
target, mean total-variation distance is 0.00220 and mean Jensen--Shannon
divergence is 0.000848. After conditioning on valid bijective samples, these
fall to 0.000244 and 0.0000847, respectively. The tiny alternative target mode
has expected count below one in each 2,000-sample run, so its absence from these
finite samples is unsurprising.

This result is encouraging for the numerical sampler but changes the scientific
question. Smooth SB did not fail to reproduce the two-mode constant-velocity
posterior; it sampled a different posterior whose much stronger smoothness
prior makes one path essentially certain. The next experiment must therefore
calibrate the Matérn length scale and variance from plausible microbubble
dynamics, then rerun the separation-by-noise map. Target definition precedes
sampler comparison.

## Physical-prior checkpoint

The velocity calibration and full separation-by-noise map are complete. The
Matérn-3/2 length scale is now derived from the prior's RMS derivative and the
99-acquisition ULMShare velocity summary, while acquisition times remain in
seconds. See `physical_calibration.md` and the
`physical_matern_calibration.*` results.

The key result is that prior calibration is necessary but insufficient. At
localization noise of 5 µm or above, the hard-marginal target is frequently
overconfident in an incorrect smooth association. Reasonable speed-scale
variation does not remove this failure. This supports moving to a
measurement-likelihood formulation rather than further tuning the hard
marginals.

## Subsequent TSBM gate

The exact WP2 likelihood gate is complete. A Gaussian localization likelihood,
its order-invariant differentiable set potential, and its analytical gradient
are implemented and tested. On the frozen 7-by-7 map, 95% reference-path
inclusion rises from 35.52% under hard observations to 99.01% under the
likelihood. See `wp2_localization_likelihood.md`.

That step has now been executed. The potential and its autograd gradient match
the local implementation at machine precision. Because public TSBM uses a
Brownian reference rather than the frozen Matérn prior, its samples were tested
against a separately derived exact Brownian/Sinkhorn target. The sampler finds
the credible modes but fails their weights, endpoint coupling, and continuous
path distribution on both easy and ambiguous cells. No missed-detection or
variable-cardinality mechanism is therefore added. See
`tsbm_wp2_integration.md`.

The independent reproduction of the authors' original one-by-one N-body
example remains part of WP1. Record wall time, peak memory, convergence
diagnostics, sampled-path distribution, and any compatibility wrapper; do not
copy the unlicensed upstream source into this repository.

An unchanged first attempt profiled the computational cost without completing
the 200-iteration run. Precomputation took 2.46 seconds, whereas message passing
reached at least iteration 90 after 15 minutes on one CPU core. The reported
norm change fell from 16,689.7 at iteration 1 to 0.002783 at iteration 90, and
resident memory remained around 250 MiB. The run was stopped by the external
time guard before sampling, so it is deliberately not marked as a successful
reproduction and no partial upstream artifacts are distributed here.
