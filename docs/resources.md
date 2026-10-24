# Audited open resources

Audit date: 2026-10-16.

This document distinguishes a public repository from code that can legally be
copied into this project. Third-party code remains external unless its license
is compatible and reuse is scientifically justified.

| Resource | Useful content | License/status | Planned use |
|---|---|---|---|
| [PALA](https://github.com/AChavignon/PALA) | MATLAB localization pipeline, simulated and in-vivo data, metrics; audited commit `f3e7813` | CC BY-NC-SA 4.0; README also asks for owner agreement | External reference pipeline, 1 kHz timing reference, and simulation benchmark; do not vendor |
| [ULMShare](https://github.com/provostultrasoundlab/ulmshare) | 99 acquisitions; metadata; precomputed compressed trajectories; audited commit `6679f16` | Repository MIT; dataset DOI 10.20383/103.01550 | Velocity-scale audit and lightweight tracks first; selected raw acquisition much later |
| [Open-3DULM](https://github.com/Lab-Imag-Bio/Open-3DULM) | Python 3D ULM pipeline, exported localizations and tracks | CC BY-NC-SA 4.0 | External Python comparator and format reference |
| [TrackingAndLocalizationULM](https://github.com/provostultrasoundlab/TrackingAndLocalizationULM) | Tracking-prior-to-localization ULM workflow | Public repository; verify component/submodule licenses before reuse | Comparator and source of realistic tracking failure cases |
| [deep-ulm-pipeline](https://github.com/tristan-deep/deep-ulm-pipeline) | Python HDF5 localization-to-track pipeline | No license visible in the audited root | Inspect interfaces; do not copy code without permission |
| [RF-ULM](https://github.com/hahnec/rf-ulm) | RF localization model and PALA data interfaces | GPL-3.0; documented missing dataloader | Later RF-level comparator, isolated from core package |
| [Smooth SB](https://github.com/WanliHongC/Smooth_SB) | Particle-tracking tutorials and smooth-SB implementation | No license visible at audited commit `2e0388a`; pinned scientific stack from 2023 | Run as an external reference implementation; do not vendor or modify its source here |
| [Sun–Paninski](https://github.com/SunRuoxi/Single_Particle_Tracking) | Bayesian particle tracking with births/deaths | MIT; Python 2.7 and Keras 2.0.5 | Conceptual baseline; modern reimplementation rather than dependency |
| [TSBM](https://github.com/maxencenoble/twisted-sb-matching) | Differentiable path potentials, PyTorch training code and checkpoints; audited commit `c9dfec7` | MIT | External calibration-tested baseline; current public configuration fails the WP2 sampler gate, so deeper reuse requires method development |
| [Beier et al. UMOT](https://doi.org/10.1007/s10851-022-01126-7) | Unbalanced multi-marginal Sinkhorn formulation | Open-access paper; no exact official implementation identified | Independent small-scale implementation and variable-mass baseline |

## Dataset order

1. Locally generated crossings with exact posterior enumeration
2. PALA synthetic localizations/ground truth
3. ULMShare precomputed trajectories for distributional realism
4. A small selected ULMShare raw acquisition only when an acoustic likelihood is defined

Downloading the complete ULMShare archive is neither necessary nor desirable
for the first paper. Its repository already exposes compressed trajectory files
for rapid exploration without the 30 TB raw archive.

The Smooth-SB interface and reproduction constraints are recorded separately
in `smooth_sb_integration.md`.

