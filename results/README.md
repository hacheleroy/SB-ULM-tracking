# Generated benchmark outputs

All files in this directory use deterministic seeds and are intentionally
versioned with the benchmark.

| Output | Reproduction |
|---|---|
| `crossing_summary.json`, `crossing_example.png` | `python -m sbulm.experiment --output-dir results --n-seeds 32` |
| `difficulty_map.csv/.json/.png` | `python -m sbulm.difficulty --output-dir results --n-seeds 16 --workers 4` |
| `sbulm_crossing_smooth_sb.npz` | `python -m sbulm.smooth_sb --output results/sbulm_crossing_smooth_sb.npz --seed 0` |
| `smooth_sb_smoke_summary.json` | External Smooth-SB validation described in `../docs/smooth_sb_integration.md` |
| `physical_matern_calibration.json`, `physical_matern_calibration_map.csv/.png`, `physical_matern_speed_sensitivity.csv/.png` | `python -m sbulm.physical_sweep --output-dir results --n-seeds 16 --sensitivity-seeds 32 --workers 6` |
| `wp2_likelihood_map.csv/.png`, `wp2_credible_coverage.csv/.png`, `wp2_localization_likelihood.json` | `python -m sbulm.wp2_likelihood --output-dir results --n-seeds 64 --workers 6` |
| `tsbm_wp2_calibration.json/.png` | `python experiments/summarize_tsbm_wp2.py --run LABEL=path/to/summary.json [...] --output-dir results` after the external runs documented in `../docs/tsbm_wp2_integration.md` |
| `smc_matern_calibration.json/.png`, `smc_matern_calibration_map.csv/.png` | `python -m sbulm.smc_calibration --output-dir results --n-particles 8192 --n-smc-repeats 3 --map-n-particles 4096 --map-n-data-seeds 4 --mh-sweeps 8 --final-mh-sweeps 8 --workers 6` |
| `wp3_null_calibration.json/.png`, `wp3_null_calibration_map.csv/.png` | `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m sbulm.wp3_null_calibration --output-dir results --n-particles 8192 --n-smc-repeats 3 --map-n-particles 4096 --map-n-data-seeds 4 --mh-sweeps 10 --final-mh-sweeps 10 --workers 6` |
| `wp4_existence_calibration.json/.png` | `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m sbulm.wp4_existence_calibration --output-dir results --n-particles 8192 --n-smc-repeats 3 --mh-sweeps 10 --final-mh-sweeps 10 --workers 6` |
| `wp4_evidence_escalation.json` | Targeted persistent-near-clean seed-2 rerun with 16,384 particles, retaining all other WP4 controls |
| `wp5_scaling_calibration.json/.png` | `python -m sbulm.wp5_scaling_calibration --output-dir results --n-particles 2048 --n-repeats 3 --baseline-replicates 100` |
| `wp6_paired_scaling.json/.png` | `python -m sbulm.wp6_paired_scaling --output-dir results --n-particles 2048 --n-repeats 3 --baseline-replicates 100` |
| `wp6_paired_heldout.json` | `python -m sbulm.wp6_paired_scaling --output-dir results --n-particles 2048 --n-repeats 3 --heldout-only --heldout-seed-start 200` |
| `wp7_convergence_audit.json` | `python -m sbulm.wp7_convergence_audit --bridge-only` followed by `python -m sbulm.wp7_convergence_audit --large-only --large-particles 4096 8192` (checkpointed) |

The external summary records aggregate and per-seed metrics, upstream commit,
parameters, and indicative runtime. Raw upstream messages and samples are not
distributed because the audited Smooth-SB repository does not state a license;
the ULM input and the exact target enumerator are sufficient to repeat the run.

The WP2 outputs compare two exact enumerable targets using the same frozen
Matérn prior and paired observations. Only the localization observation model
changes. The reported credible-set frequencies are reference-path inclusion on
the controlled simulator, not a general claim of posterior calibration.

The TSBM calibration summary retains only compact metrics and the comparison
figure. Its discrete TV/JS values use Rao--Blackwellized association
probabilities; hard nearest-assignment projections are used only for mode-visit
and support diagnostics. Raw neural checkpoints, logs, and sample arrays are
working artifacts and are intentionally excluded from version control. The
consolidated result is a no-go for downstream use of the current public TSBM
configuration, not a claim that the TSBM theory cannot be adapted.

The SMC JSON records every particle-calibration run, adaptive temperature,
ESS, mutation acceptance rate, finite-sample exact reference, and gate check.
Exact enumeration is used by this harness only for evaluation; the SMC
inference routine scores proposed associations without enumerating the support.
Its positive fixed-cardinality result and remaining scalability boundary are
documented in `../docs/smc_matern_calibration.md`.

The WP3 null-association JSON records exact and sampled partial-association
weights, missed-detection probabilities, expected clutter counts, continuous
path distances, evidence estimates, and every SMC temperature. The first frame
is complete and clutter-free to anchor labels. The calibration simulator caps
clutter only to keep exact enumeration finite; inference uses the stated
untruncated Poisson point-process likelihood. See
`../docs/wp3_null_association.md`.

The WP4 JSON extends the exact state to include a contiguous existence mask
for each exchangeable slot. It records full mask-and-association comparisons,
existence/miss/clutter marginals, birth and last-alive distributions, masked
continuous paths, and evidence estimates. All posterior and event-marginal
checks pass in every run; the strict flag remains false because one run also
exposes finite-sample mass and single-run evidence boundaries. The targeted
particle escalation is stored separately so the original frozen runs remain
auditable. See `../docs/wp4_birth_death_slots.md`.

WP6 reruns the unchanged WP5 blocked control beside the likelihood-informed
paired move, on the same exact three-slot cases and four-slot data. Both
methods pass all exact-oracle runs. The paired method substantially improves
four-slot agreement but misses the frozen 0.10 stability threshold by 0.00010
on alive-count TV; a second untouched seed triplet passes within-triplet
checks. Neither result is an exact four-slot calibration claim. See
`../docs/wp6_paired_moves.md`.

WP7 adds an exactly enumerable, sparse four-slot/four-frame bridge; three
independent paired-sampler runs pass its inherited finite-draw and marginal
checks. On the unchanged four-slot/eight-frame case, the original three seeds
pass at 4,096 particles, but the held-out triplet and combined six runs fail
the 0.10 stability threshold. At 8,192 particles, all seven metrics pass over
the combined six independent runs (worst 0.0557). This is stability evidence
for the frozen synthetic case, not an exact eight-frame posterior oracle or a
transfer claim. See `../docs/wp7_convergence_audit.md`.
