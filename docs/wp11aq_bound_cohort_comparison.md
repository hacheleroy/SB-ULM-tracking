# WP11aq — WP11al rule bound to the WP11ap absolute cohort

This executable binding is committed before the first sampler job. The
previously frozen WP11al rule is `docs/wp11al_conditional_cohort_gate.md`
(SHA-256 `693a34ae0a5a3da8f149e9087c8ee38441b748b1845596905a7a9b5afe2592ad`).
The validated WP11ap manifest is
`results/wp11ap_crowded_cohort_manifest.json` (SHA-256
`2e2e20acbb84524e5022d9eaa7ab40202420bad660214426dbcf0ecfca000e68`).
`src/sbulm/wp11aq_bound_cohort_comparison.py` checks the manifest,
geometry, selected screen and reference bytes against their pinned
hashes and regenerates each observation before a job can start.

The 18-detection index 36 gets matched seeds 1800–1803. The first
19-detection index 207 gets 1810–1813, and the second 19-detection
index 9 gets 1820–1823. Within each seed, control precedes candidate.
Each run uses the unchanged WP11g/WP11ag configuration: 2,048
particles, adaptive tempering, eight plus eight sweeps, and 20% exact
whole-slot proposals for the candidate at beta at least `.65`. All
four threading variables must equal one, and each job has its own
750-process-CPU-second limit. A hash, configuration, numerical or CPU
failure stops the comparison. Poor accuracy in an otherwise sound run
does not stop the prescribed 24 jobs.

The evaluator reuses the frozen WP11aa four-run decoy gate separately
for all three cases, including its absolute, relative, regression and
simultaneous mode limits. It adds WP11al's median CPU ratio limit
`1.25` to the per-seed `1.50` limit, then requires all three case gates
and at least 10/12 pooled full-joint TV wins and 10/12 pooled
simultaneous mode successes. The exact joint, stage summaries,
evidence error, timing and error metrics are stored per job. A decision
cannot be made from a partial cohort.

The references validate the four-slot arithmetic numerically under a
shared one-slot likelihood; they do not provide an interval proof or
an independent likelihood derivation. The WP8 holdout stays closed.

**Freeze checkpoint:** no WP11aq sampler job has run at this commit.
