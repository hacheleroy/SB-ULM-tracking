# WP11ag — Frozen whole-slot Gibbs comparison before inference

The WP11ab exposed primary still had three seeds with large low-to-high
alive-band mass errors. The WP11ac retrospective decomposition showed
that the alive-band marginal alone accounts for 64–95% of full-joint
TV in those three seeds. A window restricted to one or two late frames
cannot directly redraw a whole birth-to-death interval. This motivates
one **distinct** move: sample the complete existence, miss, assignment
and clutter consequences of one uniformly chosen slot from its exact
finite conditional given the other slots. The reference selection was
independent of this move and completed before candidate inference.

## Target-preserving move and schedule

`WholeSlotGibbs` enumerates one-slot paths and caches their interval
prior, finite-set count score and collapsed Matérn log likelihood.
For a current particle, paths that reuse a detection held by any other
slot receive zero probability. The remaining normalized scores are
the exact tempered conditional; a uniformly selected slot is redrawn
without a Metropolis correction. The finite support includes the
never-born state and every contiguous interval with all compatible
misses and assignments. An eight-frame score check on the independently
selected case agreed with direct full-model conditional log ratios to
1.1e-14; small direct enumeration checks the whole conditional and
stationary tempered histogram.

At beta below 0.65, matched-seed transitions are bitwise identical to
the archived WP11g control. At beta >=0.65, 20% of **paired** proposals
become whole-slot Gibbs, 10% remain four-particle full-path PG and 70%
remain local two-slot multiple-try. Thus at most 5% of all proposals
select the new move, because paired proposals themselves are 25% of
the sweep. Keep 2,048 particles, the other move proportions, adaptive
tempering, and eight plus eight sweeps unchanged. One-slot score tables
contain 29,568 paths on the 19-detection primary and 15,445 on the
new transfer case. This is a temperature-only mixture of invariant
kernels, without state-dependent selection.

## Pinned observations, runs and gates

Use the archived, independently exact 19-detection primary reference
and its unchanged six controls at seeds 1300–1305. Also use the new
geometry-selected, independently validated 17-detection WP11af joint
at SHA-256
`cd6f20f821a2689cda5a1a0aceb19fa9db7c6232af0e8e05b6262e312469f87b`.
Run four new matched control/candidate pairs at seeds 1600–1603 on that
case. All 14 scheduled inference runs must be completed; do not stop
when a partial result looks poor or replace a seed. Require all four
BLAS/OMP thread variables equal to one and a 750 process-CPU-second
cap per run. A cap, hash or numerical mismatch stops the comparison.

The **19-detection accuracy and replication gate** is the WP11aa
primary gate unchanged: at least 4/6 paired full-joint TV wins;
candidate medians <=0.12 full joint, <=0.10 alive-band by frame-six
clutter TV and <=0.08 maximum alive-band mass error; each median
<=0.80 of its archived control; and no more than +0.020 median
regression on each secondary metric.

The **new transfer gate** uses the WP11aa four-run decoy limits:
at least 3/4 paired full-joint wins; median full joint <=0.12 and
<=0.90 of new matched controls; median band by sixth <=0.10,
maximum alive-band error <=0.08 and maximum cell error <=0.04;
no more than +0.020 median regression on secondary metrics; and
at least 3/4 runs simultaneously with band by sixth <=0.10,
maximum alive-band error <=0.08, and absolute clutter-one mass errors
<=0.06 in both frame six and the final frame.

**Cost applies separately to both observations:** every candidate /
matched control process CPU ratio <=1.50, and candidate median CPU
<=1.25 times control median. The primary controls are archived;
transfer controls are new fixed-seed runs. Both full case gates must
pass to continue. Store final 3D joints, exact errors, per-seed CPU
times, evidence errors and stage summaries. A pass would establish
exposed-development plus independent **transfer** evidence only; it
cannot establish calibrated crowded count modes without a newly
balanced absolute reference. The untouched WP8 holdout remains closed.

Run `python -m sbulm.wp11ag_global_slot_comparison --case protocol`
to verify inputs, then `--case run --observation primary_19 --arm candidate
--seed 1300` and the remaining fixed arms and seeds, followed by
`--case aggregate`. The runner rejects any schedule or reference
mismatch. Thresholds, mixture rate, temperature switch and seeds are
frozen in this commit **before candidate inference**.

## Outcome

All 14 frozen jobs completed under the 750 process-CPU-second cap. The
unmodified aggregator returned
`exposed_and_independent_transfer_gates_passed_needs_new_crowded_reference`.
The complete joints, stage diagnostics, per-seed metrics and CPU times
are in `results/wp11ag_global_slot_runs/`; the protocol-validated
aggregate is `results/wp11ag_global_slot_comparison.json` (SHA-256
`727c207b3e9868034cf7bd227fb4a92ca0718d2ee05c242f30183c3a8c423bb4`).

| Case | Paired full-joint TV wins | Median full-joint TV, control → candidate | Median band-by-sixth TV, control → candidate | Median maximum band-mass error, control → candidate | Median CPU, control → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exposed 19 detections | 6/6 | 0.19676 → 0.07378 | 0.16060 → 0.04572 | 0.10234 → 0.02246 | 238.75 → 258.98 s |
| Independent transfer, 17 detections | 4/4 | 0.07923 → 0.06200 | 0.03302 → 0.01946 | 0.02921 → 0.01634 | 241.01 → 236.34 s |

Every individual gate passed. The largest per-seed candidate/control
CPU ratio was 1.205 on the primary and 0.987 on transfer; median ratios
were 1.085 and 0.981. All four transfer candidate runs met the joint
alive/clutter mode limits. Primary seed 1305 improved full-joint TV
(0.11678 → 0.10188) but worsened band-by-sixth TV
(0.06686 → 0.08538), an individual exception to the median improvement.
The whole-slot selection fraction averaged about 20% of paired proposals
at beta >=0.65 and zero earlier. Its count-signature change rate was
about 90% on the primary and 93% on transfer. This records changes to
the count/clutter signature, not a direct measurement of between-band
crossings or a proof of why the sampler improved. Log-evidence errors
were -1.43 to -0.33 on the primary and -0.23 to +0.44 on transfer;
evidence accuracy was recorded, not a criterion of this posterior gate.

This is a pass on the exposed 19-detection sentinel and an independently
selected absolute transfer case. The transfer exact posterior is heavily
weighted toward one frame-six clutter state and has unbalanced alive
bands; it is **not** an independent crowded-mode validation. Before any
calibration or generalization claim, freeze and validate a fresh
balanced crowded absolute reference and its comparison without looking
at the untouched WP8 holdout.
