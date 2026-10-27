# WP11ai — Frozen comparison on balanced crowded modes

The fresh WP11ah observation was selected as the **first** passing
variant under a geometry and exact-balance rule committed before its
posterior was screened. It has 17 detections and independently checked
absolute alive-total × frame-six clutter × final clutter joint at
SHA-256
`d0bb9e6496ed33cb784c05f8fd110a7a7b23b9b5ec2e84b98f1c15d91c03a6c2`.
The first screen, geometry ledger and WP11ag exposed/transfer result
are also hash pinned in `wp11ai_balanced_crowded_comparison.py`.
Its high-alive band has 0.18070 mass, only 0.00070 above the frozen
balance threshold, while sixth and final clutter modes both have
substantial alternative mass. This is a targeted independent
development case rather than a representative 19-detection sample.

## Inference and complete replication

Compare the unchanged WP11ag whole-slot candidate to its WP11g
control with four **new matched seeds** 1700–1703, 2,048 particles,
adaptive tempering and eight plus eight sweeps. The candidate retains
the exact conditional full-slot move on 20% of paired proposals at
beta >=0.65; the control has no such move. Run all eight scheduled
control/candidate jobs, even if an early result is poor. All four
BLAS/OMP variables must equal one. Each process has a 750 CPU-second
cap. A cap, hash or numerical mismatch stops the comparison.

Reuse the **previously frozen WP11aa four-run decoy gate unchanged**:
at least 3/4 paired full-joint TV wins; candidate median full-joint
TV <=0.12 and <=0.90 times the matched-control median; median
band-by-sixth TV <=0.10, maximum alive-band mass error <=0.08 and
maximum joint-cell error <=0.04; no more than +0.020 median
regression on each secondary metric. Require at least 3/4 candidate
runs *simultaneously* with band-by-sixth TV <=0.10, maximum alive-band
error <=0.08 and absolute clutter-one mass errors <=0.06 at both
frame six and the final frame. Cost must be <=1.50 times control for
every seed, and candidate median process CPU <=1.25 times control
median. Store full 3D joints, stage summaries, evidence errors,
per-seed metrics and CPU times.

The exposed 19-detection sentinel and separate independent transfer
case already passed their pinned WP11ag gate at SHA-256
`727c207b3e9868034cf7bd227fb4a92ca0718d2ee05c242f30183c3a8c423bb4`.
This new case adds a balanced alive/clutter challenge; a pass would
support the move on this frozen development geometry, while a single
17-detection case cannot establish broad calibration. The untouched
WP8 holdout remains closed.

Run `python -m sbulm.wp11ai_balanced_crowded_comparison --case protocol`
to validate the pinned inputs. The eight fixed jobs use `--case run
--arm control|candidate --seed 1700..1703`, followed by `--case
aggregate`. All thresholds, seeds and configurations in this document
and runner are frozen **before inference on WP11ah**.

## Outcome

Sampler comparison pending the next test step.
