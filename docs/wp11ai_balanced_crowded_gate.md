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

All eight frozen jobs completed under the 750 process CPU-second cap;
the aggregate is `results/wp11ai_balanced_crowded_comparison.json`
(SHA-256 `1e4f97b6270a2350fc4213c6179d2c435bcb745b66ba1132b82b94692977479b`).
All eight full run records, including the 3D joints and tempering stages,
are in `results/wp11ai_balanced_crowded_runs/`. The frozen aggregate
validated their protocols, configurations, final beta, particle joints
and metrics against the pinned absolute reference.

| Seed | Control joint TV | Candidate joint TV | Control CPU s | Candidate CPU s | Candidate replicated mode limits |
| --- | ---: | ---: | ---: | ---: | :---: |
| 1700 | 0.23779 | 0.09989 | 256.7 | 251.1 | pass |
| 1701 | 0.19734 | 0.07575 | 253.1 | 245.8 | pass |
| 1702 | 0.09841 | 0.07079 | 257.7 | 242.5 | pass |
| 1703 | 0.10329 | 0.07989 | 252.2 | 250.6 | pass |

The candidate won **4/4** matched full-joint comparisons and met all
four simultaneous mode limits (**4/4**, against a 3/4 requirement).
Median full-joint TV fell from 0.15031 to 0.07782 (candidate/control
ratio 0.518, below the 0.90 limit and 0.12 absolute cap). Median
band-by-sixth TV was 0.03073 (cap 0.10), median maximum alive-band
error 0.02421 (cap 0.08), and median maximum cell error 0.01293
(cap 0.04). All four secondary metric regressions were <=0.020;
the final clutter mean absolute error increased from 0.00952 to
0.01538, a 0.00586 regression within the frozen limit. Every paired
CPU ratio was <=0.994 (cap 1.50); the candidate/control median CPU
ratio was 0.974 (cap 1.25). All eight component gates passed.

The status is `independent_balanced_crowded_gate_passed_holdout_closed`.
This supports WP11ag on this independently frozen balanced 17-detection
development geometry as well as the previously exposed 19-detection
and independent transfer cases. The high-alive mass passed its geometry
screen by only 0.00070, and one case does not establish calibration
across crowded geometries. The WP8 holdout remains unopened. The next
study should predeclare a broader set of crowded absolute observations,
including a balanced case closer to 19 detections, and its complete
replication rule before sampler inference.
