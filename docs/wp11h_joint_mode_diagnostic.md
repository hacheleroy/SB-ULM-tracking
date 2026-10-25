# WP11h — Joint-mode diagnosis on the exposed WP8 observation

WP11g reduced all seven two-seed discrepancies against the archived control,
but expected frame-six clutter differed by 0.1335 and the total alive-count
histograms had TV 0.1171, above the frozen 0.10 limit. Its saved checkpoint
contains marginal summaries and the five most frequent count signatures,
not the joint particle population. Separate marginals cannot identify the
competing complete hypotheses.

This exploratory diagnosis reproduces the **same** exposed WP8 observation,
four-path 10% PG mixture, 8,192 particles, and seeds 100 and 200. It uses the
original 1,800-process-CPU-second cap per seed and no new observation or seed.
The original WP11g checkpoint is SHA-256 pinned. An optional observer records
only counts and never consumes random numbers or modifies the sampler. Every
non-timing field of each reproduced WP11g summary must agree exactly before
the new diagnostic is saved.

At each annealing stage the observer records a label-invariant table of total
alive slot-frames by sixth-frame clutter count, plus the weighted-to-resampled
joint-table difference. Local and PG pair proposals have separate accepted
state/count-change tallies, per-frame alive and assigned changes, frame-six
clutter changes, and transitions among total-alive bands (0–20, 21, and
22–32). Each run also keeps the full per-particle label-invariant vectors of
alive and assigned counts at all eight frames, compressed with zlib and
checksummed. The exact synthetic association arrays are also retained in
compressed, checksummed form; this allows an analysis of which detections
are clutter and of complete slot-permutation-invariant histories. These
records distinguish a marginal shift from coupled mode shifts and show
when the seed populations diverge. Full-history empirical TV is reported
only descriptively because its support is sparse at this sample size.

These are **post-hoc development diagnostics**, not a revised WP11g gate or
a posterior-calibration measurement. In particular, pair-proposal activity
alone does not identify stationary mode masses. The untouched holdout remains
closed. Results and an interpretation will be added after both frozen seed
reproductions finish.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11h_joint_mode_diagnostic --seed 100
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11h_joint_mode_diagnostic --seed 200
```
