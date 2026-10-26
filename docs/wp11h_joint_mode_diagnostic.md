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

Both seed reproductions completed under the cap (1,296 and 1,124 process CPU
seconds). Every non-timing WP11g summary field agrees exactly with its
archived counterpart. The aggregate results and per-stage summaries are in
[`results/wp11h_joint_mode_summary.json`](../results/wp11h_joint_mode_summary.json).
The complete synthetic per-particle archive is retained privately, with
SHA-256 `b92ed2e6e0379f9fe54bd09d7f9771c5b5f4566818fcfece87becb840ff9de67`;
the runner regenerates both the full record and this aggregate summary.

At beta 1, seed 100 places 0.634 of its mass on hypotheses with at most 20
total alive slot-frames, versus 0.522 for seed 200. Conversely the masses
above 21 are 0.260 and 0.362. The frame-six expected-clutter gap of 0.1335
decomposes into approximately **0.0783 from these different alive-band
weights** and **0.0553 from different clutter conditional on the bands**,
using a symmetric composition/within-band decomposition. The per-frame
alive differences (seed 200 minus seed 100) are largest in the sixth to
eighth frames: 0.174, 0.119, and 0.109. The earlier four frames are much
closer.

The residual also includes detection identity. The frame-six clutter-subset
distributions have descriptive TV 0.1475. For example, detections 2 and 3
(zero-based indexes) are jointly clutter in 0.1138 of seed-100 particles and
0.1660 of seed-200 particles; detection 2 alone is clutter in 0.0319 and
0.0745. Per-frame alive vectors differ in descriptive TV 0.344, while
per-frame assigned vectors differ by 0.178. The complete-history empirical
TV is 0.996 because almost every distinct history is rare; it is **not** a
calibrated or useful stability gate at this sample size.

The two stage populations are similar early and separate mainly after beta
about 0.75. At the fifteenth stage (beta about 0.759 in both runs), the
alive-total TV is 0.061 and the frame-six clutter-mean difference is 0.095;
at beta 1 they reach 0.117 and 0.134. The beta sequences are not exactly
aligned at every stage, so this is an exploratory trend, not a controlled
stage-by-stage effect. The weighted-to-resampled joint-table TV at each of
the last five stages is about 0.01, which does not by itself establish that
resampling is harmless over the full sequence of stages.

The global move does cross these modes. At beta 1, PG changed frame-six
clutter on 446/3,418 selected updates in seed 100 and 458/3,330 in seed
200; the local pair move did so on 415/29,600 and 428/29,290. Direct jumps
from the 0–20 to 22–32 alive bands were 114 and 95 for PG, versus 36 and
26 locally. These are *proposal activity* measurements, not independent
posterior samples or a measure of equilibrium. They show that complete
path-locking is not established by this case, while the final mode weights
still fail to agree.

**Decision.** The next bounded method test should reallocate a similar
number of four-path PG updates toward the late tempering stages, where the
seed differences grow, rather than immediately introduce ancestor sampling.
A beta-only schedule is independent of the current state and retains the
invariant kernel mixture at each fixed beta. Freeze a near-budget-neutral
schedule, check exact small-case behavior and measured CPU cost, then try it
on a new development observation before repeating an exposed 8,192-particle
test. If that fails despite more late mode transitions, an exact ancestor
sampling or joint temporal bridge can be developed against the small oracle.

These are **post-hoc development diagnostics**, not a revised WP11g gate or
a posterior-calibration measurement. The untouched holdout remains closed.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11h_joint_mode_diagnostic --seed 100
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11h_joint_mode_diagnostic --seed 200
```
