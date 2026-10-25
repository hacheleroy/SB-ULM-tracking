# WP11f — Controlled cost, cheaper PG paths, and new development case

WP11e's sparse conditional particle-Gibbs (PG) mixture changed more
label-invariant count modes, but its single-run wall-time cost exceeded
the frozen 3× limit on one seed. That failure is **not overwritten** here.
WP11f separately asks whether the measured cost is stable under an
interleaved CPU-time protocol, whether four rather than eight internal
path particles retain the exact small-case target, and whether the
cheaper mixture has operational value on a *new development* observation.

## Interleaved cost check on exposed WP11e data

The exact WP11e observation and baseline association hashes at seeds 310
and 311 were reproduced. On each identical 512-particle starting
population, local and 10%-PG pair updates (eight internal paths) were
alternated in four repeats, with the same repeat RNG seed within each
arm. BLAS/OMP thread counts were fixed to one. The predeclared gate used
median **process CPU time**, at most 3× local on each seed, while
retaining wall time and actual count movement. The protocol and all
individual measurements are in
[`results/wp11f_interleaved_cost.json`](../results/wp11f_interleaved_cost.json).

| Seed | Median CPU ratio | Median wall ratio | Sparse/local count-change rate |
|---|---:|---:|---:|
| 310 | 2.262× | 2.986× | 0.2390 / 0.1931 |
| 311 | 2.609× | 3.648× | 0.2449 / 0.1924 |

The controlled cost gate passes. Individual wall ratios range from 2.45
to 10.30; seed 311 also has one CPU outlier of 5.04, so the result is
not a claim of invariant runtime across hardware or loads. The prior
single-run WP11e wall-time gate remains failed; this is a distinct,
predeclared diagnostic using the *already exposed* data.

## Four versus eight paths on the enumerable oracle

Before another synthetic case was used, the unchanged 269-state
two-slot/three-frame exact posterior was sampled from exact-posterior
starts. Four and eight internal PG path particles were compared at seeds
81–83, with 512 starts, two sweeps, beta 1, and alternating arm order.
The predeclared gate required every label-free count-TV below the 99%
finite-sample maximum-of-three envelope (0.0889), a mean four-path count
change rate at least 0.20, and median four/eight CPU ratio at most 0.75.
All six count-TV values pass; four paths produce a mean count change
of **0.2751** (less than the eight-path mean of roughly 0.404), at a
median CPU ratio of **0.632**. The complete frozen result is in
[`results/wp11f_path_count_oracle.json`](../results/wp11f_path_count_oracle.json).
The exact incremental Gaussian-density and two-particle detailed-balance
tests are unchanged; this pilot adds finite-sample stationarity/movement
evidence for the lower path count, not an exhaustive four-particle
transition-matrix proof.

## New four-slot/eight-frame development observation

Only after those two screens passed, a *new development* case was
frozen: detection probability 0.58, expected clutter 0.30 per frame,
central separation 0.005 mm, trajectory seed 53, corruption seed 37,
at most two clutter detections per frame, and a staggered birth/death
schedule. Its cardinalities are `[0, 3, 2, 3, 2, 3, 2, 1]`; its
observation SHA-256 is
`7c211e3cfeac33e2e2487f5bcbaec6c204148668954bddbfffbfaec0537c1dd4`.
This is neither the WP8 nor WP11e observation and is **not a holdout**.

The original paired annealed SMC generated approximate posterior starts
with 512 particles and seeds 410 and 411. Each arm received two
pair-only sweeps from identical starts, repeated twice with reversed arm
order. A state-independent 10% of pair moves used conditional PG with
four internal paths; the rest used the original local pair move.
The frozen advancement gate required greater or equal mean count
movement, observed PG selection in `[0.07,0.13]`, and at most 2× median
paired CPU cost on *each* seed.

| Seed | Sparse/local count-change rate | PG fraction | Median CPU ratio |
|---|---:|---:|---:|
| 410 | 0.2173 / 0.1904 | 0.1050 | 1.567× |
| 411 | 0.2246 / 0.1777 | 0.1040 | 1.868× |

Both seeds pass that **development operational gate**. The cross-seed
maximum assigned-count TV falls from 0.1055 (local) to 0.0820, and the
expected-clutter difference from 0.1133 to 0.0820. However, alive-count
TV **rises** from 0.2080 to 0.2227. This is not a seven-metric stability
pass or a claim that posterior mode masses are correct. The baseline
annealed SMC runs themselves took 193.2 and 150.8 seconds on this
shared node. The protocol, state hashes, per-repeat diagnostics, and
full mixed outcome are in
[`results/wp11f_four_path_development.json`](../results/wp11f_four_path_development.json).

The next defensible larger experiment would freeze the four-path 10%
mixture and compare independent 8,192-particle runs on an **already
exposed** difficult observation using the inherited seven label-free
stability metrics and an explicit compute budget. Cross-run stability
there would still not prove eight-frame exact calibration or transfer
to PALA/ULMShare. A truly untouched synthetic holdout remains closed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11f_interleaved_cost
python -m sbulm.wp11f_path_count_oracle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m sbulm.wp11f_four_path_development
```
