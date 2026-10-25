# WP11d — Development screen on the unchanged difficult WP8 observations

## Profile and semantics-preserving optimization

A 32-row, eight-path-particle profile on the **already exposed** WP8
`cluttered_staggered_crossing` observation identified unnecessary HMM
forward/backward construction in the conditional particle-Gibbs (PG) move
and repeated Gaussian-predictive calculations for identical resampled
histories. A lightweight pair-frame support class and per-update caches
remove that work without changing the proposal distribution. Under
`cProfile`, this diagnostic went from 4.05 to 1.33 seconds (shared host;
not a controlled performance benchmark). Existing tests, including the
enumerated two-frame detailed-balance matrix at three tempering values,
still pass.

## Frozen development comparison

The observation hash is the original WP8 failed case,
`afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae`.
No new realization or held-out seed is used. A baseline WP6 paired annealed
SMC run with 512 particles, eight stage and final sweeps, and seeds 100 and
200 supplies approximate posterior starts. Both arms begin from *identical*
states within each seed. At `beta=1` they apply two sweeps consisting only
of either the optimized conditional-PG pair move (eight internal path
particles) or the unchanged local pair move. The predeclared operational
screen required at least as many actual label-invariant count-signature
changes and at most 10× pair-update runtime on **each** seed. The full
configuration, state hashes, histograms and timings are in
[`results/wp11d_wp8_development.json`](../results/wp11d_wp8_development.json).

| Seed | PG count changes | Local count changes | PG/local runtime | PG/local seconds |
|---|---:|---:|---:|---:|
| 100 | 0.7080 | 0.1992 | 15.7× | 27.6 / 1.8 |
| 200 | 0.6611 | 0.1650 | 18.9× | 24.9 / 1.3 |

Between-seed discrepancies decrease, but remain large:

| Label-free summary | Conditional PG | Local pair |
|---|---:|---:|
| Maximum alive-count TV | 0.1719 | 0.4180 |
| Maximum assigned-count TV | 0.1406 | 0.2637 |
| Maximum expected-clutter difference | 0.1914 | 0.2988 |

The baseline 512-particle SMC runs took 59.7 and 102.6 seconds and each
used 19 tempering stages. The diagnostic clock times vary on a shared
host. Although the global update demonstrably moves more and narrows these
two-seed discrepancies, **both** runtime ratios exceed the frozen 10×
limit. The status is `stopped_before_large_inference`. Even its reduced
cross-seed discrepancies exceed 0.10, but the inherited 0.10 gate belongs
to a different 8,192-particle, six-seed protocol and is **not** being
applied or claimed here.

These are short transitions from an approximate baseline posterior, not
independent exact-posterior draws. A state-change rate is not an effective
sample size or a proof that rare modes have correct mass. The next method
step should reduce per-frame exact predictive/proposal costs or use the
global PG update selectively within a frozen mixture, then re-check the
small oracle and an explicitly new *development* protocol before booking
another expensive high-particle comparison. Neither a fresh holdout nor
an in-vivo/ULMShare dataset has been opened.

```bash
python -m sbulm.wp11d_wp8_development
pytest tests/test_wp11d_wp8_development.py tests/test_conditional_pair_pg.py
```
