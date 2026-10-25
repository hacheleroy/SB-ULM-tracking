# WP11e — Sparse global pair updates on a new synthetic development case

WP11d showed that a full-history, likelihood-guided pair particle-Gibbs
(PG) update makes substantial count-mode transitions but is too costly
when applied to every selected pair. WP11e keeps the exact local pair
move and selects PG with a fixed, state-independent probability of 0.10
*within* the pair-move component. This convex mixture preserves the
tempered posterior because both component kernels do; the default
`paired_slot_scope="local"` and its random-number path are unchanged.
At fraction zero the opt-in mixture is bit-for-bit equal to the default.
The existing two-frame exhaustive detailed-balance checks cover PG, and
the local pair has its own oracle checks; a sparse-mixture exact-posterior
sample also passes its count-histogram check. The fraction is chosen
independently of the current state, not adaptively from its likelihood.

## Frozen development protocol

Before inference, a *new development* eight-frame four-slot synthetic
case was generated with detection probability 0.62, expected clutter 0.35
per frame, 4-µm central separation, trajectory seed 41, corruption seed
22, and the staggered birth/death schedule. Its detection counts are
`[2, 2, 5, 4, 1, 3, 0, 2]`, with SHA-256
`9665d3a3377254df2211d251ad6ae746d8880b7501f1938ae4a2886afcfd3d37`.
This is not a reserved holdout.

The original WP6 paired annealed SMC creates 512-particle approximate
posterior starting populations at seeds 310 and 311. Within each seed,
both arms start from identical association arrays and use two pair-only
rejuvenation sweeps at beta 1. The new arm uses PG for 10% of the pair
updates (eight internal path particles); the control uses only local pair
updates. The predeclared screen requires, on *each* seed, at least the
local arm's actual count-signature change rate, an observed PG fraction in
`[0.06, 0.14]`, and at most **3×** the local pair-update wall time. The
protocol, association hashes, counts, and timings are saved in
[`results/wp11e_selective_pg_development.json`](../results/wp11e_selective_pg_development.json).

| Seed | Sparse PG count changes | Local count changes | Observed PG fraction | Sparse/local time |
|---|---:|---:|---:|---:|
| 310 | 0.2461 | 0.2080 | 0.1123 | 1.26× (9.9 / 7.9 s) |
| 311 | 0.2441 | 0.1875 | 0.1025 | 4.19× (10.5 / 2.5 s) |

The conditional-PG subset itself changed count signatures in 0.687 and
0.733 of its selected updates. The overall sparse-mixture count movement
improves on both seeds, but the second runtime ratio exceeds the frozen
3× ceiling. Thus the recorded status is **`stopped_before_holdout`**.
The baseline SMC runs took 159.7 and 121.1 seconds (19 and 20 stages),
respectively. On the shared host, local-arm wall times differed by over
threefold; this is a reason to design a controlled cost protocol next,
not grounds to waive the failed gate retrospectively.

The between-seed maximum alive-count TV is 0.2500 for sparse PG versus
0.2598 locally; assigned-count TV is 0.1172 versus 0.1191; maximum
expected-clutter difference is 0.1230 versus 0.1191. These small and
mixed changes do not support a cross-run stability claim. The samples
are short updates from an approximate baseline distribution, not exact
eight-frame posterior draws; state-change counts alone are not a measure
of correct mode mass. No 8,192-particle inference, reserved synthetic
observation, PALA, or ULMShare data were opened.

The next bounded task should use an interleaved, repeated CPU-time
comparison with the same starting states, then assess whether a cheaper
PG path-particle count or additional caching preserves the small exact
oracle before freezing another *new development* observation. An
independent holdout remains reserved until a practical sampler passes.

```bash
python -m sbulm.wp11e_selective_pg_development
pytest tests/test_conditional_pair_pg.py tests/test_wp11e_selective_pg_development.py
```
