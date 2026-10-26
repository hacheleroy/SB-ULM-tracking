# WP11j — Exact ancestor sampling for four-path conditional PG

WP11h found late divergence of coupled alive-count and sixth-frame clutter
modes even though the four-path PG kernel makes global changes. WP11i's
beta-dependent allocation passed the exact small case but failed a separately
frozen death-marginal guardrail. This proposal changes the **conditional PG
kernel**, not that allocation or its gate. The default sampler and constant
10% four-path mixture remain unchanged.

The opt-in kernel uses one forced-move Metropolis step on the retained
particle's ancestor at each frame after the first. For a candidate prefix
`i`, the unnormalized ancestor weight is

\[
\widetilde w_i = w_{t-1}^{i}
  \frac{\gamma_T((x_{1:t-1}^{i},x'_{t:T}))}
       {\gamma_{t-1}(x_{1:t-1}^{i})}.
\]

The proposal chooses uniformly among the other three internal particles and
accepts with probability `min(1, w_candidate / w_current)`. The suffix ratio
is evaluated by summing **all** remaining exact birth/death, finite-set
count, and tempered collapsed Matérn predictive increments, including
incompatible splices with zero weight. Nothing is truncated or estimated.
The forced-MH-within-PGAS construction leaves the exact ancestor conditional
law invariant; see [Lindsten, Jordan and Schön (2014), equations 7 and 38](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf).
The implementation checks suffix ratios against the full enumerated target
minus prefix increments at beta 0, 0.41, and 1 before inference.

## Frozen screens before inference

1. **Exact oracle:** Reuse the unchanged 269-state, two-slot, three-frame
   posterior with observation SHA-256
   `c645fab464abe15579b843a35081078e62983d6fdfd014c7692a88a368dace72`.
   Seeds 620–622 each draw 512 independent exact initial states. Run two
   pure four-path PG or PGAS pair sweeps at beta one, alternating arm order
   PG/PGAS, PGAS/PG, PG/PGAS. All six alive/assigned count-histogram TVs
   must stay within the inherited 99% exact-multinomial envelope 0.0889279;
   mean first-frame association-change rate must exceed PG; PGAS/PG process
   CPU ratio must be at most 2.0 on each seed. A failure stops before the
   new development case.
2. **Independent synthetic development:** Only if the exact oracle passes,
   use trajectory seed 73 and corruption seed 67 with detection probability
   0.64, expected clutter 0.50 per frame, central separation 0.005, at most
   three clutter detections, and staggered truth mask. The observation hash
   is `07f83b43fdd7555921378b43bb05e75d9abc896d8e268384e6e6e3d4e164adc7`;
   frame cardinalities are `[2,3,4,5,1,2,2,2]`. Seeds 530 and 531 use
   512 outer particles, with the unchanged WP11g mixture in PG and PGAS arms
   (10% four-path PG inside its paired component). Arm order is PG/PGAS,
   then PGAS/PG. Each run is a separate one-thread process with a 500 CPU
   second hard cap. On **each** seed, PGAS/PG CPU ratio must be at most 1.50
   and sweep-weighted PG-selection ratio in `[0.80,1.20]`. The cross-seed
   maximum expected-clutter difference and alive-count TV must **both**
   decrease; each of the other five inherited differences may rise by at
   most 0.020. A failure stops before another 8,192-particle test.

## Results and decision

The exact-oracle gate **passed**. All six count TVs were between 0.0334 and
0.0689, below the 0.08893 envelope. Across seeds, the first-frame
association-change rate rose from 0.2962 under PG to 0.4883 under PGAS.
The per-seed PGAS/PG CPU ratios were 0.934, 0.789, and 0.685 on this tiny
case. An additional exhaustive two-frame, two-particle transition check
enumerates all proposals, resampling and ancestor events at beta 0, 0.37,
and 1, and verifies the exact posterior is stationary. The measured
movement is still not an independent-posterior calibration result.

The **new** development screen failed, with status
`stopped_before_high_particle_test`. Both per-seed CPU ratios (1.431 and
1.390) and PG-selection ratios (0.994 and 1.008) passed. One target
discrepancy improved, while the other became larger; missed and death
marginals also exceeded the fixed +0.020 guardrail.

| Cross-seed discrepancy | PG | PGAS | Change |
| --- | ---: | ---: | ---: |
| Maximum expected-clutter marginal | 0.349609 | 0.271484 | -0.078125 |
| Alive-count histogram TV | 0.224609 | 0.318359 | **+0.093750** |
| Maximum existence marginal | 0.130859 | 0.148438 | +0.017578 |
| Maximum missed marginal | 0.080078 | 0.150391 | **+0.070312** |
| Maximum birth marginal | 0.130859 | 0.115234 | -0.015625 |
| Maximum death marginal | 0.064453 | 0.148438 | **+0.083984** |
| Assigned-count histogram TV | 0.154297 | 0.074219 | -0.080078 |

Full protocols and per-run summaries are in
[`results/wp11j_pgas_oracle.json`](../results/wp11j_pgas_oracle.json) and
[`results/wp11j_pgas_development.json`](../results/wp11j_pgas_development.json).
PGAS used 21 and 20 tempering stages versus 19 in both PG controls, so
the CPU ratios include that schedule difference. A small exact-target
movement improvement did not yield stable eight-frame mode weights on
this fresh case. The fixed development criteria are not revised from this
outcome. No 8,192-particle test was run and the untouched holdout remains
closed. This is a recorded experimental option, not a recommended default.
