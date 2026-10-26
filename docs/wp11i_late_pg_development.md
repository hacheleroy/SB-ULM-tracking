# WP11i — Reallocate four-path PG updates to late tempering

The exposed WP11h diagnostic found that the two 8,192-particle runs separate
mostly after beta about 0.75. The existing four-path PG update frequently
crosses count and frame-six clutter modes, so a budget-neutral change in
*when* it is applied is the first candidate. This is exploratory method
development and does not revise the failed WP11g gate.

At fixed beta, the pair kernel is

\[
K_\beta=(1-p(\beta))K^{\mathrm{local}}_\beta
       +p(\beta)K^{\mathrm{PG}}_\beta.
\]

Both component kernels leave the tempered target invariant; the selection
probability depends only on beta, never on the current association state.
The default constant 10% allocation remains unchanged. The opt-in candidate
sets `p=0.025` below beta 0.75 and `p=0.25` at or above it. On the twenty
WP11h stages, including twice as many final-stage sweeps, this gives exactly
the same *expected* total number of PG selections as a constant 10%.
Actual selections and process CPU cost are measured on the new case.

The first gate uses the unchanged 269-state exact small posterior, three
frozen seeds (610–612), 512 initial exact draws per seed, two pair sweeps,
and the inherited 99% exact-multinomial count-TV envelope 0.08893. All six
run TVs must be inside the envelope and the candidate's mean count-mode
movement must exceed the constant mixture. Results are in
[`results/wp11i_late_pg_oracle.json`](../results/wp11i_late_pg_oracle.json).

The second gate uses a **new** eight-frame synthetic development observation:
trajectory seed 59, corruption seed 43, detection probability 0.63,
expected clutter 0.40 per frame, at most three clutter detections, and
observation hash
`ae7d3cfbce756ecc8368d9a152d4eee0f1f5697d75a9732dd6fc4408f1e0b34c`.
Its detection counts are `[1,3,3,3,3,3,3,2]`. Independent seeds 510 and
511 use 512 outer particles and the unchanged WP11g move mixture; each seed
has constant and late arms, ordered constant/late then late/constant. Each
run has a 400-process-CPU-second hard cap. The candidate must cost at most
1.30 times control and have a sweep-weighted PG selection ratio in
`[0.80,1.20]` on **each** seed. The cross-seed expected-clutter maximum and
alive-count TV must each improve, and none of the other five inherited
metrics may rise by more than 0.02. This is a development screen, not an
eight-frame posterior-stability test or a new holdout.

The exact-oracle gate **passed**. All six count TVs were below 0.08893; the
largest was 0.07186. Mean count-signature change per proposal increased from
0.15820 for the constant mixture to 0.17839 for the late allocation. This
checks target preservation at beta one on the small enumerated case and
movement, not eight-frame calibration.

The fresh development screen **failed its frozen guardrail**, with status
`stopped_before_high_particle_test`. Both target discrepancies decreased,
but the maximum death marginal difference rose by 0.021484, just over the
allowed 0.020. The outcome is recorded without relaxing that tolerance.

| Cross-seed discrepancy | Constant 10% | Late allocation | Change |
| --- | ---: | ---: | ---: |
| Maximum expected-clutter marginal | 0.134766 | 0.064453 | -0.070312 |
| Alive-count histogram TV | 0.197266 | 0.152344 | -0.044922 |
| Maximum existence marginal | 0.091797 | 0.107422 | +0.015625 |
| Maximum missed marginal | 0.140625 | 0.093750 | -0.046875 |
| Maximum birth marginal | 0.097656 | 0.107422 | +0.009766 |
| Maximum death marginal | 0.041016 | 0.062500 | **+0.021484** |
| Assigned-count histogram TV | 0.087891 | 0.068359 | -0.019531 |

The candidate/control process-CPU ratios were 0.5931 and 0.7048 for seeds
510 and 511; sweep-weighted PG selection ratios were 0.9451 and 1.0774.
These satisfy the cost and selection gates. The adaptive tempering schedule
had 22 stages for each arm and seed. Timing depends on this observation and
the actual PG allocation; the experiment does not establish a general
speedup. The raw aggregate summaries and protocol are in
[`results/wp11i_late_pg_development.json`](../results/wp11i_late_pg_development.json).

The 8,192-particle exposed-case test was not run. The failed fresh-case
screen should not be repaired by retuning on its observed outcome. A new
proposal would need a frozen independent development protocol and the same
exact-oracle check before any new high-particle test. The untouched holdout
remains closed.
