# WP11u — Late suffix particle Gibbs on the exposed absolute case

WP11t established an independently checked exact 198-cell posterior for the
already exposed WP11k primary, with recurrent alive-count and late-clutter
errors from the unchanged four-path PG mixture. This step tests **one**
predeclared kernel and one paired screen. The observation and its exact mode
weights have already been seen. Passing this screen cannot establish
generalization or license the untouched holdout.

## Kernel and exact-target check

The opt-in conditional particle-Gibbs kernel selects a uniformly random pair
of slots. It fixes all other slots and the selected pair's first five frames,
then samples frames six through eight with four fully adapted conditional SMC
particles and a retained reference path. At the first suffix frame the
previous alive/unborn/dead state and collapsed GP history come from the fixed
prefix; later frames use the same exact incremental birth/death, count and
Matérn factors as full-path PG. The kernel is a Gibbs update conditional on
the fixed prefix and so preserves each tempered target. We use it only at
beta exactly one, without ancestor sampling. The existing full-path PG and
local pair kernels remain in the mixture.

Before inference, enumerate the full transition of a two-particle, two-frame
suffix update on a 165-state, two-slot, three-frame exact model at beta 0,
0.37 and 1. Verify stochastic rows and stationarity to absolute 1e-12 and
relative 1e-11. Verify that the prefix is preserved by the opt-in sampler and
that the default selection path remains unchanged. A failure stops inference.

## Frozen paired screen, committed before inference

The exact-reference input is `results/wp11t_subset_exact_primary.json`, byte
SHA-256 `52f13725b9f7dd8851b44ee97025afebf98f042791b111cf508dcbd5c0c0c14c`.
The observation detection hash is
`e7fc37d377e5b956377af6a64e41837f15e42d43e61a932af48400da9bffd922`.
Use the unchanged four-slot, eight-frame, 19-detection WP11k primary and the
WP11g 10% four-path PG mixture at 2,048 particles. Each arm uses the same
seed in `1300..1305`; order alternates control/candidate, candidate/control.
Each arm is a separate one-thread process with a 750 CPU-second cap.

The candidate retains the control's 10% full-path PG at every stage. At beta
one, it additionally replaces 40% of paired proposals by the suffix kernel
(the remaining 50% use the local pair move) and increases the *additional*
final sweeps from 8 to 24, so the last stage receives 32 rather than 16
total sweeps. At earlier temperatures it is identical to control. The
selection probabilities depend only on temperature, not on the current
association. No other control changes.

Compare each final alive-total × frame-six clutter × last-frame clutter joint
to the exact 33 × 3 × 2 reference. **Every** condition below must pass:

| Criterion | Frozen threshold |
| --- | ---: |
| Candidate/control CPU ratio, each seed | ≤1.50 |
| Paired full-joint TV wins | ≥4 of 6 |
| Candidate median full-joint TV | ≤0.12 |
| Candidate median band × frame-six-clutter TV | ≤0.10 |
| Candidate median maximum band-mass error | ≤0.08 |
| Candidate/control median ratio, each of those three targets | ≤0.80 |
| Candidate median regression in alive TV, max joint-cell error, absolute sixth and last clutter-mean errors | ≤0.020 each |

Also report all per-seed joint cells, signed clutter errors, log evidence,
stage movement and the WP11t exact-IID q95 for scale. IID tails are
descriptive because SMC particles are correlated and are not a gate. Run
all six fixed pairs, including if an early pair looks unfavorable. No
parameter or threshold changes after seeing an arm. If a cap or oracle
check fails, record the failure and stop. A passing exposed-case gate still
requires **independent observation replication** with its protocol frozen
before inference. There is no 8,192-particle run or untouched WP8 holdout
in this step.

Runner: `src/sbulm/wp11u_late_suffix_pg.py`; final checkpoint:
`results/wp11u_late_suffix_pg.json`.

## Outcome

All twelve arms completed within the 750 CPU-second caps. The exhaustive
small-case transition checks passed at beta 0, 0.37 and 1. The candidate
selected the suffix PG on 39.9–41.0% of final-stage paired proposals; its
suffix updates changed a per-frame alive/assigned count signature on
40.3–43.1% of selections.

| Seed | Control full-joint TV | Candidate full-joint TV | Candidate/control CPU |
| ---: | ---: | ---: | ---: |
| 1300 | 0.21201 | 0.17772 | 1.010 |
| 1301 | 0.24269 | 0.22393 | 1.376 |
| 1302 | 0.18152 | 0.10493 | 1.015 |
| 1303 | 0.16246 | 0.16741 | 1.184 |
| 1304 | 0.23008 | 0.17181 | 1.197 |
| 1305 | 0.11678 | 0.09392 | 1.164 |

| Frozen criterion | Result | Pass? |
| --- | ---: | :---: |
| Per-seed CPU ratio ≤1.50 | Maximum 1.376 | Yes |
| Paired full-joint TV wins ≥4/6 | 5/6 | Yes |
| Candidate median full-joint TV ≤0.12 | **0.16961** | **No** |
| Candidate median band × sixth-clutter TV ≤0.10 | **0.13502** | **No** |
| Candidate median maximum band-mass error ≤0.08 | **0.09992** | **No** |
| Candidate/control median ratio ≤0.80 on all three | 0.862 / 0.841 / 0.976 | **No** |
| No other median regression >0.020 | All four improved | Yes |

Control medians for those three targets were 0.19676, 0.16060 and
0.10234. Thus the suffix proposal reduced median full-joint TV by about
14% and median band-by-clutter TV by about 16%, but missed the frozen
absolute and relative error requirements. It is **not selected**. The
status is `exposed_development_gate_failed_stop`; the criteria were not
revised from this outcome.

For descriptive scale, the exact-IID q95 for full-joint TV at 2,048 is
0.07099: all six candidate arms still exceed it. Their largest pooled
mean signed cell error is 0.01515 at alive 17, frame-six clutter 2 and
last clutter 1, versus the preexisting exact-IID familywise q95 0.00621.
These IID comparisons are not calibrated significance tests for correlated
SMC. Because the candidate changes only beta-one rejuvenation, each
matched pair has **identical** evidence estimate; its errors remain
0.831–2.643 nats below the exact log evidence. A final-stage kernel can
improve the displayed posterior sample while leaving earlier evidence
loss untouched. This exposed observation cannot identify the unique
cause of the residual mode error.

The full 198-cell tables, all per-seed signed errors, stage diagnostics,
costs and exact gate calculation are retained in the checkpoint. No
8,192-particle run or untouched holdout was opened. A next proposal
should examine *when* the count/clutter modes are lost during tempering
and freeze its protocol on an independent observation before testing
another earlier-stage move.
