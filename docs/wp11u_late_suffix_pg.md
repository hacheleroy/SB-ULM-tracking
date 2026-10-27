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

Pending the frozen six-pair screen.
