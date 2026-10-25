# WP11c — Full-history pair move, exact small-case early stop

## Question and frozen pilot

WP11b improved local mutation but failed the two-seed eight-frame stability
gate. Its largest disagreements involved alive counts and clutter/association
choices. This pilot tests whether replacing the local paired-slot move with a
whole-sequence two-slot update can cross those modes. It does **not** reopen
the inspected WP11b eight-frame observation or claim a new large-case result.

The pre-existing `PairConditionalBase` draws both selected slots jointly
from their exact conditional *base* distribution, including birth/death,
misses, clutter and associations across every frame. The two-try
`multiple_try_paired_move` reweights candidates with the full collapsed
localization likelihood and uses its exact reverse multiple-try correction.
WP11c exposes this as `paired_slot_scope="full_sequence"`; the existing
`"local"` scope remains the default. An additional diagnostic counts accepted
changes to the label-invariant sequence of alive and assigned counts, not
merely changes in slot labels.

Finite-state tests check detailed balance at tempering levels 0, 0.37 and
1, and verify that full-history moves really change counts at the base
distribution. The performance pilot uses a 269-state two-slot/three-frame
exact posterior. At beta 1, it starts 512 chains from that exact posterior,
applies two paired-move sweeps at seeds 74, 75 and 76, and compares the
full-sequence and local scopes on the same initial states. The protocol
requires a mean accepted count-signature change rate of at least 0.01 per
selected full-history move before any fresh eight-frame holdout is opened.

## Result

| Seed | Full-history acceptance and count changes | Local acceptance | Local count changes |
| --- | ---: | ---: | ---: |
| 74 | 0.0029 | 0.4277 | 0.1270 |
| 75 | 0.0029 | 0.4463 | 0.1465 |
| 76 | 0.0000 | 0.4189 | 0.1572 |

The mean full-history count-change rate is 0.0020, below the frozen 0.01
gate. This is an **early stop**: no new eight-frame observations or seeds
were opened. The exact finite-state validity of the move does not imply a
useful proposal at beta 1. On this small model, the total-variation distance
between its conditional base proposal and the exact posterior is 0.9889;
the base proposal places only 0.0013 probability on the exact posterior's
95% credible set (39 of 269 states). Two independent base candidates per
move rarely reach a plausible full trajectory. The observed pilot wall
times are approximately 5.0--5.3 seconds per full-history arm versus
0.6--0.7 seconds per local arm per seed; these are diagnostic timings, not
a controlled hardware benchmark.

The next proposal must use a likelihood-informed sequential or conditional
path construction that *maintains the current high-probability trajectory*
and can still change births, deaths and clutter associations jointly. A
conditional SMC/particle-Gibbs update is one candidate, but its proposal
and intermediate weights must account for the collapsed, temporally
correlated localization model; merely substituting a prior particle filter
would repeat the base/posterior mismatch. Such a kernel should first pass
small exact detailed-balance and mode-mass tests, then a development test,
before freezing any new held-out eight-frame validation.

```bash
python -m sbulm.wp11c_full_history_pilot
```
