# WP5 — Scaling explicit slots beyond two bubbles and four frames

## Question

Can the calibrated birth/death sampler move beyond the two-slot, four-frame
WP4 oracle without silently losing posterior mode weights?

This milestone separates two claims:

1. **exact calibration:** agreement with complete enumeration where an oracle
   still fits in memory;
2. **larger-case feasibility:** the code runs without enumerating all slot
   intervals, with stability checked across independent SMC repetitions.

Only the first is a calibration claim. Cross-run agreement without an oracle
is a necessary diagnostic, not proof of correctness.

## Sequential exact base sampler

WP4 formed the Cartesian product of all contiguous existence intervals. With

\[
L(T)=1+\frac{T(T+1)}{2}
\]

intervals per slot, this requires (L(T)^K) mask combinations for (K) slots.
The new initializer instead represents every slot at each frame as unborn,
alive, or permanently dead. The independent birth/survival transition law and
the frame count partition form a finite hidden Markov model on (3^K) joint
states.

For active mask (z_t), let

\[
Z_t(z_t)=\sum_{a_t\in\mathcal A(z_t)}
p_D^{n_A}(1-p_D)^{n_M}\kappa^{n_C}
\]

where \(\kappa=\lambda_c/|R|\). Forward filtering computes the exact base
normalizer, and backward sampling draws complete existence histories. Frame
associations are then drawn from their normalized count weights. This is an
exact sampler for the same base density as WP4; it does not use the posterior
oracle.

The current dense joint-state implementation is exponential in the number of
slots, but linear in the number of frames. For the 4-slot/8-frame stress test,
it uses 81 joint HMM states instead of materializing 1,874,161 combinations of
slot intervals.

## Conditional-base blocked slot move

One move selects a slot and conditions on every other slot. At frame (t),
let (U_t) be the number of detections not claimed by the other slots. Relative
to leaving the selected slot absent, an alive-but-missed state has weight
(1-p_D), and assignment to each available detection has weight
(p_D/\kappa). Consequently, an interval (I) has conditional base weight

\[
p(I)\prod_{t\in I}
\left[(1-p_D)+U_t\frac{p_D}{\kappa}\right]
\]

The move draws the complete birth/death interval and all temporal associations
of that slot at once. Its Hastings term is evaluated exactly. The existence
prior and detection/clutter factors cancel from the Metropolis ratio, leaving
only

\[
\beta\left[log p_{\mathrm{Mat\acute ern}}(y\mid a')-
\log p_{\mathrm{Mat\acute ern}}(y\mid a)\right]
\]

At \(\beta=0\), every blocked proposal is therefore accepted. This identity is
tested numerically to (2\times10^{-14}). The calibrated hybrid uses 15%
blocked slot moves, 45% interval moves, 30% whole-frame association moves, and
10% global slot swaps. A pure blocked kernel was correct but inefficient at
high temperature because proposals informed only by the base density often
destroyed a good Matérn trajectory.

## Exact calibration cases

Both frozen cases contain three exchangeable slots, sparse detections, explicit
misses and clutter, and staggered births/deaths. The sampler uses 2,048
particles and three independent seeds per method.

| Case | Exact hypotheses | Detection counts | Method | Mean TV | Largest event error | Mean runtime |
|---|---:|---|---|---:|---:|---:|
| 3 slots, 5 frames | 111,742 | 1, 0, 1, 1, 1 | local | 0.7443 | 0.0265 | 25.2 s |
| 3 slots, 5 frames | 111,742 | 1, 0, 1, 1, 1 | blocked hybrid | 0.7459 | 0.0237 | 44.1 s |
| 3 slots, 6 frames | 349,312 | 0, 2, 0, 1, 1, 0 | local | 0.6575 | 0.0259 | 28.9 s |
| 3 slots, 6 frames | 349,312 | 0, 2, 0, 1, 1, 0 | blocked hybrid | 0.6621 | 0.0333 | 43.8 s |

The apparently large TV distances are expected because the exact posterior is
spread over many low-mass hypotheses. Equal-size direct draws from the exact
posterior have median TVs of 0.7459 and 0.6573 in the two cases. The gate
therefore compares global distances, credible-set recall, and exact mass
coverage with finite exact-draw baselines, and separately requires every
existence, miss, clutter, birth-time, and death-time marginal error to remain
below 0.06.

All six blocked-hybrid runs pass every exact-target check. All six local-control
runs pass as well. The blocked move is therefore calibrated but not a speed
improvement in these small cases.

## Four-slot/eight-frame boundary

The larger case has four slots, eight frames, 1,874,161 possible combinations
of existence intervals, and up to four detections in a frame. The sequential
base sampler initializes it without interval-product enumeration, and final
populations retain 1,989--2,034 unique states out of 2,048 particles.

However, three independent runs do not meet the predeclared 0.10 stability
threshold:

| Cross-run diagnostic | Maximum difference |
|---|---:|
| Existence marginal | 0.1069 |
| Missed-detection marginal | 0.1069 |
| Expected clutter | 0.2495 |
| Birth-time marginal | 0.1069 |
| Death-time marginal | 0.0947 |
| Total-alive-count distribution, TV | 0.1543 |
| Assigned-count distribution, TV | 0.1206 |

Extra final mutations, a local one-frame birth/death-boundary proposal, and a
finer 0.95-ESS temperature schedule were tested diagnostically. They changed
the failure magnitude but did not make the four-slot posterior reproducible,
so they were not adopted as evidence of calibration.

## Decision

The validated ceiling is now **three slots over six frames**, rather than two
slots over four frames. The sequential HMM removes interval enumeration as the
immediate implementation bottleneck and the conditional-base blocked move is
reversible and exactly calibrated on the expanded oracle.

The four-slot/eight-frame result is a no-go for a stronger claim. The next
methodological decision is whether to develop a collective, likelihood-informed
multi-slot move that preserves competing mode weights, or to leave labelled
slots for a relaxed assignment or variable-mass representation. In-vivo ULM
should not be used to decide between these samplers because it lacks an exact
posterior oracle.

## Reproduction

```bash
python -m sbulm.wp5_scaling_calibration \
  --output-dir results \
  --n-particles 2048 \
  --n-repeats 3 \
  --baseline-replicates 100
```

Outputs:

- `results/wp5_scaling_calibration.json`
- `results/wp5_scaling_calibration.png`
