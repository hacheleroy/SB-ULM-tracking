# WP4 — Explicit birth/death slots

Benchmark date: 2026-10-19.

## Scope and representation

WP3 could distinguish a detected bubble from a missed persistent bubble, but a
long sequence of misses still meant that the latent bubble existed throughout
the acquisition. WP4 adds a separate existence variable. For frame (t) and
slot (i):

\[
z_{t,i}\in\{0,1\},\qquad
a_{t,i}\in
\begin{cases}
\{-2\}, & z_{t,i}=0\\
\{-1,0,\ldots,M_t-1\}, & z_{t,i}=1
\end{cases}
\]

The sentinels have different meanings:

- `-2`: the slot does not exist in this frame;
- `-1`: the slot exists but was missed;
- a non-negative index: the slot generated that detection.

Non-negative indices remain injective within each frame. Any unused detection
is uniform Poisson clutter. Each slot has either no alive frame or one
contiguous alive interval, so a bubble can be born once and die once. The
calibrated benchmark uses two exchangeable labelled slots. Their labels have no
physical meaning: a global slot swap has exactly the same probability.

This is variable cardinality *within a fixed maximum number of slots*. It is
not yet an unbounded random-finite-set model.

## Birth--survival--death prior

Each slot follows the same three-state Markov process:

1. it is alive in frame zero with probability (p_0);
2. while unborn, it is born before the next frame with probability (p_B);
3. while alive, it survives to the next frame with probability (p_S);
4. death is absorbing.

The frozen calibration values are (p_0=0.50), (p_B=0.25), and
(p_S=0.80). The process assigns normalized probability to the never-alive
mask and to every possible non-empty interval. For an interval from birth
frame (b) through last-alive frame (d), its probability is evaluated
directly from these transitions. Slots are independent under the prior and
exchangeable because they share all existence and motion parameters.

Conditional on the mask, the count factor is:

\[
p(Y,a\mid z)\propto
e^{-T\lambda_c}
p_D^{n_A}(1-p_D)^{n_M}
\kappa^{n_C}
\prod_{i,d}p(y_{a_i,d})
\]

where (n_M) counts only alive-but-missed states, not absent states,
(kappa=\lambda_c/|R|), and each (p(y_{a_i,d})) analytically integrates the
calibrated Matérn-3/2 path using the detections assigned to slot (i). A
conditional continuous path is returned only on the alive interval; absent
coordinates are represented by `NaN` in the sampled-path array.

## Exact base distribution and reversible moves

The annealed SMC base includes both the existence prior and the complete
detection/miss/clutter count density:

\[
q_0(z,a)\propto p(z)
p_D^{n_A}(1-p_D)^{n_M}\kappa^{n_C}
\]

For two slots and four frames, all interval-mask pairs can be enumerated
cheaply. The implementation sums each frame's partial-injection partition
function for every mask pair, then samples exactly from normalized (q_0).
This enumeration is internal to the tractable base distribution; it does not
enumerate the Matérn target and does not use the exact posterior oracle.

The Matérn marginal density is introduced through adaptive tempering. Three
move types rejuvenate every temperature:

1. **Whole-frame association:** replace the complete compatible frame state.
   This includes assigned-to-missed, missed-to-assigned, reassignment, and
   target/clutter exchange moves.
2. **Birth/death interval:** select one slot and replace its complete existence
   interval. Every frame whose active set changes receives a newly drawn
   compatible partial injection. If (N_t(z)) is the number of frame states,
   the proposal correction is

   \[
   \log\frac{q(\text{old}\mid\text{new})}{q(\text{new}\mid\text{old})}
   =\sum_{t\in\Delta}
   \left[\log N_t(z_{\mathrm{new}})-\log N_t(z_{\mathrm{old}})\right]
   \]

3. **Global slot permutation:** swap two complete association and existence
   columns. This symmetric move crosses the otherwise arbitrary labelled
   representations of the same physical configuration and is accepted with
   probability one up to numerical roundoff.

After the final temperature, continuous paths are sampled exactly from the
Gaussian conditional associated with each selected discrete state.

## Enumerable oracle and calibration protocol

For each pair of slot intervals, the oracle enumerates the Cartesian product
of every compatible frame-level partial injection. The full exact state
therefore includes both the existence mask and the association path. The
inference routine never calls this oracle; it is used only to assess:

- total-variation and Jensen--Shannon error over the full labelled support;
- exact 95% credible-set recall and exact mass covered;
- per-frame existence, missed-detection, and expected-clutter marginals;
- birth-frame and last-alive-frame distributions, including never-alive;
- a sliced-Wasserstein distance on masked continuous paths;
- the SMC estimate of the exact normalizing constant.

The primary gate uses four frames, two slots, 5 µm localization noise, the
6.84 mm/s Matérn velocity scale, 8,192 SMC particles, ten mutation sweeps per
temperature, ten additional final sweeps, and three SMC seeds per regime.
Finite-sample TV, JS, credible-set, mass, and path thresholds are derived from
independent exact posterior draws of the same size. Every event-marginal error
must be below 0.05 and the absolute log-evidence error below 0.20 nat.

| Regime | Truth | (p_D) | (lambda_c) | Detection counts | Exact support |
|---|---|---:|---:|---|---:|
| Persistent near-clean | both alive in frames 0--3 | 0.95 | 0.05 | 2, 2, 2, 2 | 16,877 |
| Delayed birth | slot 1 born in frame 2 | 0.80 | 0.15 | 2, 0, 2, 2 | 3,529 |
| Early death | slot 0 last alive in frame 1 | 0.80 | 0.15 | 2, 2, 1, 1 | 5,347 |
| Turnover + corruption | slot 0 dies after frame 2; slot 1 born in frame 1 | 0.70 | 0.55 | 1, 1, 2, 1 | 3,267 |

The clutter simulator is capped at one false positive per frame solely to keep
the realized oracle small. Inference still scores the stated untruncated
Poisson point-process model.

## Results

The posterior-sampling checks pass in all 12 primary runs:

- TV and JS over the complete mask-and-association support pass in 12/12;
- exact 95% credible-set recall passes in 12/12;
- existence, missed-detection, clutter, birth-time, and death-time marginals
  pass their 0.05 error ceiling in 12/12;
- the masked continuous-path sliced-Wasserstein check passes in 12/12;
- the maximum errors for existence, birth time, and last-alive time are
  respectively 0.0130, 0.0103, and 0.0135;
- the largest TV is 0.0839 and the largest path distance is 0.0958 µm.

| Regime | Mean TV | Existence error | Birth-time error | Death-time error | Path SW (µm) | Mean absolute evidence error (nat) |
|---|---:|---:|---:|---:|---:|---:|
| Persistent near-clean | 0.00889 | 0.00199 | 0.00040 | 0.00221 | 0.0752 | 0.2387 |
| Delayed birth | 0.03299 | 0.00359 | 0.00495 | 0.00091 | 0.0656 | 0.0539 |
| Early death | 0.03422 | 0.00357 | 0.00089 | 0.00312 | 0.0634 | 0.0205 |
| Turnover + corruption | 0.08226 | 0.01024 | 0.00812 | 0.00881 | 0.0694 | 0.0141 |

The deliberately strict aggregate flag is nevertheless **false (11/12 full
checks)**. Persistent-near-clean seed 2 has two failures:

1. it covers 0.997395 of exact mass versus a Monte-Carlo fifth-percentile
   threshold of 0.997403, a difference of (7.6\times10^{-6}), despite
   observing every state in the exact 95% credible set;
2. its single-run log-evidence error is +0.473 nat versus the predeclared
   0.20-nat ceiling.

The first is a retained finite-sample boundary event. The second is a genuine
numerical limitation of the present normalizing-constant estimator. Pooling
the three independent normalizing-constant estimates in linear space gives
per-regime log-evidence errors of +0.119, +0.0540, -0.0205, and -0.0129 nat,
all below 0.20 nat, but this pooled statistic was not substituted for the
predeclared individual-run gate.

A targeted escalation of the failed seed from 8,192 to 16,384 particles
improves TV from 0.00789 to 0.00609 and exact mass coverage from 0.99740 to
0.99921. Event-marginal errors remain below 0.0031. Its evidence error changes
from +0.473 to -0.265 nat and still fails. Thus the evidence issue is kept
visible rather than being labelled a posterior failure or hidden by selecting
a favorable rerun. The complete sensitivity record is
`results/wp4_evidence_escalation.json`.

## Interpretation and remaining boundary

The exact target can distinguish a bubble that is absent from one that exists
but was missed, and the sampler recovers birth and death uncertainty without a
post-processing heuristic. The global slot move also removes the complete
first-frame label anchor used by WP3. The positive claim is posterior
calibration for this small representation, not calibrated single-run evidence.
The current evidence estimate must not yet be used for Bayes factors or model
selection.

This milestone does not yet validate:

- more than two candidate slots or sequences longer than four frames;
- an unknown or unbounded maximum number of slots;
- split, merge, or re-entry events beyond one contiguous interval per slot;
- inference of (p_0), (p_B), (p_S), (p_D), or (lambda_c);
- non-uniform, structured, or temporally correlated clutter;
- real ULM model fit, IQ/RF likelihoods, or acquisition-level inference.
- stable single-run marginal-likelihood estimation in the concentrated
  persistent regime.

The next algorithmic step is therefore scalability, not another change of
small-case semantics: blocked/sequential birth-death proposals and pruning
must be tested while retaining exact-oracle regression cases. Evidence-specific
work should separately test replicate averaging, denser tempering, and improved
intermediate mutation before any marginal-likelihood claim. Only after those
gates should explicit slots be compared with relaxed assignment matrices or
variable-mass formulations.

## Reproduction

To avoid BLAS thread oversubscription with several workers:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m sbulm.wp4_existence_calibration \
  --output-dir results \
  --n-particles 8192 \
  --n-smc-repeats 3 \
  --mh-sweeps 10 \
  --final-mh-sweeps 10 \
  --workers 6
```

The compact outputs are `wp4_existence_calibration.json/.png` and
`wp4_evidence_escalation.json`. The primary JSON stores every exact and sampled
marginal, temperature, acceptance rate, finite-sample reference, threshold,
and gate decision.
