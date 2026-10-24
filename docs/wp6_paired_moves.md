# WP6 — Likelihood-informed paired-slot moves

## Frozen question

WP5 calibrated the explicit birth/death posterior through three slots and six
frames, but independent four-slot/eight-frame runs disagreed on expected
clutter by as much as 0.2495 (predeclared stability limit: 0.10). This study
tests a joint move without changing the posterior, observation model, prior,
particle count, exact-oracle cases, or large-case dataset.

## Proposal and correction

Choose two slots and a starting frame uniformly. Usually update one frame;
attempt a two-frame block half the time when its **raw** option count is at
most 128. That count depends only on the detections available after fixing the
other slots, not on either selected slot's current state. The resulting
finite conditional support contains every valid pair assignment and existence
state inside the window, with all outside frames fixed. The old and proposed
states therefore have the same candidate set and the same uniform proposal
law. Both slots may change together; the move can affect a birth/death
boundary, missed detections, false-positive clutter, and identity association.

For the annealed target, write

\[
\pi_\beta(z)\propto B(z)\exp\{\beta L(z)\}
\]

where \(B\) is the exact existence-and-detection-count base and \(L\) is the
collapsed Matérn localization log likelihood. Draw \(M=2\) candidates
uniformly from the finite conditional support; choose one with weight
\(w(z)=B(z)\exp\{\beta L(z)\}\). Draw \(M-1\) fresh reverse candidates and
include the current state in the reverse set. Accept the selected candidate
with probability

\[
\min\left\{1,\frac{\sum_{j=1}^{M}w(y_j)}
{w(x)+\sum_{j=1}^{M-1}w(x_j^*)}\right\}
\]

The common uniform proposal factor cancels. This is an exact independent
multiple-try Metropolis update on each selected conditional support. A mixture
of these reversible updates with the existing WP5 moves still targets the
same posterior. Selecting between one- and two-frame blocks based on the
current selected-slot state would break this argument; the implementation
uses only the fixed other slots for that decision.

The frozen comparison assigns 25% of moves to this paired update, 15% to the
previous conditional-base single-slot update, 35% to interval replacement,
20% to whole-frame association, and 5% to label swaps. The original WP5
blocked mixture is rerun with identical seeds as a control. The pair move is
scored by **actual state changes**, not acceptance alone, because a candidate
may equal the current state.

## Decision gates

1. Match the exact three-slot/five- and six-frame posteriors against the same
   finite-draw reference and 0.06 event-marginal limit used in WP5.
2. On the unchanged four-slot/eight-frame case, require *every* cross-run
   existence, miss, clutter, birth, death, alive-count, and assigned-count
   disagreement to be at most 0.10 over independent runs.
3. Report the control, acceptance/change rates, and runtime even if the new
   sampler fails. Cross-run stability is necessary but never a substitute for
   a four-slot exact posterior oracle.

Run the frozen comparison with:

```bash
python -m sbulm.wp6_paired_scaling \
  --output-dir results \
  --n-particles 2048 \
  --n-repeats 3 \
  --baseline-replicates 100
```

For a separate, untuned three-seed check on the same four-slot data:

```bash
python -m sbulm.wp6_paired_scaling \
  --output-dir results \
  --n-particles 2048 \
  --n-repeats 3 \
  --heldout-only \
  --heldout-seed-start 200
```

## Observed result

The complete 2,048-particle comparison used three seeds per method and
oracle case. All 12 three-slot runs (six blocked controls and six paired)
pass the finite-sample-calibrated support, TV, JS, credible-set, and event
checks. The largest paired event error is 0.0341 for five frames and 0.0240
for six frames, below the 0.06 limit. This validates the target-preserving
paired kernel on the expanded enumerable benchmark; it does not validate a
four-slot posterior.

| Four-slot/eight-frame cross-run metric | WP5 blocked | Paired | Limit |
|---|---:|---:|---:|
| Existence | 0.10693 | 0.07666 | 0.10 |
| Missed detection | 0.10693 | 0.07666 | 0.10 |
| Expected clutter | 0.24951 | 0.06445 | 0.10 |
| Birth time | 0.10693 | 0.07666 | 0.10 |
| Death time | 0.09473 | 0.06055 | 0.10 |
| Alive-count TV | 0.15430 | **0.10010** | 0.10 |
| Assigned-count TV | 0.12061 | 0.05322 | 0.10 |

Thus the original frozen three-seed gate **does not pass**: alive-count TV
exceeds the limit by about 0.00010. The paired move reduces the worst clutter
disagreement by roughly fourfold and makes actual state changes in about
31% of selected paired proposals, but cannot be called stably calibrated at
four slots on this evidence. Average four-slot runtime rises from about
67.6 s to 129.1 s per run (1.9 times as long).

Without modifying the sampler or threshold, a second triplet of seeds
(200--202) passes all seven within-triplet checks. Its largest discrepancy is
alive-count TV = 0.07715; clutter disagreement is 0.05371. This confirms
that the first gate was close, not that its predeclared failure may be
discarded. No exact four-slot oracle was constructed. The calibrated ceiling
remains **three slots/six frames**; the four-slot result is promising but
borderline and computationally costly.

The full per-seed comparisons and generated figure are in
`results/wp6_paired_scaling.json/.png`, and the untuned second triplet is in
`results/wp6_paired_heldout.json`. The pilot at 256 particles is exploratory
and is not used to make a calibration claim.
