# WP11b — GP-predictive non-self tracklet proposal

## Method

WP11's reversible three-frame proposal passed the small exact oracles but
produced accepted state changes in only about 1% of selected tracklet moves;
all three eight-frame pairs failed the inherited gate. WP11b retains the
same posterior target and mixture weights while changing that proposal.

For two selected slots and a three-frame window, assigned detections receive
their Gaussian predictive log density under the target's Matérn covariance,
conditioned **only on fixed detections outside the window**. The auxiliary
finite-state HMM combines that score with the detection/clutter ratio and a
half-weight displacement edge. It guides proposals; the full collapsed
Matérn likelihood still determines Metropolis acceptance.

An augmented forward recursion tracks whether the proposed path has already
diverged from the current one. Backward sampling conditions on divergence,
thus excluding self-draws whenever an alternative exists. The exact reverse
correction is:

\[
\log\frac{q(x\mid y)}{q(y\mid x)}
=s(x)-s(y)+\log Z_{\ne x}-\log Z_{\ne y}
\]

where \(s\) is the auxiliary HMM path score and \(Z_{\ne x}\) is its
partition function excluding \(x\). The selected pair and window are
weighted only by existence masks and observation cardinalities, which the
move cannot alter, so their selection probabilities cancel on reversal.
Birth/death changes remain the responsibility of the other moves. Exact
finite-state tests check pairwise detailed balance at temperatures 0, 0.37
and 1, including the non-self normalizers.

## Frozen validation, before reserved inference

1. On the existing 282,469-state four-slot/four-frame bridge (SHA-256
   `62d3aeeafff58eb47404f87291829004f5bd498d07ab1494afa86ba77115b8ef`),
   run 2,048 particles at seeds 400, 401 and 402. All must pass the
   previously recorded WP7 finite-draw TV, JS, support-coverage and 0.06
   event-marginal checks. Record edge/clutter errors and actual accepted
   state-change rates as extra diagnostics.
2. Only on an oracle pass, open a previously unused cluttered staggered
   eight-frame realization: trajectory seed 53, corruption seed 29,
   detection probability 0.65, expected clutter 0.45, observation SHA-256
   `bd8c6aed400ffd38c24621610e34b7baf9c74a8a9eca620b65a33b8e4006815c`.
   At 8,192 particles, compare the new mixture with the old WP6 paired
   mixture, each at independent inference seeds 310 and 410. Both use the
   identical model, observations, ESS target 0.80, eight sweeps per stage
   and eight extra final sweeps. The new probabilities are 0.15 frame
   association, 0.30 interval, 0.05 slot swap, 0.15 blocked slot, 0.20
   paired slot, and 0.15 conditional-GP tracklet.
3. Apply the inherited seven-metric inclusive 0.10 pairwise gate to each
   arm separately. Report per-detection clutter, adjacent association
   edges, count-signature coverage and runtime as ungated diagnostics.

A passing pair is only a diagnostic lead, not exact eight-frame posterior
calibration or the six-run stability result. Neither move probabilities nor
the reserved observations/seeds are changed after seeing the reserved runs.
The checkpoint verifies hashes and blocks reserved inference until all
three exact-oracle runs pass.

## Exact-oracle gate

All three predeclared four-slot runs pass the inherited exact-oracle checks.
Maximum event-marginal errors are 0.0200, 0.0244 and 0.0253 against the
0.06 limit. Maximum adjacent association-edge errors are 0.0007, 0.0003
and 0.0007; per-detection clutter errors are 0.0044, 0.0041 and 0.0026.
The mean accepted tracklet state-change rates are 0.394, 0.397 and 0.395
per selected move, at 137, 170 and 192 seconds per run. The sparse oracle
therefore supports correctness and meaningful local mutation, but does not
establish eight-frame mode-weight calibration or transfer.

## Reserved eight-frame comparison

Both arms completed the predeclared seeds 310 and 410 on the same held-out
observation. Neither passes the inherited inclusive 0.10 seven-metric
two-seed gate. The GP-predictive non-self mixture has a worst discrepancy of
0.1827 (alive-count total variation); the paired-slot control has a worst
discrepancy of 0.2484 (maximum clutter marginal). The new mixture improves
six of seven gated discrepancies relative to the control, but its
assigned-count total variation is slightly worse.

| Cross-seed diagnostic | GP-predictive non-self | Paired control |
| --- | ---: | ---: |
| Maximum existence-marginal difference | 0.0851 | 0.1206 |
| Maximum missed-marginal difference | 0.0570 | 0.1351 |
| Maximum clutter-marginal difference | 0.1653 | 0.2484 |
| Maximum birth-marginal difference | 0.0685 | 0.0691 |
| Maximum death-marginal difference | 0.0631 | 0.0916 |
| Maximum alive-count TV | 0.1827 | 0.2136 |
| Maximum assigned-count TV | 0.1136 | 0.0975 |
| Maximum per-detection clutter difference (ungated) | 0.1174 | 0.2406 |
| Maximum association-edge difference (ungated) | 0.1744 | 0.2377 |
| Count-signature histogram TV (ungated) | 0.6831 | 0.7670 |

The new move changes the selected state in about 0.400 and 0.411 of its
applications across the two runs, versus approximately 0.01 for WP11's
earlier guided tracklet move on different observations. Final unique-state
fractions are 0.983 and 0.980 for the new mixture, versus 0.958 and 0.965
for the paired control. These are useful within-run diagnostics, but the
remaining between-run count, clutter, and edge disagreements show that
frequent local mutations do not guarantee adequate global mode mixing.

Both arms take 21 and 20 tempering stages for seeds 310 and 410,
respectively. Recorded elapsed times are 3,119 and 1,493 seconds for the
new mixture and 3,112 and 2,281 seconds for the paired control. The shared
host's load varied markedly during these runs, so these wall-clock times
should not be interpreted as a controlled speed comparison. The large-case
pair is diagnostic only: it has no exact posterior oracle, uses just two
seeds on one realization, and establishes neither calibrated mode weights
nor generalization to in-vivo ULM. A subsequent proposal should target
global count/existence and association-clutter alternatives, then pass the
small exact oracles before another untouched large-case validation.

```bash
python -m sbulm.wp11b_conditional_validation --stage oracle
python -m sbulm.wp11b_conditional_validation --stage reserved
```
