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

```bash
python -m sbulm.wp11b_conditional_validation --stage oracle
python -m sbulm.wp11b_conditional_validation --stage reserved
```
