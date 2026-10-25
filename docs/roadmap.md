# Scientific roadmap and decision gates

## WP0 — Reproducible controlled benchmark

**Question:** Can the method preserve multiple coherent association hypotheses
at a bubble crossing?

- Fixed number of bubbles
- Unordered noisy localizations
- Exact enumeration of all associations under a specified dynamical model
- Hungarian position and constant-velocity baselines
- Metrics: identity switches, RMSE, true posterior mass, credible-set coverage
- Frozen regime map over central separation and localization-noise level

**Go/no-go criterion:** an approximate sampler must recover both the leading
hypotheses and their weights on enumerable cases before it is trusted on real
ULM data.

## WP1 — Smooth-SB reproduction

- [x] Audit and pin the public Hong, Shi & Niles-Weed implementation
- [x] Map the controlled ULM crossing to its expected `(T, N, D)` representation
- [x] Implement exact-versus-sampled distributional comparison metrics
- [x] Run an external reduced-discretization smoke test on the ULM crossing
- [x] Enumerate the Matérn-GP target implied by the notebook settings
- [x] Repeat ULM sampling at the published `Mi=32` over five random seeds
- [ ] Reproduce the authors' original one-by-one-matching example
- [x] Map the Smooth-SB target to a physically calibrated ULM motion prior
- Record runtime and failure modes as particle count and frame count increase

**Decision gate:** completed for the hard-marginal model. A velocity-calibrated
prior behaves plausibly at low localization noise but becomes confidently
wrong at realistic noise levels. WP2 confirms that an explicit localization
likelihood removes this dominant overconfidence failure on the frozen map.

## WP2 — ULM localization likelihood

- [x] Freeze one physical Matérn prior across the complete benchmark map
- [x] Replace hard intermediate marginals with an exact Gaussian observation
  likelihood for noisy localizations
- [x] Implement an order-invariant differentiable set potential and analytical
  gradient
- [x] Verify the gradient by finite differences and preserve exact association
  diagnostics
- [x] Run 3,136 paired cases on the frozen 7-by-7 map
- [x] Implement the validated potential in the public TSBM framework
- [x] Construct an exact Brownian-reference target matched to public TSBM
- [x] Compare forward and backward TSBM samples with exact discrete and
  continuous targets
- [x] Select calibration-first SMC (Option A) after the public-TSBM no-go
- [x] Implement collapsed adaptive-tempering SMC for the original Matérn
  localization posterior
- [x] Draw continuous Matérn paths exactly conditional on sampled associations
- [x] Pass the three-regime discrete, continuous, support, and evidence gate
- [x] Stress-test 196 separation/noise/data-seed cases against exact targets
- [x] Pass the fixed-cardinality gate before introducing missed detections

**Validation required:** physical ranges of bubble velocity and localization
error. Acceleration calibration requires Matérn-5/2 or a different motion model;
Matérn-3/2 does not have a finite acceleration variance.

**Decision gate:** the exact Gaussian-likelihood posterior passes, but the
public TSBM sampler does not. On the matched Brownian target, it discovers the
credible modes yet fails their weights, endpoint coupling, and continuous path
distribution. After Rao--Blackwellizing the association variable, the best
ambiguous canonical TV is 0.201 versus a 0.021 finite-sample 95% reference; the
best easy-case TV is 0.878. Variable cardinality was therefore paused at that
checkpoint. See
`tsbm_wp2_integration.md` for the controlled no-go result and the two resulting
method choices.

**Option-A result:** collapsed Matérn SMC passes all nine primary calibration
runs. On the 196-case map, TV, JS, and calibrated credible-set recall pass in
every case; the minimum exact posterior mass covered is 98.80%. The gate for
fixed-cardinality localization inference is therefore passed. See
`smc_matern_calibration.md`. Moving to WP3 is now blocked by a modelling choice,
not by an uncalibrated fixed-cardinality sampler.

## WP3 — Null associations before variable latent cardinality

- [x] Retain two persistent latent bubbles but allow a per-frame null
  association for a missed detection
- [x] Treat every unassigned observation as uniform Poisson clutter
- [x] Implement reversible whole-frame partial-injection moves and coherent
  segment label swaps
- [x] Preserve exact enumeration and continuous conditional path draws on
  five-frame calibration problems
- [x] Pass all 12 primary discrete, continuous, marginal, and evidence gates
- [x] Stress-test 36 detection-probability/clutter-rate/data-seed cases; all
  substantive metrics pass in every case, while one finite sample omits one
  low-mass member of an exact 95% credible set
- [x] Freeze the null-association model as the persistent-particle reference

**Decision gate:** the explicit null state is now calibrated for missed
detections and false positives, but it is not a birth/death model: every latent
bubble still exists over the full sequence. This gate motivated the separate
slot-existence model in WP4. See `wp3_null_association.md`.

## WP4 — Explicit births and deaths

- [x] Replace the first-frame label anchor by exchangeable global slot labels
- [x] Distinguish absent (`-2`) from alive-but-missed (`-1`) states
- [x] Restrict each slot to one contiguous alive interval, including a
  never-alive state
- [x] Add a normalized unborn--alive--dead Markov prior
- [x] Sample the exact existence-and-count base distribution
- [x] Add reversible whole-frame association, birth/death-interval, and global
  slot-permutation moves with the required Hastings correction
- [x] Preserve complete enumeration for two slots and four frames
- [x] Calibrate association/mask weights, existence, birth/death times, misses,
  clutter, conditional paths, and evidence against the exact posterior
- [x] Develop blocked and sequential proposals beyond the enumerable two-slot
  benchmark
- [ ] Compare explicit discrete slots with relaxed assignments or
  variable-mass representations

**Decision gate:** the representation is accepted for the small benchmark only
if every persistent, delayed-birth, early-death, and turnover repeat passes the
exact-target gate. A positive result validates the sampler for a fixed maximum
of two exchangeable slots; it does not establish scalable unknown-cardinality
ULM tracking. See `wp4_birth_death_slots.md`.

**Observed result:** all distributional, event-marginal, and conditional-path
checks pass in 12/12 runs. The strict all-check count is 11/12 because one
near-clean finite sample is infinitesimally below its exact-mass threshold and
its single-run evidence estimate fails. The birth/death posterior gate is
therefore positive, while marginal-likelihood estimation remains a separate
no-go until stabilized.

## WP5 — Scalable explicit slots

- [x] Replace interval-product base initialization by exact joint-HMM forward
  filtering/backward sampling
- [x] Add a conditional-base move that jointly redraws one slot's complete
  interval and temporal associations
- [x] Verify analytical cancellation and unit acceptance at beta zero
- [x] Pass six exact-target runs at three slots and five or six frames
- [x] Run a four-slot/eight-frame stress test without enumerating 1,874,161
  interval combinations
- [ ] Stabilize posterior mode weights at four slots/eight frames
- [ ] Choose between collective likelihood-informed moves and a
  relaxed/variable-mass representation

**Decision gate:** the calibrated ceiling advances to three slots over six
frames. The four-slot/eight-frame sampler is computationally feasible but not
calibrated: independent runs disagree by up to 0.2495 on expected clutter,
against a predeclared 0.10 stability threshold. Do not proceed to in-vivo
claims until the representation/move choice is resolved. See
`wp5_scalable_slots.md`.

## WP6 — Likelihood-informed paired-slot moves

- [x] Add a two-slot, one-/two-frame conditional-support multiple-try move
  with exact forward/reverse weight-sum correction
- [x] Verify conditional support reversibility and posterior stationarity on
  enumerable small cases
- [x] Compare with the unchanged WP5 blocked mixture on the three-slot exact
  oracle using identical seeds
- [x] Rerun the frozen four-slot/eight-frame stability check at 2,048
  particles and three independent seeds per method
- [x] Repeat the paired four-slot check on three untouched seeds without
  modifying the sampler or the 0.10 criterion

**Observed gate:** all six paired exact-oracle runs pass at three slots.
Four-slot clutter disagreement decreases from 0.2495 to 0.0645, but the
frozen three-seed gate fails narrowly on alive-count TV (0.10010 versus
0.10); runtime is approximately 1.9 times the control. The second triplet
passes within-triplet stability, which is encouraging but does not overturn
the frozen failure or provide a four-slot exact oracle. The calibrated ceiling
remains three slots/six frames. See `wp6_paired_moves.md`.

## WP7 — Four-slot convergence audit

- [x] Freeze an exactly enumerable four-slot/four-frame sparse benchmark
  (282,469 labelled states) and compare three paired-sampler runs with it
- [x] Increase to 4,096 particles on the unchanged four-slot/eight-frame
  data with the original and independent held-out seed triplets
- [x] Compare within-triplet and combined six-run event and count marginals
- [x] Test the discordant 200/201 pair at 8,192 particles as a diagnostic;
  their alive-count TV falls from 0.1418 to 0.0267
- [x] Finish the two original independent triplets at 8,192 particles and
  assess all seven metrics over the combined six-run set

**Observed boundary:** all three sparse four-slot oracle runs pass the
inherited checks. The combined six eight-frame runs at 4,096 particles fail
the 0.10 stability gate (alive-count TV 0.1453, clutter 0.1084,
assigned-count TV 0.1182); at 8,192 particles, all seven combined six-run
metrics pass (worst 0.0557). This establishes independent-run stability at
that count on the frozen synthetic case, not exact calibration of its mode
weights or transfer to other ULM conditions. See `wp7_convergence_audit.md`.

## WP8 — Held-out synthetic generalization

- [x] Freeze two new four-slot/eight-frame observation regimes before inference:
  a miss-heavy simultaneous crossing and a cluttered staggered crossing
- [x] Test the unchanged 8,192-particle paired sampler against the inherited
  seven-metric, 0.10 stability gate, requiring six seeds for a pass
- [x] Preserve failures and early stops rather than tuning to pass a regime

**Observed boundary:** the miss-heavy simultaneous crossing passes over six
independent seeds (worst 0.0497), while the cluttered staggered crossing
fails after two seeds (clutter difference 0.2013, alive-count TV 0.1511),
triggering the predeclared early stop. The unchanged sampler's stability
does not transfer across both cases. These are computational stability
checks conditional on simulator-informed prior parameters, not exact
eight-frame posterior calibration. See `wp8_generalization.md`.

## WP9 — Diagnose the difficult synthetic regime

- [x] Predeclare four counterfactual interventions on the exact WP8 failed
  realization: observed clutter, missed true detections, crossing offset
  and existence schedule
- [x] Compare seeds 100 and 200 at the unchanged 8,192-particle budget on
  the inherited seven metrics, including label-invariant count histograms

**Observed boundary:** none of the four single interventions brings the
two-seed discrepancy below 0.10. The worst metric is 0.1801 without
observed clutter, 0.6143 with all true detections completed, 0.3907 with
a wider crossing and 0.3832 with the WP7 existence schedule. This does
not isolate a unique physical cause; the unchanged sampler remains
unstable on every tested counterfactual. See `wp9_ablation.md`.

## WP10 — Sampler convergence on the frozen hard case

- [x] Predeclare doubled particles (16,384 with eight sweeps) and doubled
  rejuvenation (8,192 with 16 sweeps) on the original WP8 observations
  and seeds, holding the likelihood and paired move mixture fixed
- [x] Compare all seven inherited metrics and stage, ancestry, state-change
  and runtime diagnostics against the unchanged 8,192-particle baseline

**Observed boundary:** neither two-seed arm passes 0.10. Doubling particles
gives worst alive-count TV 0.2758. Doubling sweeps brings six metrics below
0.10, including alive-count TV 0.0674, but leaves frame-3 expected clutter
discrepancy 0.1652. Extra applications of the same moves help without
resolving this case; these pairs do not establish posterior calibration.
See `wp10_convergence.md`.

## WP11 — Targeted association/clutter mode moves

- [x] Design a reversible multi-frame proposal aimed at competing assignments
  and clutter decisions while retaining explicit null and existence states
- [x] Test detailed balance and posterior weights on the existing small exact
  oracles before any large-case claim
- [x] Freeze a new comparison on the unchanged hard case and untouched
  synthetic realizations, reporting mode and count coverage as well as cost

**Observed boundary:** the three-frame guided two-slot move passes all three
2,048-particle runs on the exactly enumerable four-slot bridge and the
finite-state detailed-balance checks. At 8,192 particles, all three
predeclared eight-frame two-seed pairs fail the inherited 0.10 gate: worst
0.3456 on the unchanged WP8 hard case, 0.3708 and 0.1439 on two untouched
synthetic realizations. Accepted tracklet state changes occur in only
about 1% of selected moves. The original case takes about 904 seconds/run,
against 432 seconds for the earlier paired mixture. This is a correct but
inefficient proposal under the tested protocol, not a posterior-calibration
claim. See `wp11_tracklet_validation.md`.

## WP11b — GP-predictive non-self tracklets

- [x] Condition the auxiliary localization proposal only on fixed detections
  outside a selected three-frame window, and exclude the current path with
  an exact non-self Hastings correction
- [x] Check finite-state detailed balance and three inherited small exact
  oracles before the reserved eight-frame inference
- [x] Compare two seeds against the paired-slot control on one previously
  unused synthetic realization with the same model and computational budget

**Observed boundary:** all three small exact-oracle runs pass, and accepted
tracklet state changes rise to about 0.40 per selected move. The reserved
two-seed pair nevertheless fails the inherited 0.10 gate, with worst
discrepancy 0.1827 (alive-count TV). The paired control also fails, at
0.2484 (clutter marginal). Six of seven gated discrepancies and the ungated
edge/clutter diagnostics improve relative to the control, but assigned-count
TV worsens slightly. This is a relative gain on one held-out synthetic
realization, not calibrated posterior mass or generalization. The next
methodological boundary is global count/existence and association-clutter
mode mixing, checked against exact small oracles before a new held-out case.
See `wp11b_conditional_validation.md`.

## WP11c — Full-history pair move pilot

- [x] Expose the existing whole-sequence two-slot multiple-try move as an
  opt-in alternative while preserving the local paired default
- [x] Check finite-state detailed balance and measure actual count-mode
  transitions on an exactly enumerable posterior at three fixed seeds
- [x] Apply the predeclared small-case early-stop gate before any new
  eight-frame holdout

**Observed boundary:** the full-history kernel is valid but its mean
accepted count-signature change rate is 0.0020, below the 0.01 pilot gate;
the local paired control yields 0.1270--0.1572 across the same three seeds.
The base proposal has negligible overlap with the small-case posterior.
No fresh large-case observation was opened. Next, test a likelihood-guided
sequential or conditional path update on small exact oracles before
reserving a new synthetic realization. See `wp11c_full_history_pilot.md`.

## WP11d — Conditional particle-Gibbs pair histories

- [x] Construct a likelihood-guided two-slot full-history conditional SMC
  proposal with an exact GP predictive increment and retained reference path
- [x] Verify incremental density identity and an enumerated two-frame
  detailed-balance transition matrix at three temperatures
- [x] Pass the frozen three-seed, 269-state small-case mixing/stationarity
  gate without changing the default sampler
- [x] Profile and screen operational mixing/cost on the already exposed WP8
  difficult eight-frame observations before reserving any new holdout
- [ ] Reduce global-update cost and pass a new development protocol before
  attempting a large 8,192-particle validation

**Observed small-case boundary:** count-signature transitions increase to 0.4118
per selected update (mean over three exact-posterior pilot seeds), from
0.1270--0.1572 for the local comparator. Exact-count TV remains within
the predeclared 0.0877 envelope, but this small-posterior pilot is around
an order of magnitude slower. This validates the small-kernel target and
mixing, not full SMC recovery or eight-frame calibration. No fresh holdout
was opened. See `wp11d_conditional_pair_pilot.md`.

**Unchanged-WP8 development result:** caching and removing unnecessary HMM
construction accelerate the pair-PG kernel by about 3× in a 32-row profile.
On two 512-particle, eight-frame approximate-posterior starts, count changes
rise to 0.708/0.661 versus 0.199/0.165 for the local control. The between-
seed alive-count TV drops to 0.1719 from 0.4180, yet the pair-update runtime
costs 15.7×/18.9× more, failing the predeclared 10× operational gate.
Stopped before large inference and any new holdout. See
`wp11d_wp8_development.md`.

## WP12 — Open in-vivo trajectories

- Start with ULMShare `tracks.json.xz`
- Use existing tracks as reference outputs, not unquestioned ground truth
- Create controlled corruptions and held-out prediction tasks
- Compare with classical ULM tracking pipelines

## WP13 — Image/IQ likelihood

- Select a small PALA or ULMShare raw subset
- Define the PSF/acoustic observation model and noise assumptions explicitly
- Assess whether the added physical fidelity justifies a separate article

Raw-IQ inference is intentionally outside the critical path of the first
localization-level paper.

