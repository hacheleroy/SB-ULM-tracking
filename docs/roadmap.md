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

## WP11e — Sparse global pair updates

- [x] Add a state-independent, opt-in 10% conditional-PG fraction to the
  existing paired move, retaining the default local kernel and diagnostics
- [x] Verify zero-fraction equivalence and small exact-posterior count
  behavior before development inference
- [x] Freeze a new eight-frame development observation and two same-start
  paired comparisons, with a 3× runtime and movement gate
- [ ] Establish controlled cost and stable count-mode behavior before
  reserving another expensive high-particle/held-out test

**Observed boundary:** the sparse mixture raises count-signature change
rates on both development seeds (0.246/0.244 versus 0.208/0.188), but its
runtime ratios are 1.26× and 4.19×, failing the frozen 3× gate on the
second seed. Two-seed count agreement barely changes and remains far
from a calibrated eight-frame claim. The development result is preserved
as `stopped_before_holdout`. See `wp11e_selective_pg_development.md`.

## WP11f — Cost-controlled four-path selective PG

- [x] Reproduce the exposed WP11e baseline hashes and compare local versus
  sparse PG in four interleaved CPU-time repeats per seed
- [x] Pass a frozen four-versus-eight internal-path exact-oracle screen
- [x] Freeze a new eight-frame development observation before inference
- [x] Pass its two-seed same-start count-movement and median-CPU-cost gate
- [x] Test independent high-particle stability on an already exposed hard
  observation before contemplating a truly untouched holdout

**Observed boundary:** median eight-path sparse/local CPU ratios are 2.262
and 2.609 on the previously exposed WP11e data, despite variable wall
ratios. Four-path PG preserves the small exact count histogram and gives
0.275 count-mode changes at 0.632 of the eight-path CPU cost. The new
development case passes a two-seed, 10%-PG operational gate at 1.567×
and 1.868× median CPU cost. Assigned-count and clutter cross-run
discrepancies improve but alive-count TV worsens to 0.223 from 0.208;
no eight-frame posterior-calibration or stability claim follows.
See `wp11f_cost_path_count_development.md`.

## WP11g — Frozen 8,192-particle exposed-case diagnostic

- [x] Reuse and verify the original failed WP8 two-seed control checkpoint
- [x] Run independent seeds 100 and 200 with the frozen 10% four-path PG
  mixture and a 1,800-process-CPU-second cap per seed
- [x] Apply the unchanged seven label-free metrics and 0.10 limit; stop on
  the first failed pair without opening a new observation
- [ ] Resolve the remaining clutter and alive-count mode disagreements
  before a new holdout or eight-frame calibration claim

**Observed boundary:** all seven seed-pair discrepancies decrease versus
the archived control, but the candidate still fails on expected clutter
(0.1335) and alive-count histogram TV (0.1171). Each candidate seed
costs about 1,460 process CPU seconds at 8,192 particles; archived and
current wall times are not a controlled speed comparison. The status is
`failed_early_before_new_holdout`, not a larger-case stability pass. See
`wp11g_exposed_wp8_high_particle.md`.

## WP11h — Passive joint-mode diagnosis of WP11g

- [x] Reproduce both exposed WP11g seeds with an RNG-free observer and the
  original 1,800-process-CPU-second cap
- [x] Preserve full synthetic associations, label-invariant count signatures,
  stage joint distributions, and local/PG mode transitions
- [x] Decompose the remaining frame-six clutter and alive-count disagreements
  without changing the original stability verdict or opening a holdout

**Observed mechanism:** 0.0783 of the 0.1335 clutter gap comes from different
weights on low- and high-alive hypotheses; 0.0553 persists within those
bands, with differing detection-subset assignments. The discrepancy grows
mostly after beta about 0.75, despite appreciable PG mode transitions.
The next method test is a fixed-budget, beta-dependent allocation of the
existing PG update, first on the exact oracle and new development data. See
`wp11h_joint_mode_diagnostic.md`.

## WP11i — Late-stage allocation of four-path PG

- [x] Add an opt-in, beta-only PG selection schedule while preserving the
  default constant mixture and exact tempered-target invariance
- [x] Pass the frozen 269-state exact-oracle movement and count-TV screen
- [x] Run both arms on two seeds of a fresh, hash-frozen development case
- [ ] Pass all frozen development guardrails before another exposed-case
  8,192-particle test

**Observed boundary:** the late allocation reduced the fresh-case
cross-seed clutter difference from 0.1348 to 0.0645 and alive-count TV
from 0.1973 to 0.1523, while staying within the CPU and PG-selection
budget gates. Maximum death-marginal disagreement increased by 0.02148,
exceeding its allowed +0.020. The status is
`stopped_before_high_particle_test`; no new high-particle or holdout
observation was opened. See `wp11i_late_pg_development.md`.

## WP11j — Exact ancestor sampling inside four-path PG

- [x] Freeze a new observation, exact-oracle gate, cost caps, and all seven
  development comparisons on a branch before inference
- [x] Check full-suffix ancestor weights and exact small-case stationarity;
  pass the three-seed exact-oracle movement gate
- [x] Complete both arms for two seeds on the independent fresh case
- [ ] Pass its frozen paired stability guardrails before another exposed
  8,192-particle test

**Observed boundary:** PGAS increased exact-oracle first-frame association
movement from 0.296 to 0.488, and all six count TVs passed. On the fresh
512-particle development case, clutter difference fell from 0.350 to 0.271,
but alive-count TV rose from 0.225 to 0.318; missed and death differences
also rose beyond the predeclared +0.020 guardrail. CPU and selection caps
passed. The status is `stopped_before_high_particle_test`; the default
sampler is unchanged, and no new high-particle or holdout run was opened.
See `wp11j_pgas_ancestor_development.md`.

## WP11k — Variance and calibration audit

- [x] Freeze a medium exact reference and two independent eight-frame
  observations, seeds, particle counts, CPU caps, and read-only joint observer
  before inference
- [x] Compare six seeds at 512 and 2,048 particles against the exact
  posterior and independent-draw finite-sample envelope
- [x] Measure four-seed seven-metric spread at 512 and 2,048 on the primary
  eight-frame case and at 512 on a separate stress case
- [x] Establish absolute eight-frame mode-weight calibration on an
  independently frozen two-slot bridge before another move-selection gate

**Observed boundary:** all six exact-case seeds at both particle counts were
within the independent-draw 95th-percentile envelope for alive-total TV,
count-signature TV, and maximum clutter error. On the primary eight-frame
observation, median pairwise alive-count TV fell from 0.3145 to 0.1738 at
2,048 particles, while median clutter difference remained about 0.309 at
both sizes; the worst clutter pair was still 0.4072 at 2,048. A separate
512-particle stress observation had smaller spread, but its frame six had
only one detection. This is an observation-dependent variance diagnosis,
not an eight-frame calibration claim or a new sampler gate. The default
sampler and holdout are unchanged. See `wp11k_variance_calibration.md`.

## WP11l — Exact eight-frame mode-weight bridge

- [x] Freeze a two-slot/eight-frame observation, eight-model clutter grid,
  exact mode definitions, selection rule, independent-draw baseline, seeds,
  and CPU caps before inference
- [x] Enumerate all 667,989 states and verify a selected bridge with both
  competing alive/clutter modes above 0.10 exact posterior mass
- [x] Measure absolute mode weights and full joint distributions across six
  seeds at 512 and 2,048 particles with the unchanged WP11g sampler
- [x] Extend independent absolute calibration to a three-slot/eight-frame
  bridge before selecting a new sampler move

**Observed boundary:** the frozen exact selection chose clutter rate 1.50,
with mode masses 0.3825 and 0.3386; the second mode rises sharply after
beta 0.75. At 2,048 particles all six maximum mode errors were at most
0.0192 versus a 0.0260 IID 95th-percentile reference, and all full-joint
TVs were below their IID envelope. One of six 512-particle mode errors and
one marginal 2,048-particle clutter-mean error exceeded their respective
descriptive envelopes. This validates mode weights on the reduced exact
bridge, not on WP11k's four-slot primary case. No new move was selected;
the default sampler and untouched holdout remain unchanged. See
`wp11l_exact_bridge.md`.

## WP11m — Three-slot exact eight-frame bridge

- [x] Freeze a three-slot observation with sixth-frame clutter competition,
  exact-rate grid, mode and late-emergence validity criteria, seeds, and caps
- [x] Score all 8,375,839 labelled states in bounded batches and validate
  the exact count-rate reweighting and base normalizers
- [x] Compare twelve independent sampler runs with absolute exact mode,
  alive/clutter joint, and evidence references
- [x] Extend the absolute reference to a denser observation or a four-slot
  geometry before drawing conclusions about WP11k's primary case

**Observed boundary:** the exact rule chose clutter rate 0.35, with P(A)
0.2675 and P(B) 0.2147; B grows from 0.0538 at beta 0.75. At 2,048
particles, the largest three-mode error was 0.024002, just below the
0.024224 simulated IID 95th percentile; all six full-joint TVs were below
their IID envelope. One sixth-frame clutter-mean error exceeded its
descriptive envelope. This sparse six-detection bridge has three slots but
does not explain the four-slot/19-detection instability. No new move was
selected and the holdout remains closed. See `wp11m_three_slot_bridge.md`.

## WP11n — Denser three-slot exact bridge

- [x] Freeze the seven-detection observation, exact support and clutter grid,
  validity rule, reference checks, seeds, CPU caps, and diagnostic flag before
  posterior evaluation
- [x] Enumerate all 20,768,284 labelled states and validate the independent
  absolute posterior
- [x] Compare six seeds at 512 and 2,048 particles against the exact
  alive/clutter mode and joint references

The added early detection closes one of WP11m's blank frames. This is a
budgeted intermediate geometry between its six detections and WP11k's
four-slot/19-detection observation. The new absolute comparison is described
in `wp11n_dense_three_slot_bridge.md`; it does not open the high-particle
test or untouched holdout.

**Observed boundary:** the exact rate rule chose 0.35, with mode A/B masses
0.2452 and 0.2697 and a late-rising B mode. All six 2,048-particle
maximum mode errors fell below the exact IID 95th-percentile reference,
but two full alive/clutter joint TVs exceeded their descriptive reference.
The frozen mode-error flag did not fire and no new move was selected. This
suggests further joint-cell scrutiny on a more demanding independently
referenced geometry, without treating the seven-detection result as proof
about WP11k's 19-detection case. See `wp11n_dense_three_slot_bridge.md`.

## WP11o — Paired four-slot absolute joint-cell references

- [x] Freeze two distinct five-frame/four-slot observations, their exact
  support and rate grid, 63-cell comparison, familywise cell threshold,
  replication rules, seeds and CPU caps before inference
- [x] Enumerate each 24,193,016-state support and validate both independent
  absolute references before any sampler run
- [x] Compare six seeds at 512 and 2,048 particles on each observation,
  report joint-tail and same-signed-cell replication flags

This raises the slot count to four while keeping exact enumeration feasible.
The five-frame/six-detection cases remain distinct from WP11k's eight-frame,
19-detection primary case. See `wp11o_four_slot_joint_reference.md`.

**Observed boundary:** both exact references selected clutter rate 1.0,
with A/B masses 0.2556/0.2953 and 0.2672/0.2904. In the twelve
2,048-particle runs, no full-joint TV or maximum-cell error exceeded its
case-specific exact IID 95th percentile. Neither six-run mean had a cell
beyond its familywise exact threshold; both frozen replication flags were
false. One primary alive-total TV exceeded its descriptive q95. These
five-frame/six-detection cases do not replicate the WP11n joint tail, but
they cannot establish calibration on WP11k's eight-frame/19-detection
primary geometry. No new move was selected and the holdout remains closed.

## WP11p — Exact four-slot/eight-frame late-mode bridge

- [x] Freeze the eight-frame observation, canonical exact reference method,
  difficulty validation, 99-cell comparisons, seeds, diagnostic flags and
  resource caps before posterior evaluation
- [x] Validate the symmetry-weighted exact posterior and check the frozen
  late competing mode criterion before any sampler run
- [x] Stop sampler inference when the bridge fails that criterion

This restores eight-frame temporal geometry at four slots while retaining
tractable exact evidence through slot-label symmetry. It has six detections,
not WP11k's 19, and is a diagnostic bridge. See
`wp11p_eight_frame_exact_reference.md`.

The exact reference passed its count and analytic checks, but the selected
rate 0.2 gave P(A)/P(B) = 0.1973/0.0563. P(B) failed the predeclared 0.10
minimum, so no sampler run was performed. The checkpoint preserves the
invalid bridge and the next design requires its own frozen protocol.

## WP11q — Seven-detection, four-slot/eight-frame absolute bridge

- [x] Freeze a distinct four-slot posterior, extended clutter grid,
  exact lookup validation, mode criterion, joint comparison, and seeds
  before posterior evaluation
- [x] Validate and checkpoint the exact 2,460,244,269-labelled-state
  posterior before sampler inference
- [x] Compare twelve unchanged sampler runs with the absolute
  reference under the frozen diagnostic flags

WP11p failed as designed; this is a separate proposal informed by that
outcome. It adds one middle-frame detection and reuses WP11n's observation
with a four-slot latent model. See
`wp11q_seven_detection_exact_reference.md`.

The exact weighted pass succeeded in 177.6 CPU seconds. At the selected
rate 0.35, P(A)/P(B) = 0.1984/0.1354; P(B) was 0.00123 at beta 0.5.
The frozen difficulty rule passed, and the independent reference was
checkpointed before any WP11q sampler run.

At 2,048 particles, only one of six maximum mode errors and one of six
full-joint TVs exceeded their exact-IID q95 values. Maximum six-seed mean
cell residual was 0.00405 against familywise q95 0.00729. All three
frozen diagnostic flags were false; 512-particle joint and cell metrics
showed 2/6 q95 exceedances. This valid seven-detection bridge does not
settle the 19-detection WP11k primary or select a new move. The next
reference should increase observation density while retaining independent
absolute validation.

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

