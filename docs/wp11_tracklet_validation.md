# WP11 — Reversible multi-frame association/clutter proposal

## Why this move

The inherited WP6 paired-slot update changes one or two frames at a time.
WP10 showed that doubling its rejuvenation sweeps improves six of seven
cross-run metrics on the unchanged WP8 cluttered staggered crossing but
leaves a frame-3 expected-clutter discrepancy of 0.1652. This does not
identify a particular deficient kernel. WP11 tests whether a three-frame,
temporally coherent alternative is more useful at the same particle and
sweep budget.

Two ordered slots and a contiguous three-frame window are selected with
probabilities independent of the particle state. Their existence masks,
all other slots, and all frames outside the window remain fixed. At every
selected frame, an exact finite list of legal assignments includes missed
detections and unused points as clutter. A finite-state auxiliary Markov
chain weighs assigned detections by the detection/clutter ratio and favors
nearby consecutive observations on the same slot with a broad Gaussian
displacement score. Its forward-filter/backward-sample draw can reassign
both slots over three frames. The auxiliary Gaussian is **only a proposal**;
the scientific target still uses the exact collapsed Matérn localization
likelihood, count density, and birth/death prior.

For a fixed selection, the complete context on which the auxiliary law
depends is unchanged by a proposal. Thus both directions share its same
normalizer, and the Metropolis log-ratio adds the old minus proposed
auxiliary path score. Every legal local assignment has positive proposal
probability. Other moves remain responsible for changing existence masks.
The selection of the slot pair and window has the same probability in both
directions. An exhaustive finite-state test verifies detailed balance for
every pair of conditional states at temperatures 0, 0.37, and 1; a separate
test verifies the reverse law and unchanged masks on the 282,469-state
four-slot/four-frame oracle bridge.

## Frozen evaluation, before eight-frame inference

The WP6 mixture is replaced with probabilities 0.15 frame association,
0.30 interval, 0.05 slot swap, 0.15 conditional blocked slot, 0.20 paired
slot, and 0.15 guided tracklet. There is no change to the target, Matérn
hyperparameters, observations, ESS target 0.80, eight sweeps per stage,
extra eight final sweeps, or 8,192 particles in the large comparisons.

First run 2,048 particles at seeds 300, 301, and 302 on the existing
four-slot/four-frame exact oracle (SHA-256
`62d3aeeafff58eb47404f87291829004f5bd498d07ab1494afa86ba77115b8ef`).
All three must pass the previously recorded WP7 finite-draw TV/JS/support
benchmarks and 0.06 maximum event-marginal error before a large run begins.
Per-detection clutter and adjacent-frame same-slot detection-edge errors
are additional descriptive diagnostics. These are label invariant.

Then use two independent seeds 100 and 200 on each predeclared eight-frame
observation set, in order:

| Case | Simulator seeds (trajectory, corruption) | Observation SHA-256 |
|---|---|---|
| Unchanged WP8 failed staggered crossing | 23, 12 | `afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae` |
| Previously unused realization A | 29, 18 | `c5c1ddd70435c6fdc4af36c0d2b4d490bb9cb59465e9d8a41d218e1462768d2c` |
| Previously unused realization B | 31, 19 | `c0301404d31cacaf6db0be996b8997324319ebce1225160092ba9f6117758d2a` |

For each pair, all seven inherited WP8 differences must be at most 0.10
to count as *two-seed diagnostic agreement*. Additional unthresholded
diagnostics compare per-detection clutter, adjacent-frame assignment edges,
and full histograms of per-frame alive/assigned count signatures, as well
as actual accepted tracklet state changes and elapsed runtime. A pair
below 0.10 is **not** the earlier six-run stability result, and neither
pairwise agreement nor a four-frame oracle proves an exact eight-frame
posterior or prospective performance on in-vivo data. The new realization
seeds and observation hashes were fixed before any WP11 large inference;
none of these runs is used to tune the move probabilities.

```bash
python -m sbulm.wp11_tracklet_validation --stage oracle
python -m sbulm.wp11_tracklet_validation --stage large
```

The result file checkpoints each completed seed, verifies the previous
WP7/WP8 artifacts and all observation hashes, and blocks large runs unless
every exact-oracle run passes. Large runs can be resumed without repeating
finished seeds.

## Results and interpretation

All three four-slot/four-frame oracle runs pass the inherited finite-draw
and event-marginal checks. Maximum event errors are 0.0287, 0.0189, and
0.0216 (limit 0.06). Additional maximum per-detection clutter errors are
0.0069, 0.0049, and 0.0073; adjacent same-slot association-edge errors are
0.0008, 0.0213, and 0.0237. The exact conditional detailed-balance test
passes at three temperatures. This validates the new kernel on the
enumerable cases; it does not establish an eight-frame oracle.

No eight-frame two-seed pair passes the seven-metric 0.10 diagnostic gate:

| Observations | Worst inherited metric | Value | Expected-clutter difference | Alive-count TV | Mean seconds/run |
|---|---|---:|---:|---:|---:|
| Original WP8 failed case, **old paired mixture** | Expected clutter | 0.2013 | 0.2013 | 0.1511 | 432 |
| Original WP8 case, **WP11** | Expected clutter | 0.3456 | 0.3456 | 0.0574 | 904 |
| Untouched realization A, **WP11** | Expected clutter | 0.3708 | 0.3708 | 0.0806 | 689 |
| Untouched realization B, **WP11** | Alive-count TV | 0.1439 | 0.0754 | 0.1439 | 692 |

For the original hard case, assigned-count TV is also above the gate at
0.1177. The largest expected-clutter discrepancy is again frame 3: the
two runs give 0.349 and 0.694. The additional maximum per-detection
clutter and adjacent association-edge disagreements are 0.2142 and
0.2864, respectively. On A, these extra disagreements are 0.1844 and
0.1775. On B, they are 0.0884 and 0.1228; its coarse alive-count TV
still fails. Per-frame alive/assigned count-signature histogram TV is
0.6224, 0.7864, and 0.5952 across the three pairs. These sparse, broad
signature histograms are **descriptive**, not exact full-mode TV scores;
finite-draw reference limits were not established for them.

The average *acceptance* probability of selected tracklet moves is
roughly 36–39% on the large cases, but the mean rate of an **accepted
state change per selected tracklet move** is only 1.17%, 1.00%, and 0.97%
across the original case, A, and B. Many accepted draws leave the
assignment unchanged. That low effective mutation rate, together with
904 seconds/run on the original case versus 432 seconds/run for WP8,
means this fixed guided proposal is not a practical improvement at the
tested budget. The outcome does not invalidate the exact target or the
reversible transition: it is a mixing and efficiency failure on these
eight-frame realizations.

Do not increase the particle count or retune this kernel on these same
seeds and describe the result as independent validation. A later method
iteration could construct more posterior-informed, explicitly non-self
multi-frame proposals, still with a computable reverse probability and
the same small-oracle-first discipline, then freeze genuinely new
observations and seeds for transfer. These data do not uniquely identify
which internal design choice causes the poor mixing; the observed
accepted-change rate is a diagnostic, not a causal proof.
