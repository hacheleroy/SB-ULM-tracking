# WP11l — Exact eight-frame alive/clutter bridge

WP11k found that the unchanged four-path PG sampler stayed within an IID
finite-sample envelope on a four-frame exact reference, yet four independent
eight-frame runs still disagreed on sixth-frame clutter by about 0.31 at
2,048 particles. This study measures **absolute mode-weight error** on a new
eight-frame model small enough for complete enumeration. The protocol and
runner are committed before computing its exact posterior or sampling it.

## Frozen construction

The two exchangeable slots have eight frames separated by 1 ms. The detection
arrays in coordinate units are:

| Frame (zero based) | Detections |
| ---: | --- |
| 0 | `(-0.018, 0.001)` |
| 1 | none |
| 2 | `(-0.008, 0.002)`, `(0.010, -0.001)` |
| 3 | `(-0.003, 0.001)`, `(0.005, -0.002)` |
| 4 | `(0.000, 0.001)` |
| 5 | `(-0.004, 0.001)`, `(0.004, 0.000)` |
| 6 | `(0.011, -0.001)` |
| 7 | none |

The observation SHA-256 from `_case_metadata` is
`9b99c281f85c2265a6c10d333379e90eb80bc928b273d9d4c76bdf76f735e77e`;
the support size is **667,989**. Use localization standard deviation 0.005,
Matérn coordinate scales `(0.02, 0.01)`, length scale 0.003, detection
probability 0.55, ROI `[-0.04, 0.04]` in each coordinate, and the existing
default birth/death prior. The only varied model parameter is expected
clutter per frame, fixed in advance to
`[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0]`. These are eight model
variants of **one observation**, not eight independent observations.

Define the label-invariant macro modes from the total number of alive
slot-frames and the sixth-frame clutter count:

- **A:** at most 10 alive slot-frames and at least one of the two sixth-frame
  detections is clutter.
- **B:** at least 11 alive slot-frames and neither sixth-frame detection is
  clutter.
- **Other:** all remaining association states.

Exactly enumerate all eight model variants before running the sampler. Choose
the grid member maximizing `min(P(A), P(B))` under its exact posterior; ties
go to the smaller clutter rate. The bridge is valid only if **both** selected
mode masses are at least 0.10. If it fails this validity check, stop and
record `bridge_invalid_no_competing_modes`; do not revise the grid, modes,
observation, or rule after seeing exact or sampler results. Report the exact
mode weights for **all** grid members to expose the selection.

For a valid bridge, enumerate exact mode weights at beta
`[0, 0.25, 0.5, 0.75, 0.9, 1]` using the SMC's normalized
birth/death/count base and tempered collapsed Matérn likelihood. Check the
enumerated base normalizer against the model's analytical base normalizer.
At beta one, record the full joint histogram of alive slot-frames by
sixth-frame clutter, plus the three macro-mode, alive, and clutter marginals.

Draw 500 independent multinomial samples from the exact posterior at each
particle count, with seed 30303. The median and 95th percentile of absolute
maximum mode error, three-mode TV, full joint TV, alive-total TV, and
sixth-frame mean-clutter error are descriptive finite-sample baselines.
Then run the **unchanged** WP11g 10% four-path PG sampler in separate
single-thread processes, 512 particles and seeds 900–905 first, followed by
2,048 particles and the same six seeds. Reference enumeration has a 1,800
process-CPU-second cap; sampler runs have 300 and 900 second caps. A passive
observer records stagewise A/B/other weights, resampling effects, and accepted
A↔B moves selected by PG versus other paired proposals.

The primary outputs are each seed's signed error in P(A) and P(B), three-mode
TV, the six-seed ranges and mean signed errors, and the number above each
IID 95th percentile. Also report full joint TV, alive-total TV, clutter mean
error, log-evidence error, and costs. This is a diagnostic, **not** a new
candidate-selection threshold. A two-slot exact bridge cannot establish
calibration on the original four-slot observation. Do not open the WP8
exposed case, an 8,192-particle run, or the untouched holdout.

Implementation: `src/sbulm/wp11l_exact_bridge.py`. The result checkpoint is
`results/wp11l_exact_bridge.json`.

## Exact reference and measured errors

Enumeration took 132.6 process CPU seconds and confirmed the analytical base
normalizer for every grid member. The complete predeclared grid gave:

| Clutter/frame | Exact P(A) | Exact P(B) | Other |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.0345 | 0.8790 | 0.0865 |
| 0.35 | 0.0658 | 0.7953 | 0.1389 |
| 0.50 | 0.1016 | 0.7172 | 0.1813 |
| 0.75 | 0.1686 | 0.5992 | 0.2322 |
| 1.00 | 0.2406 | 0.4970 | 0.2623 |
| **1.50** | **0.3825** | **0.3386** | 0.2789 |
| 2.25 | 0.5569 | 0.1924 | 0.2507 |
| 3.00 | 0.6794 | 0.1138 | 0.2068 |

The frozen balance rule selected 1.50; both mode masses exceeded 0.10.
This is a deliberately conditioned synthetic observation under a moderately
cluttered model, not a fresh draw from that model's typical-case distribution.
The exact mode B mass is only 0.00635 at beta 0.75, rises to 0.1130 at beta
0.90, and reaches 0.3386 at beta one. The competing mode indeed appears
late in tempering. The exact log evidence is 44.0060.

The table compares the six-seed sampler median and range with the 95th
percentile from 500 independent exact-posterior multinomial samples at each
particle count. The exceedance column counts seeds above that reference;
it is descriptive and is not a familywise hypothesis test.

| Particles | Absolute metric | IID 95th | Sampler median | Sampler range | Above IID 95th |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Maximum macro-mode error / three-mode TV | 0.04725 | 0.02005 | 0.00358–0.05892 | 1/6 |
| 512 | Full alive/clutter joint TV | 0.10717 | 0.09506 | 0.08756–0.10571 | 0/6 |
| 512 | Alive-total TV | 0.07842 | 0.06674 | 0.04716–0.07634 | 0/6 |
| 512 | Absolute clutter-mean error | 0.04654 | 0.01307 | 0.00038–0.04139 | 0/6 |
| 2,048 | Maximum macro-mode error / three-mode TV | 0.02605 | 0.00734 | 0.00361–0.01924 | 0/6 |
| 2,048 | Full alive/clutter joint TV | 0.05461 | 0.04403 | 0.03750–0.05013 | 0/6 |
| 2,048 | Alive-total TV | 0.04015 | 0.03265 | 0.01849–0.03656 | 0/6 |
| 2,048 | Absolute clutter-mean error | 0.02385 | 0.01403 | 0.00256–0.02479 | 1/6 |

At 512 particles, mode A signed errors spanned -0.0309 to +0.0589 and
mode B errors -0.0163 to +0.0130. Their mean signed errors were +0.0026
and -0.0049. At 2,048 particles, A ranged -0.0075 to +0.0174 and B
-0.0192 to +0.0032, with means +0.0015 and -0.0061. The single 2,048
clutter-mean exceedance (0.02479 versus 0.02385) is marginal relative to
the Monte Carlo estimate of the reference percentile. Estimated-minus-exact
log evidence ranged from -0.2237 to +0.2239 at 512 and -0.1011 to +0.0767
at 2,048. Mean process CPU time was 24.4 versus 102.8 seconds per run.
The passive observer recorded accepted A→B and B→A crossings in both PG
and other paired proposals; these counts show movement, not independent
samples or proof of mixing.

**Decision:** `absolute_bridge_audit_complete_no_move_selected`. This
two-slot eight-frame model has a genuine late two-mode exact posterior. The
unchanged sampler's final macro-mode and full-joint errors were mostly
consistent with finite independent sampling, especially at 2,048 particles.
There is no compelling systematic mode-weight bias on this bridge. The
four-slot WP11k primary case still has large between-run clutter spread;
the present result does not explain or cure it. A new sampler move would be
premature based on this smaller case. The next diagnostic should increase
slot and observation complexity while retaining an independent absolute
reference, for example streaming exact aggregation on a carefully budgeted
three-slot/eight-frame observation, or another independently validated
high-dimensional reference. Its observations and criteria must be frozen
before inference. No 8,192-particle run or untouched holdout was used.
