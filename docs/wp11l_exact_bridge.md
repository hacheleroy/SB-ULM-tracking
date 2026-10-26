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
