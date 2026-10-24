# WP8 — Held-out synthetic generalization

## Frozen question

WP7 found independent-run stability on one simultaneous four-slot/eight-frame
crossing at 8,192 particles, but not at 4,096. That is not a transfer claim.
This study asks whether the **unchanged** likelihood-informed paired sampler
also produces reproducible posterior summaries on new observation and
trajectory realizations with harder misses, clutter and geometry.

The following cases, generated before inference, have eight frames, four
exchangeable birth/death slots, 1 ms frame period, 0.005 mm localization
standard deviation, the same physical speed and Matérn calibration rule,
ROI `[-0.04, 0.04]^2` mm, and the same birth/death prior as WP7:

| Case | Detection probability | Clutter mean/frame | Crossing midpoint offsets | Simulator/corruption seeds | True detections / clutter |
|---|---:|---:|---:|---|---:|
| More misses, simultaneous crossing | 0.50 | 0.15 | 0 | 17 / 11 | 11 / 1 |
| Staggered crossing and heavier clutter | 0.65 | 0.45 | 0.006 mm spacing | 23 / 12 | 16 / 7 |

The first case retains the WP7 staggered birth/death truth but realizes more
misses. The second uses a different contiguous birth/death pattern: one slot
persists across the whole window while others start and stop at different
frames. Its clutter is sampled from an uncapped Poisson process; the first
case retains the one-clutter-per-frame simulation guard used in WP7. Both
cases have distinct noisy localization realizations and shuffled detection
orders. Exact observed cardinalities, reference counts and SHA-256 hashes
are stored in `results/wp8_generalization.json`.

As in WP7, the synthetic harness calibrates the Matérn prior length scale
from the simulated trajectory spread and a fixed reference speed. This is
controlled, truth-informed hyperparameter calibration; it tests sampler
stability conditional on that prior, not whether those parameters can be
estimated prospectively from in-vivo observations.

## Unchanged sampler and decision rule

Use the WP6 paired move mixture, 8,192 particles, ESS target 0.80, eight
rejuvenation sweeps per stage and eight at the end. The independent sampler
seeds are fixed in this order:

```text
100, 200, 101, 201, 102, 202
```

For each case, compare all completed runs on the same seven summaries as
WP7: existence, missed-detection, expected-clutter, birth-time and
death-time marginals, plus alive-count and assigned-count histogram TV.
Every maximum pairwise discrepancy must be at most 0.10. The schedule
interleaves the two seed groups. If any metric exceeds 0.10, stop that case
immediately: adding runs can only increase a pairwise maximum and cannot
reverse failure. This is a fixed computational stopping rule, not threshold
tuning. A pass requires all six independent runs.

```bash
python -m sbulm.wp8_generalization \
  --output results/wp8_generalization.json --n-particles 8192
```

The JSON checkpoint records each seed separately and supports safe resume.
Cross-run agreement is **necessary but not sufficient** for posterior
calibration. The simulated truth provides task diagnostics, not an exact
posterior over the much larger eight-frame state space. No in-vivo uncertainty
claim follows directly from a pass here.

## Results

The miss-heavy simultaneous crossing passes the frozen gate over all six
independent 8,192-particle runs: its largest pairwise difference is 0.0497
(alive-count histogram TV), versus the 0.10 limit. Its other six maxima are
existence 0.0375, missed-detection 0.0326, expected-clutter 0.0081,
birth-time 0.0350, death-time 0.0253, and assigned-count TV 0.0201.
The true simulated count is 20 alive slot-frames, 11 retained true
detections and one clutter detection. The six posterior means for alive
slot-frames range from 18.70 to 18.95, and for assigned detections from
11.65 to 11.68. These truth comparisons are descriptive single-case
diagnostics rather than posterior-probability calibration tests.

The cluttered staggered crossing **fails** after the first two independent
seeds, 100 and 200. Expected clutter differs by 0.2013 in frame 3 (zero-based
index 2), versus the 0.10 limit. Existence (0.1338), missed-detection
(0.1373), and alive-count TV (0.1511) also fail; birth-time (0.0544),
death-time (0.0551), and assigned-count TV (0.0752) pass for this pair.
The true simulated count is 25 alive slot-frames, 16 retained true
detections and seven clutter detections. The two posterior means are 20.67
and 19.40 alive slot-frames and 14.76 and 14.53 assigned detections.
These mean-versus-truth discrepancies alone do not diagnose calibration;
the independent-run discrepancy is the decisive failure of this gate.

The four unused seeds were deliberately not run: a pairwise maximum cannot
decrease when additional runs are added. This is not evidence that the
sampler always fails under clutter or that 8,192 particles cannot suffice
under a different proposal; it shows that the previously observed stability
does **not** transfer to this particular changed geometry and corruption
realization under the frozen sampler and budget. The joint change of
clutter, detection rate, crossing offsets and existence schedule prevents
attributing the failure to any one factor. A subsequent, separately
predeclared ablation would be needed for that diagnosis. In particular,
final-state uniqueness above 0.995 in both runs did not ensure agreement
between their posterior summaries.
