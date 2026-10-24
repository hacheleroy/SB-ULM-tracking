# WP9 — Counterfactual diagnostic of the hard synthetic case

## Frozen question and design

The WP8 cluttered staggered-crossing case failed the inherited 0.10
independent-run gate on seeds 100 and 200, including expected clutter
(0.2013) and label-invariant alive-count TV (0.1511). Which single controlled
change to that exact realization reduces the two-seed discrepancy?

The WP8 observations are the common baseline. Each arm reuses their retained
true coordinates, clutter coordinates and ordering whenever applicable, and
changes only one generative factor:

| Arm | Counterfactual intervention | Realized true / clutter detections |
|---|---|---:|
| Remove observed clutter | Delete all seven clutter points and change the model's clutter mean from 0.45 to 0.15 per frame | 16 / 0 |
| Complete true detections | Add each of the nine missed true localizations from the same noisy source and change the model's detection probability from 0.65 to 0.95 | 25 / 7 |
| Widen crossing | Move each true localization by its known clean-trajectory offset, increasing midpoint spacing from 0.006 to 0.010 mm while preserving its noise, detection status and the clutter points | 16 / 7 |
| WP7 existence schedule | Use the 20-slot-frame WP7 truth mask instead of the 25-slot-frame staggered mask; drop three true observations in now-inactive slots, retaining all others | 13 / 7 |

The widened trajectories and source localization noise share trajectory seed
23, and all points remain in the original ROI. The alternate existence mask
is a subset of the original mask, so no new random detection is required.
The Matérn prior calibration, localization noise scale, ROI, birth/death
prior, four slots and eight frames remain exactly those of the WP8 hard case
in every arm. Changes in detection probability or clutter mean accompany
their respective observation-level interventions so the likelihood describes
the intended counterfactual acquisition regime. All observation hashes and
cardinalities are fixed in `src/sbulm/wp9_ablation.py` before sampling.

The paired sampler and 8,192-particle budget are unchanged. Reuse exactly
the two original independent seeds, 100 and 200, and compute the seven WP8
pairwise discrepancies with the same 0.10 threshold. This is a **diagnostic
two-seed screen**; an arm below 0.10 is a candidate for a separate six-seed
validation, not a pass of the WP8 six-run gate. No arm is selected or
modified adaptively while this screen runs. If every arm remains above
0.10, no tested single intervention resolves the observed failure.

These are counterfactual manipulations of one deterministic synthetic
realization, not estimates of population-level causal effects. In
particular, deleting clutter and filling missed detections also alter the
number of observed points, and switching masks changes the latent truth.
The prior is still calibrated with simulation truth, so these experiments
cannot establish in-vivo posterior calibration.

```bash
python -m sbulm.wp9_ablation \
  --wp8-output results/wp8_generalization.json \
  --output results/wp9_ablation.json
```

The JSON checkpoint records metadata and hashes, the frozen WP8 failed pair,
both per-seed summaries per arm, and their pairwise metrics. It resumes
without rerunning completed seeds.

## Results

Every predeclared arm fails the two-seed diagnostic gate. The largest of
the seven inherited discrepancies is:

| Observation set | Worst pairwise metric | Difference | Alive-count TV |
|---|---|---:|---:|
| Original WP8 failed case | Expected clutter | 0.2013 | 0.1511 |
| Remove observed clutter | Alive-count TV | 0.1801 | 0.1801 |
| Complete true detections | Expected clutter | 0.6143 | 0.3573 |
| Widen crossing | Expected clutter | 0.3907 | 0.3218 |
| WP7 existence schedule | Expected clutter | 0.3832 | 0.2659 |

Removing clutter reduces the pair's expected-clutter difference from
0.2013 to 0.0880, but alive-count TV remains 0.1801 (>0.10). Completing
all nine missed true localizations increases the worst discrepancy to
0.6143 (expected clutter in frame 6); more detections did not make the
unchanged sampler reliable in this case. Widening the crossing preserves
the original cardinalities and clutter yet leaves a 0.3907 clutter
difference. Switching only the existence schedule also fails at 0.3832.
The full seven metrics, per-seed summaries, observed-data hashes and
runtimes are retained in the JSON checkpoint.

No tested *single* intervention resolves the original two-seed
instability. This is neither a causal decomposition of the original
failure nor a claim that clutter, missingness, geometry or birth/death
structure are irrelevant: each intervention defines a different
posterior, and the tested changes may interact. The next controlled
diagnostic should assess particle-count and move-mixing sensitivity on
the **unchanged WP8 failed observation set**, while preserving its two
original seeds and seven metrics. A modified proposal should then pass
the existing small exact-oracle checks before claiming any large-case
improvement. In-vivo uncertainty claims remain premature.
