# WP10 — Particle count versus rejuvenation on the frozen hard case

## Frozen diagnostic

WP8 failed on the four-slot/eight-frame cluttered staggered crossing with
8,192 particles and seeds 100 and 200: the maximum expected-clutter
difference was 0.2013 and alive-count histogram TV was 0.1511. WP9 found
that none of four single-factor counterfactual observation changes restored
two-seed agreement. This experiment changes **only the sampler's
computational budget** on the original WP8 failed observations (SHA-256
`afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae`).
The localization likelihood, Matérn prior, birth/death prior,
detection ordering, four slots, eight frames and ESS target 0.80 are fixed.

| Arm | Particles | Sweeps at each stage | Extra final sweeps | Move mixture |
|---|---:|---:|---:|---|
| WP8 failed baseline | 8,192 | 8 | 8 | WP6 paired mixture |
| Double particles | 16,384 | 8 | 8 | same |
| Double rejuvenation | 8,192 | 16 | 16 | same |

The second arm offers more opportunities for the existing reversible
association, interval, swap, blocked-slot and paired-slot proposals to
mix. It does not introduce a new proposal kernel or alter their relative
probabilities. This contrasts the two plausible resource bottlenecks:
particle diversity during tempering versus mutation/mode crossing after
resampling. It does not fully separate those mechanisms, because changing
one budget can affect the subsequent temperature schedule as well.

For both arms, use independent seeds 100 and 200 and the seven inherited
WP8 pairwise metrics: maximum differences in existence, missed-detection,
expected-clutter, birth-time and death-time marginals, and TV distances
between alive-count and assigned-count histograms. The fixed gate is 0.10
for every metric. Record the tempering-stage count, minimum stage ESS,
parent diversity, final distinct-state fraction, paired-move acceptance
and actual state-change rates, and elapsed runtime. The arms and seed
order are frozen before inference. An arm below 0.10 on this pair is only
a **diagnostic lead**, not a six-run stability pass or exact posterior
calibration; an arm above 0.10 fails at this tested budget, not at every
possible budget.

```bash
python -m sbulm.wp10_convergence \
  --wp8-output results/wp8_generalization.json \
  --output results/wp10_convergence.json
```

The output checkpoint includes the original WP8 comparison, each new
seed and the pairwise summaries. The code checks the observation hash and
the baseline discrepancies before reusing them. The 16-sweep arm uses
the same reversible kernels that passed the existing small exact-oracle
checks; no new transition rule is claimed to be calibrated here.

## Results

Neither resource escalation brings the original seed pair below the
inherited 0.10 gate:

| Budget | Worst pairwise metric | Worst value | Expected-clutter difference | Alive-count TV | Mean time/run |
|---|---|---:|---:|---:|---:|
| WP8 baseline: 8,192, eight sweeps | Expected clutter | 0.2013 | 0.2013 | 0.1511 | 432 s |
| 16,384, eight sweeps | Alive-count TV | 0.2758 | 0.1700 | 0.2758 | 1,118 s |
| 8,192, 16 sweeps | Expected clutter | 0.1652 | 0.1652 | 0.0674 | 852 s |

Doubling particles alone does not rescue this pair: assigned-count TV also
reaches 0.1865. The two runs took 1,097 and 1,139 seconds with 20 and 21
tempering stages. The nonmonotonic change in finite-run discrepancies is
not evidence that more particles intrinsically worsen the posterior; it
shows failure at this particular doubled budget and pair of seeds.

Doubling rejuvenation improves six of the seven metrics below 0.10:
existence 0.0457, missed detection 0.0365, birth time 0.0364, death time
0.0222, alive-count TV 0.0674 and assigned-count TV 0.0643. Expected
clutter still differs by 0.1652 in frame 3 (zero-based index 2). The two
runs took 873 and 832 seconds with 21 and 20 stages. Both had more than
0.999 distinct final states, yet the clutter marginal remained inconsistent.
Extra applications of the existing move mixture therefore improve much
of the agreement but do not satisfy the frozen gate at this budget.

This suggests a remaining association/clutter mode that the present
mutations explore unevenly, not a proven diagnosis of a particular kernel.
Further resource escalation on these same observed seeds would be an
exploratory tuning exercise, not independent validation. A next method
step is to design a targeted, reversible multi-frame association/clutter
move, establish its finite-state correctness on the small exact oracles,
and then evaluate it on the unchanged hard case and untouched realizations
under a newly frozen protocol. No eight-frame exact posterior calibration
or in-vivo uncertainty claim follows from WP10.
