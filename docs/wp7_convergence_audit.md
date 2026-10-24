# WP7 — Four-slot oracle bridge and particle-scaling audit

## Question and frozen design

WP6's likelihood-informed paired-slot sampler passed the exact three-slot
checks, but its predeclared four-slot/eight-frame three-seed stability gate
missed on alive-count TV (0.10010 versus 0.10). A second three-seed group
passed. This audit **does not change the posterior, paired-move mixture, ESS
target, rejuvenation sweeps, observation generator, or 0.10 gate**.

There are two complementary tests:

1. A separate four-slot/four-frame sparse crossing, simulator seed 9,
   corruption seed 3, detection probability 0.35, clutter rate 0.15, and
   observed cardinalities `(1, 0, 1, 1)`. All 282,469 labelled association
   and existence hypotheses can be scored and normalized exactly. The same
   finite-draw TV, JS, mass-coverage, credible-set and 0.06 event-marginal
   checks used for the earlier exact experiments are applied to three
   independent 2,048-particle runs.
2. The **unchanged** WP6 four-slot/eight-frame observations (cardinalities
   `(0, 2, 4, 3, 4, 2, 1, 1)`) at higher particle counts. The seed groups
   100–102 and 200–202 correspond to the original frozen and held-out
   2,048-particle groups. Pairwise differences are assessed on the same five
   event marginals and two count histograms, with every metric required to be
   at most 0.10. Additional diagnostics report stage ESS, resampling-parent
   diversity, final distinct states, paired-move changes, runtime and
   label-invariant per-frame alive/assigned count signatures.

The four-frame oracle tests correctness on a **different and much sparser
problem**. Agreement there cannot prove that the eight-frame posterior is
calibrated. Eight-frame cross-run agreement, even at larger particle counts,
is a necessary stability check and not an exact posterior oracle.

## Reproducibility

```bash
python -m sbulm.wp7_convergence_audit --bridge-only \
  --bridge-particles 2048 --bridge-seeds 300 301 302

python -m sbulm.wp7_convergence_audit --large-only \
  --large-particles 4096 --large-seed-starts 100 200 --large-repeats 3

# Diagnostic stress pair selected after inspecting the 4,096-particle runs:
python -m sbulm.wp7_convergence_audit --large-only \
  --large-particles 8192 --large-seeds 200 201
```

The audit is checkpointed in `results/wp7_convergence_audit.json` after each
run and refuses to resume against changed observation hashes. Later
particle counts or seed groups may be added without rerunning completed
experiments. The original WP5/WP6 result files remain unchanged.

## Results and decision

All three 2,048-particle runs against the new exact four-slot/four-frame
posterior pass the inherited finite-draw and event checks. The maximum
absolute event-marginal errors are 0.0253, 0.0308, and 0.0282 (limit 0.06).
The exact posterior contains 282,469 labelled states and has effective
support about 9,965, so only a small fraction of full states can be visited
with 2,048 draws. Total variation around 0.86 is in line with the exact
multinomial finite-draw reference; it must not be interpreted as close
pointwise mass recovery. The inherited credible-set-recall lower bound is
zero on this especially broad posterior and is not an informative gate here.

| Four-slot/eight-frame maximum disagreement | 2,048 frozen | 2,048 held-out | 4,096 frozen | 4,096 held-out | 4,096 combined six |
|---|---:|---:|---:|---:|---:|
| Existence | 0.07666 | 0.06348 | 0.04468 | 0.06006 | 0.07227 |
| Missed | 0.07666 | 0.05762 | 0.04980 | 0.06836 | 0.07690 |
| Clutter | 0.06445 | 0.05371 | 0.05444 | **0.10840** | **0.10840** |
| Birth | 0.07666 | 0.05908 | 0.04590 | 0.03394 | 0.06909 |
| Death | 0.06055 | 0.04980 | 0.04492 | 0.05566 | 0.06226 |
| Alive-count TV | **0.10010** | 0.07715 | 0.07324 | **0.14185** | **0.14526** |
| Assigned-count TV | 0.05322 | 0.03516 | 0.05347 | **0.11816** | **0.11816** |

The 2,048-particle columns preserve WP6's independently reported triplets.
Each column's limit is 0.10. The 4,096-particle frozen triplet
passes, but the held-out triplet and combined six runs fail. Seed 200 has
20.068 mean alive slot-frames versus 19.386–19.719 for seeds 100–102;
the difference is label-invariant and cannot be eliminated by slot relabelling.
Similar evidence estimates and high final-state uniqueness do not establish
correct relative mass between count modes. Average runtime is about 313 s
per 4,096-particle run, against about 129 s at 2,048 in WP6.

The exact-target ceiling now includes this **particular sparse four-slot,
four-frame benchmark**, in addition to the earlier three-slot/five- and
six-frame benchmarks. There is still no exact eight-frame oracle or reliable
six-run stability at four slots/eight frames. The original WP6 frozen failure
remains a failure; the 4,096-particle frozen pass does not erase it. A targeted
8,192-particle repeat of the discordant seeds 200 and 201 is reported below
as a diagnostic: their alive-count TV falls from 0.14185 at 4,096 to
0.02673 at 8,192, clutter disagreement from 0.10840 to 0.02588, and
assigned-count TV from 0.11816 to 0.02063. Because these seeds were selected
after seeing 4,096-particle results, this pair alone cannot certify stable
posterior weights. The original two triplets were therefore repeated at 8,192
particles without changing the sampler or data:

| Eight-frame maximum disagreement at 8,192 particles | Frozen 100–102 | Held-out 200–202 | All six | Limit |
|---|---:|---:|---:|---:|
| Existence | 0.03516 | 0.03845 | 0.04785 | 0.10 |
| Missed | 0.02490 | 0.02625 | 0.03381 | 0.10 |
| Clutter | 0.04297 | 0.04260 | **0.05566** | 0.10 |
| Birth | 0.02173 | 0.03198 | 0.03711 | 0.10 |
| Death | 0.03320 | 0.04663 | 0.04712 | 0.10 |
| Alive-count TV | 0.03345 | 0.04797 | 0.04822 | 0.10 |
| Assigned-count TV | 0.04797 | 0.02429 | 0.04797 | 0.10 |

All seven checks pass within each three-seed group and across all six runs.
This is the first reproducible four-slot/eight-frame result under the stated
0.10 criterion, **at 8,192 particles on this one fixed synthetic dataset**.
It does not retroactively pass WP6's failed 2,048-particle gate, establish an
exact eight-frame mode-weight oracle, or guarantee comparable performance in
other crossing/corruption regimes. The exact calibrated envelope remains the
three-slot cases plus the sparse four-slot/four-frame bridge; the larger case
has independent-run stability evidence only. Higher count is costly: the six
8,192-particle runs took 405–1,132 s each, with substantial shared-machine
contention; their wall-clock ratios should not be read as clean scaling laws.

The next bounded validation is to freeze several new crossing and corruption
conditions, including increased misses and clutter, and repeat independent
stability checks. Only if those results remain stable should uncertainty
claims be explored on open in-vivo ULM tracks; those tracks are not exact
posterior ground truth.
