# WP11q — Seven-detection, four-slot/eight-frame absolute bridge

WP11p's frozen six-detection geometry failed its mode-balance criterion,
so it was stopped before sampler inference. This separate proposal adds a
middle-frame detection and broadens the clutter-rate grid. The observation
coordinates reproduce WP11n's seven-detection case, but the latent model
has **four rather than three slots**, so its posterior is a new exact
problem. The new threshold is an interpretable 16 alive slot-frames
(two slot-frames per frame); it is frozen here before evaluating this
four-slot posterior. Its beta-0.5 check matches WP11k's observation that
the difficult primary case was still concentrated in the low band near
the middle of tempering. The WP11p posterior motivated a separate design;
it is not reclassified as a pass.

## Frozen observation and model

Times are 0–7 ms, separated by 1 ms. Coordinates are in the model units:

| Frame | Detections |
| ---: | --- |
| 0 | `(-0.019, 0.001)` |
| 1 | `(-0.014, 0.001)` |
| 2 | `(-0.007, 0.002)` |
| 3 | none |
| 4 | `(0.001, 0.000)` |
| 5 | `(-0.005, 0.002)`, `(0.006, -0.001)` |
| 6 | `(0.012, -0.001)` |
| 7 | none |

Cardinalities are `[1,1,1,0,1,2,1,0]`; case-metadata detection SHA-256 is
`11a2f6f7c529f080ed0fdbb3eaf3da39c2b9170ce660e95f0b119de5d241734b`.
Use four exchangeable slots, localization standard deviation 0.005,
Matérn coordinate scales `(0.02, 0.01)`, time length 0.003, detection
probability 0.55, ROI `[-0.04,0.04]` in each coordinate, and default
birth/death prior. Only expected clutter per frame varies over the frozen
grid `[0.05,0.1,0.15,0.2,0.35,0.5,0.75,1,1.5,2.25,3]`.

Mode **A** has at most 15 alive slot-frames and at least one frame-six
(index 5) clutter detection. **B** has at least 16 alive slot-frames and
no frame-six clutter; other states are **other**. Select the rate maximizing
`min(exact P(A), exact P(B))`, ties to the smaller rate. A valid bridge
requires both selected masses at least 0.10 and exact P(B) at beta 0.5
at most half beta-one P(B), on `[0,.25,.5,.75,.9,1]`. If invalid, save
the exact result and stop before sampler inference. No adjustment of the
new observation, rate grid, or criterion follows its posterior evaluation.

## Exact reference and frozen comparison

The support contains **126,031,206 canonical** mask-assignment states,
representing **2,460,244,269 labelled** states. Enumerate nondecreasing
tuples of four birth/death interval masks and all compatible frame
assignments. Weight each state by `4! / product(factorial(repeated mask
counts))`. The Matérn likelihood factorizes over labelled slots; cache
the exact one-slot likelihood for every possible detection pattern and
sum its four lookup scores. Verify that lookup against the direct
collapsed model on deterministic blocks. Validate the weighted support
and alive-count histogram against full labelled enumeration on a small
case, and check the canonical and labelled state totals, direct endpoint
rate reweighting, and all **eleven** analytic HMM base normalizers. A
failed check or 12,000 CPU-second cap ends the reference without changing
the design. Save the exact **33×3** alive-total/frame-six-clutter table,
all modes, marginals, beta curve, and evidence before any sampler run.

For each 512 and 2,048 particle size, simulate 1,000 exact-posterior
multinomial IID draws (seed 34351). Compare each sampler run's maximum
three-mode absolute error, full joint TV, maximum absolute cell error,
alive-total TV and clutter-mean error against its descriptive IID q95.
Simulate 1,000 exact groups of six IID runs to obtain a familywise q95
for the maximum absolute six-run mean signed cell residual across 99
cells. Report every signed cell and per-seed error.

After a valid exact reference checkpoint, run the **unchanged WP11g 10%
four-path PG mixture** in twelve separate one-thread processes: seeds
1120–1125 at 512 particles, then those six at 2,048. Per-run CPU caps
are 600 and 1,600 seconds. A passive observer records mode weights,
resampling and accepted A↔B moves without consuming random numbers.

At 2,048 particles, the predeclared diagnostic flags are `mode_error_tail`
for at least 3/6 maximum mode errors above their exact IID q95;
`joint_tail` for at least 3/6 full-joint TVs above q95; and `cell_shift`
if a six-run mean signed cell exceeds the familywise q95 with the same
sign in both ordered three-seed halves. Report all flags separately.
These descriptive comparisons do not select a sampler move or open an
8,192-particle run or untouched holdout. This designed seven-detection
bridge is much sparser than WP11k's 19-detection primary and cannot
establish calibration on that case.

Runner: `src/sbulm/wp11q_seven_detection_exact_reference.py`. Checkpoint:
`results/wp11q_seven_detection_exact_reference.json`. This protocol and
runner are committed before evaluating the four-slot posterior.

## Exact reference checkpoint

The independent weighted pass covered all **126,031,206** canonical states
and **2,460,244,269** represented labelled states in **177.6 CPU seconds**.
The direct GP lookup checks, endpoint Poisson rate reweighting, weighted
count and all eleven analytic base-normalizer checks passed. The frozen
balance rule selected rate **0.35**. Exact P(A), P(B), and other are
**0.198423, 0.135446, 0.666130**; log evidence is **37.488233**.

At beta 0.5, exact P(B) is **0.001230**, rising to **0.049947** at beta
0.75 and **0.135446** at beta one. Both masses exceed the predeclared
0.10 minimum, and the beta-0.5 mass is less than half its final mass.
The bridge is therefore valid under the frozen rule. The full eleven-rate
grid, tempered curve, 99-cell table and IID reference thresholds are saved
in the checkpoint, committed **before** the sampler phase.

## Absolute sampler comparison

All twelve unchanged sampler runs completed within their frozen CPU caps.
The table reports six independent seeds per size. `Exceed` counts errors
strictly above the exact posterior's 1,000-draw IID q95. The ranges are
absolute errors; these IID thresholds are descriptive for SMC particles.

| Particles | Metric | IID q95 | Sampler median | Absolute range | Exceed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Maximum mode error | 0.04308 | 0.02384 | 0.01690–0.04962 | 1/6 |
| 512 | Full joint TV | 0.10735 | 0.09316 | 0.07223–0.13726 | 2/6 |
| 512 | Maximum cell error | 0.03521 | 0.03009 | 0.01557–0.04266 | 2/6 |
| 512 | Alive-total TV | 0.08490 | 0.07560 | 0.04071–0.11340 | 2/6 |
| 512 | Absolute clutter-mean error | 0.03608 | 0.00977 | 0.00689–0.05767 | 1/6 |
| 2,048 | Maximum mode error | 0.02088 | 0.01294 | 0.00330–0.03651 | 1/6 |
| 2,048 | Full joint TV | 0.05330 | 0.04689 | 0.03534–0.05538 | 1/6 |
| 2,048 | Maximum cell error | 0.01679 | 0.01340 | 0.00665–0.01606 | 0/6 |
| 2,048 | Alive-total TV | 0.04270 | 0.03276 | 0.02135–0.03727 | 0/6 |
| 2,048 | Absolute clutter-mean error | 0.01763 | 0.00982 | 0.00347–0.02094 | 1/6 |

The largest absolute six-seed mean cell residual is **0.00581** at 512
versus familywise q95 **0.01492**, and **0.00405** at 2,048 versus
**0.00729**. Neither particle count has a flagged cell. At 2,048 the
predeclared `mode_error_tail`, `joint_tail`, and `cell_shift` flags are all
**false**; status is `eight_frame_absolute_audit_no_flag`. Individual
2,048 mode errors range 0.00330–0.03651 and joint TVs 0.03534–0.05538.
Mean sampler CPU seconds were 31.5 and 121.2. All six 2,048 estimated
log evidences are below the exact log evidence by 0.048–0.255, an
additional descriptive finding outside the frozen flags.

The validated reference establishes absolute mode weights in this
four-slot/eight-frame, seven-detection case. The unchanged sampler has
no replicated absolute joint-cell signal under the frozen rule; the
512-particle joint and cell tails and one high-particle mode outlier are
retained rather than discarded. The case reuses a designed WP11n
observation, so it is neither an independent population replication nor
a surrogate for WP11k's 19 detections. No new sampler move was selected,
and no 8,192-particle run or untouched holdout was opened. A denser
eight-frame reference with independently validated absolute weights is
needed before attributing WP11k's four-slot instability to a particular
transition mechanism.
