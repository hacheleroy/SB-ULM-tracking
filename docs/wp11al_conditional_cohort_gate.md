# WP11al — Conditional replication rule for a future crowded cohort

WP11aj and WP11ak stopped on their frozen numerical rules. Neither
produced a complete validated cohort, and neither permits inference
on its partial selection. The following **conditional sampler rule**
is fixed before sampler inference on any future crowded cohort. It
cannot be invoked until a new prospective observation protocol has
independently selected and numerically validated one 18-detection and
two distinct 19-detection absolute joints, with their geometry,
screen, and reference SHA-256 values bound in an executable runner.
The WP11ah 17-detection result and exposed 19-detection sentinel remain
separate context, not members of the new replication count.

For each of the three independently selected cases, compare the
unchanged WP11g control with the unchanged WP11ag whole-slot candidate.
Use 2,048 particles, adaptive tempering, eight plus eight sweeps, and
the candidate's 20% exact full-slot conditional proposal fraction at
beta >=0.65. Assign the 18-detection case matched seeds 1800–1803,
the first 19-detection case 1810–1813, and the second 19-detection case
1820–1823. Run **all 24** control/candidate jobs, including after an
early poor accuracy result. Set all four BLAS/OMP thread variables to
one. Each job has a 750 process CPU-second cap; any cap, hash,
configuration or numerical mismatch stops comparison. Save full 3D
joint cells, stage summaries, evidence errors, CPU and per-seed metrics.

Apply the previously frozen WP11aa **four-run decoy gate independently
to each case**, unchanged: at least 3/4 paired full-joint TV wins;
candidate median full-joint TV <=0.12 and <=0.90 of the matched-control
median; candidate median band-by-sixth TV <=0.10, maximum alive-band
error <=0.08, maximum joint-cell error <=0.04; no more than +0.020
median regression on each of alive-total TV, maximum joint-cell error,
sixth clutter mean absolute error and last clutter mean absolute error.
At least 3/4 candidate runs in **each case** must simultaneously have
band-by-sixth TV <=0.10, maximum alive-band mass error <=0.08 and
absolute clutter-one mass error <=0.06 at both frame six and the final
frame. Every candidate/control CPU ratio must be <=1.50 and each
case's candidate/control median process CPU ratio <=1.25.

The cohort passes only if **all three** cases pass every per-case gate
and, across the 12 matched seed pairs, there are at least 10/12
full-joint TV wins **and** 10/12 runs meeting all simultaneous mode
limits. Do not pool medians across cases with different exact joints.
Do not amend seeds, thresholds or the number of cases after examining
sampler output. If no independently validated complete cohort exists,
the rule remains dormant and the untouched WP8 holdout stays closed.
