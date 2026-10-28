# WP11an — Prospective positive-partition crowded cohort

The WP11am positive partition method was validated on toy enumeration and
the **already exposed** stopped WP11aj/WP11ak cases in commit
`88a1437a2020a29f412302069ada6840a99cbc8f`. Neither stopped cohort is
reinterpreted. This protocol and its geometry-only ledger are pinned
**before any WP11an posterior or sampler inference**. The WP8 holdout stays
closed.

`results/wp11an_geometry_ledger.json` has SHA-256
`81e7be1d80d8f9c905bfc7b568696b05c9e1315ada8717a8b1329dd371142d03`.
The new 480 seed pairs, 18/19-detection eligibility, two decoy placements,
three offsets per placement, geometry ordering, distinct-base quotas (one
18 and two 19), and finite screening budgets are specified in
`src/sbulm/wp11an_positive_crowded_cohort.py`. Geometry alone yielded
seven eligible 18-detection and ten eligible 19-detection bases, with
63 and 90 eligible variants, respectively. No likelihood was screened
when counting these.

Screens proceed in the exact frozen order, with a limit of 24 (18) and
54 (19). At beta one, the paired-square joint must have minimum raw cell
at least `-1e-10`, clipping TV at most `1e-9`, imaginary residual and
raw mass error each at most `1e-8`. Its log partition must agree with
the positive extended-precision four-slot calculation within `1e-8`;
positive float64 and extended calculations must agree within `1e-10`.
Candidates additionally need low/high alive bands at least `.18`, middle
at least `.05`, and two clutter categories at least `.15` in each of
frames six and eight. First qualifying variants from distinct bases
fill the quotas. Numerical failure or CPU cap stops the whole cohort;
exhaustion leaves it unselected. No partial cohort enters sampler tests.

After all three candidates are selected, validation at beta 0, .5, 1
repeats paired/positive checks, toy direct enumeration, and compares
normalized direct-fourth versus paired joints: TV at most `1e-7`, maximum
cell difference at most `1e-8`. The direct joint additionally needs raw
minimum at least `-1e-8`, clipping TV at most `1e-7`, imaginary residual
and mass error at most `1e-8`. Direct-fourth log partition is recorded
as a diagnostic, not used as a gate. At beta zero, paired log partition
must agree with the analytic base within `1e-8`. A validation failure
stops the cohort without replacements. Screen and validation CPU caps
are 900 and 3000 seconds per process. The implementation pins every
observation and reference hash into a manifest only after complete
validation.

Only a complete validated manifest can activate the separately frozen
WP11al 24-job matched sampler comparison. These exact numerical checks
share the one-slot factors, so they test the four-slot convolution and
normalized joint, without a formal floating-point interval certificate
or an independent derivation of the likelihood.

## Frozen outcome

The 34 screens selected index 45 (18 detections) and indices 126 and 84
(19 detections on distinct bases). All 34 passed the paired/positive
partition screen, with maximum absolute log-partition difference
`1.82e-11` and maximum float64/extended positive difference `2.85e-14`.
The first 18- and 19-detection references passed all three validation
temperatures. Index 84 passed beta zero and .5 but **failed the frozen
beta-one numerical validation**. The direct-fourth joint's raw minimum
was `-1.476e-7` (limit `-1e-8`), clipping TV was `8.429e-7` (limit
`1e-7`), direct/paired joint TV was `6.073e-7` (limit `1e-7`), and its
maximum cell difference was `1.051e-7` (limit `1e-8`). Direct/paired
log-partition difference, reported diagnostically, was `9.987e-7`.
The paired/positive log-partition difference remained `9.38e-12` and
float64/extended positive difference `1.42e-14` on that case.

The terminal diagnostic is
`results/wp11an_terminal_validation_failure.json` (SHA-256
`dd1555a41caa49a9f6309300cd6170cc2ac6ae82d521616abeb9077caa24fd7f`).
It can be recomputed with
`python -m sbulm.wp11an_terminal_diagnostic`; this post hoc program
only records why the already frozen validation stopped.
The index 84 reference file contains only the first two successful
temperatures and is marked `reference_in_progress`; it is **not** a
validated reference. The cohort is stopped. No three-case manifest or
WP11al sampler comparison is authorized by this result, and the WP8
holdout remains closed. The numerical method improved the absolute
partition check, but direct-fourth is not a reliable independent full
joint check on this crowded case.
