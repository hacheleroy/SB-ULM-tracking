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
