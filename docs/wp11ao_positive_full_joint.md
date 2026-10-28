# WP11ao — Nonnegative full-joint numerical method

WP11an stopped when the direct-fourth joint failed its predeclared
numerical envelope on index 84 at beta one. This is **method development
on exposed observations**; the stopped cohort is not reopened.

For nonnegative one-slot coefficients \(f_l(S)\), where \(l\) is alive
length and \(S\) is the set of assigned detections, explicitly form
the ordered, disjoint two-slot coefficient

\[
p_a(S)=\sum_{A\cap B=\varnothing,\ A\cup B=S}\sum_{l+m=a}f_l(A)f_m(B).
\]

Split detection bits into frames six and eight, \(H\), and all other
frames, \(O\). For each pair of disjoint target-frame masks, positive
subset zeta sums over the complement of the first pair's \(O\)-mask
sum all disjoint second-pair contributions. Accumulate them at alive
total \(a+b\) and the two clutter counts implied by the union of the
target-frame masks. All terms are nonnegative. This computes the full
alive-total × sixth-clutter × final-clutter joint and log partition
without Möbius subtraction or Fourier inversion. It shares the
one-slot factors with the paired-square implementation, so it checks
the four-slot combinatorics and numerical stability, not an
independent derivation of the observation likelihood.

The audited computation matches full direct enumeration on a toy case
at beta 0, .5, and 1 with joint TV below `1.1e-15`. On the exposed
WP11an index 84 at beta one, its joint TV versus paired-square is
`3.965e-11`, maximum cell difference `4.603e-12`, and log-partition
difference `9.379e-12`. Positive float64 and extended precision differ
by `2.23e-16` in joint TV. The earlier unstable direct-fourth joint
had paired TV `6.073e-7` and raw minimum `-1.476e-7` on that case.
The exposed 18-detection index 45 and 19-detection index 126, plus
index 84 at beta zero and .5, also pass the same audited envelope.
Extended index-84 positive joint cost about 35–41 CPU seconds across
these temperatures, versus 318 CPU seconds for a paired beta-one
recomputation.

The audit is `results/wp11ao_positive_joint_audit.json` (SHA-256
`806d3028eb0dfb1557e1df7e51aa43cc9bf831a70b002cbabe972ff57e56699d`).
The archived paired beta-one joint is
`results/wp11ao_exposed_paired84.json` (SHA-256
`e010b3e05a08be8230df1805fa8c8fccad25a2534357f7ef4c9a8141f0918b6d`).
The audit pins hashes of all exposed inputs and recomputes the positive
joint. CPU timings are environment-specific, and agreement of float64
with extended precision is an empirical check, not a formal interval
bound. A fresh cohort must freeze its observation rule and complete
numerical gate before any new posterior.
