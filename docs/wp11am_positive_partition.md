# WP11am — Positive four-slot partition cross-check

The WP11ak 19-detection screen stopped because direct-fourth and
paired-square squarefree calculations differed in log partition by
`4.54e-8` (frozen limit `1e-8`). The normalized joints still differed
by only `7.66e-8`. This numerical audit is on **already exposed**
WP11aj/WP11ak observations and does not reopen either failed cohort.

For one-slot nonnegative coefficients \(f(S)\), form the ordered
two-slot disjoint convolution

\[
p(S)=\sum_{A\cap B=\varnothing,\ A\cup B=S}f(A)f(B).
\]

Use the positive subset zeta sum \(g(C)=\sum_{T\subseteq C}p(T)\).
The four-slot partition is

\[
Z_4=\sum_S p(S)g(U\setminus S).
\]

All additions and products are nonnegative. This path shares the
one-slot factors with the joint computer but avoids its Möbius
subtraction and alive-count Fourier inversion. It provides an
independent convolution check of the **absolute partition**, while
paired-square remains the source of the full alive × sixth-clutter ×
last-clutter joint. The implementation computes the positive sum in
both float64 and extended precision. It does not supply formal
rounding intervals or independently verify the one-slot likelihood.

The `wp11am_positive_partition_audit.json` checkpoint (SHA-256
`fb464d3d9ec56d892c294b1944696805c67c73244ef155975e4967f12d75d54a`)
evaluates beta
0, 0.5 and 1 against full direct enumeration on a toy model. Worst
absolute log-partition difference is below `2e-15`. On the two exposed
stopped cases, positive extended precision minus archived paired-square
log partition is `2.26e-11` (18 detections) and `-8.67e-13`
(19 detections); the positive result differs from the archived direct
19-detection result by `-4.54e-8`. Double versus extended precision
agrees to displayed float precision in both exposed cases. The tiny
normalized four-slot masses (`1.43e-23` and `1.46e-16`) explain why
subtractive convolution can be sensitive to cancellation.

A future **fresh** observation protocol can prospectively require
paired-square versus positive log partition agreement <=`1e-8` and
float64 versus extended positive agreement <=`1e-10`, while using the
paired-square joint's existing raw-cell/clipping envelope. Direct
fourth-power log partition should be reported as a diagnostic rather
than treated as a reference on a dense case. If a full joint is
selected, independently compare its normalized direct-fourth joint
at beta 0, 0.5 and 1 under a separately frozen error budget. These
criteria must be committed before any posterior on new observations.
