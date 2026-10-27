# WP11af — Geometry-selected independent absolute transfer check

WP11ac and WP11ad stopped on their numerical rules; WP11ae exhausted
nine numerically sound but unbalanced cases. WP11af is a separate,
fresh-seed **transfer check**. It selects an observation by geometry
alone, before any posterior calculation. It does not claim a new
independently balanced count-mode case. The exposed 19-detection
primary remains mandatory in the later two-case sampler gate.

The fixed twelve ordered trajectory/corruption pairs, generator and
eligibility are in `src/sbulm/wp11af_geometry_transfer_reference.py`.
The generator uses detection probability 0.60, clutter rate 0.35,
separation 0.007 mm, at most two injected clutter points per frame,
and the four-slot staggered truth mask. Select the **first** generated
observation with 16–17 detections, at least two in frame six, at least
one in the last frame and at least seven across frames three through
five. Generator errors are ineligible. There is no posterior-balance
search and no replacement following a numerical failure. The pinned
geometry ledger is `results/wp11af_geometry_transfer_ledger.json`
(SHA-256 `14350ebaff594949d18c40cad0e4c02b33dd0c8b926134767209dbdcf12f5f98`).
It selects pair index 3, seeds 2243/2251, with 17 detections and
cardinalities (0, 2, 3, 3, 1, 3, 2, 3). This selection and ledger are
committed before its posterior is examined.

Compute the exact 33-by-(frame-six clutter count)-by-(last clutter
count) joint with the existing four-slot squarefree subset method.
Validate independent direct and paired parenthesizations at beta
0, 0.5 and 1.0. Both must have minimum raw cell >=-1e-10, imaginary
residual <=1e-8 and raw mass error <=1e-8; their absolute log-partition
difference <=1e-8, joint TV <=1e-7 and maximum cell difference <=1e-8.
Beta zero must match the analytic base partition within 1e-8. The
directly enumerated toy must pass its existing tighter tests. Use one
thread and a 1,000 process-CPU-second validation cap. Stop before
sampler inference if any condition fails.

If validated, freeze its SHA-256 and a paired control/candidate rule
before running inference. It can reveal lack of transfer across a new
observation but cannot establish calibration of the crowded count-band
geometry. No untouched holdout is opened.

## Outcome

The selected observation **passed** the toy, beta-zero analytic, direct
and paired numerical checks at beta 0, 0.5 and 1.0. Its reference
checkpoint is `results/wp11af_geometry_transfer_exact.json` (SHA-256
`cd6f20f821a2689cda5a1a0aceb19fa9db7c6232af0e8e05b6262e312469f87b`).
Validation took 232.5 process CPU seconds, below the 1,000-second cap.
At beta one, direct/paired joint TV was 3.27e-11, log-partition
difference 4.20e-11 and minimum raw cell -4.65e-12.

The exact low/middle/high alive bands are 0.65436/0.11833/0.22731;
frame-six clutter-one mass is 0.93352. This is a numerically validated
independent absolute transfer reference, **not** a balanced crowded
count/clutter-mode reference. WP11ag therefore retains the exposed
19-detection primary as a mandatory separate sentinel.
