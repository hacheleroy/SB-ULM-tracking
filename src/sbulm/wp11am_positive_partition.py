"""Subtraction-free four-slot partition cross-check for crowded references.

The one-slot factors are shared with the squarefree joint computer, but
the four-slot partition is summed by positive pair convolution followed
by a positive subset zeta transform. This avoids the Mobius inversion
used by the direct-fourth and paired-square joint calculations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .slot_existence import enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11aj_crowded_cohort_reference import _checked_case as _aj_case
from .wp11aj_crowded_cohort_reference import _ledger as _aj_ledger
from .wp11aj_crowded_cohort_reference import _ordered as _aj_ordered
from .wp11ak_crowded_cohort_reference import _checked_case as _ak_case
from .wp11ak_crowded_cohort_reference import _ledger as _ak_ledger
from .wp11ak_crowded_cohort_reference import _ordered as _ak_ordered
from .wp11v_tempering_mode_audit import _ExactComputer, _toy_model
from .wp11w_balanced_exact_reference import _limit

AUDIT_CPU_CAP = 300
TOY_LOGZ_TOL = 1e-9
PRECISION_LOGZ_TOL = 1e-10
EXPOSED_PAIRED_LOGZ_TOL = 1e-8


def positive_four_slot_partition(computer: _ExactComputer, beta: float,
                                 *, extended: bool = True) -> dict[str, Any]:
    """Evaluate ordered four-slot Z with nonnegative additions only.

    Let f[S] be the sum of one-slot factors using detection subset S.
    The ordered two-slot coefficient p[S] is the disjoint convolution
    sum_{A disjoint B, A union B=S} f[A]f[B]. The positive zeta transform
    g[C]=sum_{T subset C}p[T] gives Z4=sum_S p[S]g[full \\ S].
    """
    if computer.model.n_slots != 4 or not 0 <= beta <= 1:
        raise ValueError("requires four slots and beta in [0,1]")
    started = time.process_time()
    dtype = np.longdouble if extended else np.float64
    scores = computer.base + beta * computer.gp
    logz1 = float(logsumexp(scores))
    n = computer.n
    f = np.zeros(1 << n, dtype=dtype)
    np.add.at(f, computer.subsets, np.exp(scores.astype(dtype) - dtype(logz1)))
    masks = np.flatnonzero(f).astype(np.int32)
    weights = f[masks]
    pair = np.zeros(1 << n, dtype=dtype)
    for index, mask in enumerate(masks):
        disjoint = (masks & mask) == 0
        np.add.at(pair, masks[disjoint] | mask,
                  weights[index] * weights[disjoint])
    zeta = pair.copy()
    for bit in range(n):
        view = zeta.reshape(-1, 2, 1 << bit)
        view[:, 1, :] += view[:, 0, :]
    if np.any(pair < 0) or np.any(zeta < 0):
        raise ValueError("positive convolution produced a negative coefficient")
    complement = (1 << n) - 1 - np.arange(1 << n)
    mass = np.dot(pair, zeta[complement])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("nonpositive or nonfinite four-slot partition")
    model = computer.model
    logz = (dtype(n) * np.log(dtype(model.clutter_intensity))
            - dtype(model.sequence.n_frames)
            * dtype(model.expected_clutter_per_frame)
            + 4 * dtype(logz1) + np.log(mass))
    return {"beta": float(beta), "log_partition": float(logz),
            "normalized_four_slot_mass": float(mass),
            "n_nonzero_one_slot_subsets": len(masks),
            "precision": "longdouble" if extended else "float64",
            "cpu_seconds": time.process_time() - started}


def _toy_audit() -> dict[str, Any]:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    prior = model.log_existence_prior(support)
    count = model.log_count_density(support, include_poisson_void=True)
    gp = model.log_marginal_density(support)
    computer = _ExactComputer(model, (0, 2))
    checks = {}
    for beta in (0.0, .5, 1.0):
        expected = float(logsumexp(prior + count + beta * gp))
        double = positive_four_slot_partition(computer, beta, extended=False)
        extended = positive_four_slot_partition(computer, beta)
        paired = computer.compute(beta, "paired")
        if (abs(extended["log_partition"] - expected) > TOY_LOGZ_TOL
                or abs(double["log_partition"] - extended["log_partition"])
                > PRECISION_LOGZ_TOL
                or abs(extended["log_partition"] - paired["log_partition"])
                > TOY_LOGZ_TOL):
            raise ValueError(f"toy positive partition validation failed at {beta}")
        checks[str(beta)] = {"direct_enumeration_log_partition": expected,
                             "double": double, "extended": extended,
                             "paired_log_partition": paired["log_partition"]}
    return checks


def exposed_audit(aj_ledger: Path, ak_ledger: Path,
                  aj_diagnostic: Path, ak_screen: Path,
                  output: Path) -> dict[str, Any]:
    _limit(AUDIT_CPU_CAP)
    toy = _toy_audit()
    aj = _aj_ordered(_aj_ledger(aj_ledger), "18")[0]
    ak = next(entry for entry in _ak_ordered(_ak_ledger(ak_ledger), "19")
              if entry["index"] == 63)
    rows = {}
    archived = {
        "wp11aj_stopped_18": json.loads(aj_diagnostic.read_text()),
        "wp11ak_stopped_19": json.loads(ak_screen.read_text()),
    }
    for name, case in (("wp11aj_stopped_18", _aj_case(aj)),
                       ("wp11ak_stopped_19", _ak_case(ak))):
        computer = _ExactComputer(case.model, (5, 7))
        double = positive_four_slot_partition(computer, 1.0, extended=False)
        extended = positive_four_slot_partition(computer, 1.0)
        record = archived[name]
        numerical = record["numerical"] if name.endswith("19") else record
        paired = numerical["paired"]["log_partition"]
        direct = numerical["direct"]["log_partition"]
        if (abs(double["log_partition"] - extended["log_partition"])
                > PRECISION_LOGZ_TOL
                or abs(extended["log_partition"] - paired)
                > EXPOSED_PAIRED_LOGZ_TOL):
            raise ValueError(f"positive partition disagrees with {name}")
        rows[name] = {"double": double, "extended": extended,
                      "archived_paired_log_partition": paired,
                      "archived_direct_log_partition": direct,
                      "positive_minus_paired": extended["log_partition"] - paired,
                      "positive_minus_direct": extended["log_partition"] - direct}
    result = {"experiment": "wp11am_positive_partition_exposed_audit",
              "toy": toy, "exposed": rows,
              "status": "toy_and_exposed_numerical_method_validated_no_new_posterior"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aj-ledger", type=Path,
                        default=Path("results/wp11aj_geometry_ledger.json"))
    parser.add_argument("--ak-ledger", type=Path,
                        default=Path("results/wp11ak_geometry_ledger.json"))
    parser.add_argument("--aj-diagnostic", type=Path,
                        default=Path("results/wp11aj_numeric_diagnostic.json"))
    parser.add_argument("--ak-screen", type=Path,
                        default=Path("results/wp11ak_screens/63.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11am_positive_partition_audit.json"))
    args = parser.parse_args()
    print(exposed_audit(args.aj_ledger, args.ak_ledger, args.aj_diagnostic,
                        args.ak_screen, args.output)["status"])


if __name__ == "__main__":
    main()
