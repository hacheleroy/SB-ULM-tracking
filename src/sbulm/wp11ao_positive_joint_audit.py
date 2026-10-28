"""Validate the positive count/clutter joint only on exposed observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .slot_existence import ABSENT, enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11an_positive_crowded_cohort import _checked_case, _ledger, _ordered
from .wp11ao_positive_joint import positive_four_slot_joint
from .wp11v_tempering_mode_audit import _ExactComputer, _toy_model
from .wp11w_balanced_exact_reference import _limit

AUDIT_CPU_CAP = 800
SOURCE_HASHES = {
    "results/wp11an_geometry_ledger.json":
        "81e7be1d80d8f9c905bfc7b568696b05c9e1315ada8717a8b1329dd371142d03",
    "results/wp11an_references/45.json":
        "723a391f3e909c0c6decd00be3e5a1773549cc5fbf7fc2eca180bd30dc843726",
    "results/wp11an_references/126.json":
        "c1e47e0b1d6cde09c922f15df2a9743e24fa543caaecb2e0fac4951ab47e3505",
    "results/wp11an_references/84.json":
        "a1a804959ade026d34f0a15175bb612b3b8da5aeef05680ef446df89e358b3db",
    "results/wp11ao_exposed_paired84.json":
        "e010b3e05a08be8230df1805fa8c8fccad25a2534357f7ef4c9a8141f0918b6d",
}
TOY_LOGZ = 1e-10
TOY_TV = 1e-10
TOY_CELL = 1e-11
EXPOSED_LOGZ = 1e-8
EXPOSED_TV = 1e-7
EXPOSED_CELL = 1e-8
PRECISION_LOGZ = 1e-10
PRECISION_TV = 1e-10
PRECISION_CELL = 1e-11


def _comparison(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    difference = np.asarray(first["joint"]) - np.asarray(second["joint"])
    return {"log_partition_absolute_difference": abs(first["log_partition"]
                                                      - second["log_partition"]),
            "joint_tv": float(np.abs(difference).sum() / 2),
            "maximum_cell_absolute_difference": float(np.abs(difference).max())}


def _toy() -> dict[str, Any]:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    base = (model.log_existence_prior(support)
            + model.log_count_density(support, include_poisson_void=True))
    gp = model.log_marginal_density(support)
    alive = np.count_nonzero(support != ABSENT, axis=(1, 2))
    first = 1 - np.count_nonzero(support[:, 0] >= 0, axis=1)
    last = 1 - np.count_nonzero(support[:, 2] >= 0, axis=1)
    cells = 4 * alive + 2 * first + last
    computer = _ExactComputer(model, (0, 2))
    checks = {}
    for beta in (0.0, .5, 1.0):
        scores = base + beta * gp
        expected_logz = float(logsumexp(scores))
        expected_joint = np.bincount(
            cells, weights=np.exp(scores - expected_logz), minlength=13 * 2 * 2
        ).reshape(13, 2, 2)
        actual = positive_four_slot_joint(computer, beta)
        difference = np.asarray(actual["joint"]) - expected_joint
        errors = {"log_partition_absolute_difference": abs(
                      actual["log_partition"] - expected_logz),
                  "joint_tv": float(np.abs(difference).sum() / 2),
                  "maximum_cell_absolute_difference": float(np.abs(difference).max())}
        if (errors["log_partition_absolute_difference"] > TOY_LOGZ
                or errors["joint_tv"] > TOY_TV
                or errors["maximum_cell_absolute_difference"] > TOY_CELL):
            raise ValueError(f"positive joint toy enumeration failed at {beta}")
        checks[str(beta)] = errors
    return checks


def audit(root: Path, output: Path) -> dict[str, Any]:
    _limit(AUDIT_CPU_CAP)
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"exposed input changed: {name}")
    toy = _toy()
    ledger = _ledger(root / "results/wp11an_geometry_ledger.json")
    rows = {}
    for index, betas in ((45, (1.0,)), (126, (1.0,)),
                         (84, (0.0, .5, 1.0))):
        stratum = "18" if index == 45 else "19"
        entry = next(item for item in _ordered(ledger, stratum)
                     if item["index"] == index)
        computer = _ExactComputer(_checked_case(entry).model, (5, 7))
        reference = json.loads((root / f"results/wp11an_references/{index}.json")
                               .read_text())
        checks = {}
        for beta in betas:
            paired = (json.loads((root / "results/wp11ao_exposed_paired84.json")
                                 .read_text()) if index == 84 and beta == 1.0
                      else reference["checks"][str(beta)]["paired"])
            double = positive_four_slot_joint(computer, beta, extended=False)
            extended = positive_four_slot_joint(computer, beta)
            comparison = _comparison(extended, paired)
            precision = _comparison(double, extended)
            if (comparison["log_partition_absolute_difference"] > EXPOSED_LOGZ
                    or comparison["joint_tv"] > EXPOSED_TV
                    or comparison["maximum_cell_absolute_difference"] > EXPOSED_CELL
                    or precision["log_partition_absolute_difference"] > PRECISION_LOGZ
                    or precision["joint_tv"] > PRECISION_TV
                    or precision["maximum_cell_absolute_difference"] > PRECISION_CELL):
                raise ValueError(f"positive joint exposed check failed: {index}/{beta}")
            checks[str(beta)] = {"paired_comparison": comparison,
                                 "double_extended_comparison": precision,
                                 "double_cpu_seconds": double["cpu_seconds"],
                                 "extended_cpu_seconds": extended["cpu_seconds"],
                                 "positive_log_partition": extended["log_partition"],
                                 "minimum_positive_cell": extended["minimum_cell"]}
        rows[str(index)] = {"detection_sha256": entry["detection_sha256"],
                            "checks": checks}
    result = {"experiment": "wp11ao_positive_full_joint_exposed_audit",
              "sources_sha256": SOURCE_HASHES,
              "thresholds": {"toy_logz": TOY_LOGZ, "toy_tv": TOY_TV,
                             "toy_max_cell": TOY_CELL,
                             "exposed_logz": EXPOSED_LOGZ,
                             "exposed_tv": EXPOSED_TV,
                             "exposed_max_cell": EXPOSED_CELL,
                             "precision_logz": PRECISION_LOGZ,
                             "precision_tv": PRECISION_TV,
                             "precision_max_cell": PRECISION_CELL},
              "toy": toy, "exposed": rows,
              "status": "toy_and_exposed_full_joint_validated_no_fresh_posterior"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ao_positive_joint_audit.json"))
    args = parser.parse_args()
    print(audit(args.root, args.output)["status"])


if __name__ == "__main__":
    main()
