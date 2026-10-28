"""Post hoc numerical diagnosis of WP11an's stopped validation case.

This repeats the selected index-84 beta-one calculation and records the
measurements behind the already failed frozen validation. It never
changes the selection rule or produces a validated reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .wp7_convergence_audit import _save
from .wp11an_positive_crowded_cohort import (
    LEDGER_SHA256,
    _checked_case,
    _joint_checked,
    _ledger,
    _ordered,
    _partition_checked,
)
from .wp11v_tempering_mode_audit import _ExactComputer
from .wp11w_balanced_exact_reference import _limit


def diagnose(ledger_path: Path, output: Path) -> dict:
    """Repeat beta one for the stopped selected observation, for diagnosis."""
    _limit(1_200)
    ledger = _ledger(ledger_path)
    entry = next(x for x in _ordered(ledger, "19") if x["index"] == 84)
    computer = _ExactComputer(_checked_case(entry).model, (5, 7))
    paired = computer.compute(1.0, "paired")
    partition_ok, partition = _partition_checked(computer, 1.0, paired)
    direct = computer.compute(1.0)
    joint_ok, joint = _joint_checked(paired, direct)
    keys = ("minimum_raw_cell", "clipping_tv", "maximum_relative_imaginary_residual",
            "raw_mass_error", "log_partition", "cpu_seconds")
    result = {"experiment": "wp11an_terminal_validation_failure_diagnostic",
              "ledger_sha256": LEDGER_SHA256,
              "index": 84, "beta": 1.0,
              "selected_detection_sha256": entry["detection_sha256"],
              "paired": {key: paired[key] for key in keys},
              "positive": partition, "partition_pass": partition_ok,
              "direct": {key: direct[key] for key in keys}, "joint": joint,
              "joint_pass": joint_ok,
              "status": "validation_failed_stop_no_manifest_no_sampler"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11an_geometry_ledger.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11an_terminal_validation_failure.json"))
    args = parser.parse_args()
    result = diagnose(args.ledger, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
