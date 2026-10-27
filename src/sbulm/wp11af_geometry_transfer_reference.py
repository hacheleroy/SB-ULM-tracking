"""Geometry-selected absolute transfer reference with no posterior-based search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .wp7_convergence_audit import _save
from .wp8_generalization import STAGGERED_TRUTH_MASK, GeneralizationRegime, _metadata, make_case
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11v_tempering_mode_audit import (
    NUMERIC_CELL_TOL,
    NUMERIC_LOGZ_TOL,
    NUMERIC_TV_TOL,
    _ExactComputer,
    _numeric_ok,
    _toy_validations,
)

PAIRS = ((2203, 2207), (2213, 2221), (2237, 2239), (2243, 2251),
         (2267, 2269), (2273, 2281), (2287, 2293), (2297, 2309),
         (2311, 2333), (2339, 2341), (2347, 2351), (2357, 2371))
LEDGER_SHA256 = "14350ebaff594949d18c40cad0e4c02b33dd0c8b926134767209dbdcf12f5f98"
EXACT_CPU_CAP = 1_000


def _regime(index: int) -> GeneralizationRegime:
    trajectory, corruption = PAIRS[index]
    return GeneralizationRegime(
        name=f"wp11af_transfer_{index}", detection_probability=.60,
        expected_clutter_per_frame=.35, central_separation=.007,
        trajectory_seed=trajectory, corruption_seed=corruption,
        max_clutter_per_frame=2, truth_mask=STAGGERED_TRUTH_MASK,
    )


def _protocol() -> dict[str, Any]:
    return {
        "pairs": [list(pair) for pair in PAIRS],
        "regime": {"detection_probability": .60, "expected_clutter_per_frame": .35,
                   "central_separation": .007, "max_clutter_per_frame": 2,
                   "truth_mask": [list(mask) for mask in STAGGERED_TRUTH_MASK]},
        "selection": "first generated case with 16..17 detections, frame six >=2, "
                     "last >=1, frames 3..5 >=7; generator errors skip; "
                     "no posterior-balance selection or replacement after numerical failure",
        "numeric": {"minimum_raw_cell": -1e-10, "max_imaginary_residual": 1e-8,
                    "raw_mass_error": 1e-8, "direct_paired_logz": NUMERIC_LOGZ_TOL,
                    "direct_paired_joint_tv": NUMERIC_TV_TOL,
                    "direct_paired_max_cell": NUMERIC_CELL_TOL,
                    "base_logz": NUMERIC_LOGZ_TOL},
        "exact_cpu_cap": EXACT_CPU_CAP, "no_sampler_or_holdout": True,
    }


def geometry(output: Path) -> dict[str, Any]:
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["protocol"] != _protocol():
            raise ValueError("geometry protocol changed")
        return saved
    candidates = []
    for index, pair in enumerate(PAIRS):
        try:
            case = make_case(_regime(index))
        except ValueError as exc:
            candidates.append({"index": index, "pair": list(pair),
                               "eligible": False, "generator_error": str(exc)})
            continue
        cards = tuple(case.model.sequence.cardinalities)
        metadata = _metadata(_regime(index), case)["model_and_detections"]
        candidates.append({
            "index": index, "pair": list(pair), "cardinalities": list(cards),
            "detection_sha256": metadata["detections_sha256"],
            "eligible": 16 <= sum(cards) <= 17 and cards[5] >= 2 and cards[7] >= 1
            and sum(cards[2:5]) >= 7,
        })
    selected = next((item["index"] for item in candidates if item["eligible"]), None)
    result = {"experiment": "wp11af_geometry_transfer_ledger", "protocol": _protocol(),
              "candidates": candidates, "selected_index": selected,
              "status": "geometry_only_before_posterior"}
    _save(output, result)
    return result


def _selected_case(ledger_path: Path) -> tuple[Any, dict[str, Any]]:
    if LEDGER_SHA256 is None or hashlib.sha256(ledger_path.read_bytes()).hexdigest() != (
        LEDGER_SHA256
    ):
        raise ValueError("geometry-only ledger hash changed or is unpinned")
    ledger = json.loads(ledger_path.read_text())
    if ledger["protocol"] != _protocol() or ledger["status"] != (
        "geometry_only_before_posterior"
    ):
        raise ValueError("frozen geometry rule differs")
    expected = next((item for item in ledger["candidates"] if item["eligible"]), None)
    if expected is None or ledger["selected_index"] != expected["index"]:
        raise ValueError("no selected observation under the frozen geometry rule")
    case = make_case(_regime(expected["index"]))
    metadata = _metadata(_regime(expected["index"]), case)["model_and_detections"]
    if (metadata["detections_sha256"] != expected["detection_sha256"]
            or list(case.model.sequence.cardinalities) != expected["cardinalities"]):
        raise ValueError("selected observation changed")
    return case, expected


def validate(ledger_path: Path, output: Path) -> dict[str, Any]:
    case, entry = _selected_case(ledger_path)
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved["ledger_sha256"] != LEDGER_SHA256
                or saved["selected_index"] != entry["index"]):
            raise ValueError("existing exact checkpoint differs")
        if saved["status"] == "validated_reference_no_sampler":
            return saved
        if saved["status"] != "reference_in_progress":
            raise ValueError("invalid exact checkpoint status")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < EXACT_CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen exact cap")
    resource.setrlimit(resource.RLIMIT_CPU, (EXACT_CPU_CAP, EXACT_CPU_CAP + 5))
    started = time.process_time()
    toy = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    checks = {}
    for beta in (0.0, .5, 1.0):
        direct, paired = computer.compute(beta), computer.compute(beta, "paired")
        d, p = np.asarray(direct["joint"]), np.asarray(paired["joint"])
        agreement = {
            "log_partition_absolute_difference": abs(
                direct["log_partition"] - paired["log_partition"]
            ),
            "joint_tv": float(np.abs(d - p).sum() / 2),
            "maximum_cell_absolute_difference": float(np.max(np.abs(d - p))),
        }
        checks[str(beta)] = {"direct": direct, "paired_numerical": {
            **agreement, **{key: paired[key] for key in (
                "minimum_raw_cell", "maximum_relative_imaginary_residual",
                "raw_mass_error", "clipping_tv", "cpu_seconds"
            )}}}
        if (not _numeric_ok(direct) or not _numeric_ok(paired)
                or agreement["log_partition_absolute_difference"] > NUMERIC_LOGZ_TOL
                or agreement["joint_tv"] > NUMERIC_TV_TOL
                or agreement["maximum_cell_absolute_difference"] > NUMERIC_CELL_TOL
                or beta == 0 and abs(direct["log_partition"]
                                     - case.model.base_log_normalizer) > NUMERIC_LOGZ_TOL):
            raise ValueError(f"independent exact validation failed at beta={beta}")
        _save(output, {"experiment": "wp11af_geometry_transfer_exact",
                       "ledger_sha256": LEDGER_SHA256, "selected_index": entry["index"],
                       "selected_detection_sha256": entry["detection_sha256"],
                       "toy_validations": toy, "checks": checks,
                       "status": "reference_in_progress"})
    result = {"experiment": "wp11af_geometry_transfer_exact",
              "ledger_sha256": LEDGER_SHA256, "selected_index": entry["index"],
              "selected_detection_sha256": entry["detection_sha256"],
              "toy_validations": toy, "checks": checks,
              "base_log_normalizer": case.model.base_log_normalizer,
              "cpu_seconds": time.process_time() - started,
              "status": "validated_reference_no_sampler"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry", "validate"), required=True)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11af_geometry_transfer_ledger.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11af_geometry_transfer_exact.json"))
    args = parser.parse_args()
    result = geometry(args.ledger) if args.case == "geometry" else validate(
        args.ledger, args.output
    )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
