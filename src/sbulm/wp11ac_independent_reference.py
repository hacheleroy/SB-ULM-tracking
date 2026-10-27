"""Independent crowded eight-frame absolute-reference search before inference."""

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

PAIRS = ((601, 607), (613, 617), (619, 631), (641, 643),
         (647, 653), (659, 661), (673, 677), (683, 691),
         (701, 709), (719, 727), (733, 739), (743, 751),
         (757, 761), (769, 773), (787, 797), (809, 811))
LEDGER_SHA256 = "8a9dd92829f62b6094913682950f8eb39eaaf7dfe48365e46890a0794d9f0623"
MAX_SCREENS = 5
SCREEN_CPU_CAP = 450
VALIDATION_CPU_CAP = 1_800
LOW_HIGH_MIN = .18
MIDDLE_MIN = .05
CLUTTER_TWO_MIN = .15


def _regime(index: int) -> GeneralizationRegime:
    a, b = PAIRS[index]
    return GeneralizationRegime(
        name=f"wp11ac_independent_{index}", detection_probability=.68,
        expected_clutter_per_frame=.28, central_separation=.007,
        trajectory_seed=a, corruption_seed=b, max_clutter_per_frame=2,
        truth_mask=STAGGERED_TRUTH_MASK,
    )


def _ledger_protocol() -> dict[str, Any]:
    return {
        "pairs": [list(pair) for pair in PAIRS],
        "regime": {"detection_probability": .68, "expected_clutter_per_frame": .28,
                   "central_separation": .007, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "eligibility": "18..19 detections; frame six >=2, last >=1; frames 3..5 >=8",
        "selection": "screen the first five eligible observations in seed order; accept first balanced",
        "generator_error": "record as ineligible, never replace seed",
        "acceptance": {"low_min": LOW_HIGH_MIN, "high_min": LOW_HIGH_MIN,
                       "middle_min": MIDDLE_MIN, "two_sixth_min": CLUTTER_TWO_MIN,
                       "two_last_min": CLUTTER_TWO_MIN},
        "numerical_envelope": {"minimum_raw_cell": -1e-10,
                               "maximum_relative_imaginary_residual": 1e-8,
                               "raw_mass_error": 1e-8,
                               "direct_paired_logz": NUMERIC_LOGZ_TOL,
                               "direct_paired_tv": NUMERIC_TV_TOL,
                               "direct_paired_max_cell": NUMERIC_CELL_TOL},
        "maximum_screens": MAX_SCREENS,
        "screen_cpu_cap": SCREEN_CPU_CAP, "validation_cpu_cap": VALIDATION_CPU_CAP,
        "no_sampler_or_holdout": True,
    }


def geometry(output: Path) -> dict[str, Any]:
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["protocol"] != _ledger_protocol():
            raise ValueError("geometry protocol differs from frozen rule")
        return saved
    candidates = []
    for index in range(len(PAIRS)):
        regime = _regime(index)
        try:
            case = make_case(regime)
        except ValueError as exc:
            candidates.append({"index": index, "pair": list(PAIRS[index]),
                               "eligible": False, "generator_error": str(exc)})
            continue
        cards = tuple(case.model.sequence.cardinalities)
        metadata = json.loads(json.dumps(_metadata(regime, case)))
        candidates.append({
            "index": index, "pair": list(PAIRS[index]),
            "cardinalities": list(cards),
            "detection_sha256": metadata["model_and_detections"]["detections_sha256"],
            "eligible": 18 <= sum(cards) <= 19 and cards[5] >= 2 and cards[7] >= 1
            and sum(cards[2:5]) >= 8,
        })
    result = {"experiment": "wp11ac_geometry_ledger", "protocol": _ledger_protocol(),
              "candidates": candidates, "status": "geometry_only_before_posterior"}
    _save(output, result)
    return result


def _validated_ledger(path: Path) -> dict[str, Any]:
    if LEDGER_SHA256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != LEDGER_SHA256:
        raise ValueError("geometry ledger hash changed or not pinned")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["protocol"] != _ledger_protocol() or saved["status"] != (
        "geometry_only_before_posterior"
    ):
        raise ValueError("invalid frozen geometry ledger")
    return saved


def _checked_case(entry: dict[str, Any]) -> Any:
    case = make_case(_regime(entry["index"]))
    metadata = _metadata(_regime(entry["index"]), case)["model_and_detections"]
    if (tuple(case.model.sequence.cardinalities) != tuple(entry["cardinalities"])
            or metadata["detections_sha256"] != entry["detection_sha256"]):
        raise ValueError("candidate observation changed from frozen geometry ledger")
    return case


def _limit(cpu: int) -> None:
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set every BLAS/OMP thread variable to one")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < cpu + 5:
        raise RuntimeError("system CPU hard limit below frozen cap")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))


def _masses(joint: np.ndarray) -> dict[str, Any]:
    alive = joint.sum(axis=(1, 2))
    sixth = joint.sum(axis=(0, 2))
    last = joint.sum(axis=(0, 1))
    bands = [float(alive[:21].sum()), float(alive[21]), float(alive[22:].sum())]
    return {"alive_bands": bands, "sixth_clutter": sixth.tolist(),
            "last_clutter": last.tolist()}


def _accepted(masses: dict[str, Any]) -> bool:
    low, mid, high = masses["alive_bands"]
    return (low >= LOW_HIGH_MIN and high >= LOW_HIGH_MIN and mid >= MIDDLE_MIN
            and sum(v >= CLUTTER_TWO_MIN for v in masses["sixth_clutter"]) >= 2
            and sum(v >= CLUTTER_TWO_MIN for v in masses["last_clutter"]) >= 2)


def _eligible(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in ledger["candidates"] if entry["eligible"]][:MAX_SCREENS]


def screen(ledger_path: Path, run_dir: Path, index: int) -> dict[str, Any]:
    ledger = _validated_ledger(ledger_path)
    entries = _eligible(ledger)
    if index not in [entry["index"] for entry in entries]:
        raise ValueError("index is not cardinality eligible")
    previous = [entry for entry in entries if entry["index"] < index]
    for entry in previous:
        path = run_dir / f"{entry['index']}.json"
        if not path.exists():
            raise ValueError("earlier eligible candidate has not been screened")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["status"] != "balance_failed_continue":
            raise ValueError("earlier candidate accepted or numerical check failed: stop")
    output = run_dir / f"{index}.json"
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["ledger_sha256"] != LEDGER_SHA256 or saved["index"] != index:
            raise ValueError("candidate checkpoint differs from frozen ledger")
        return saved
    _limit(SCREEN_CPU_CAP)
    entry = entries[next(i for i, item in enumerate(entries) if item["index"] == index)]
    case = _checked_case(entry)
    started = time.process_time()
    direct = _ExactComputer(case.model, (5, 7)).compute(1.0)
    masses = _masses(np.asarray(direct["joint"]))
    numeric = {key: direct[key] for key in (
        "minimum_raw_cell", "maximum_relative_imaginary_residual",
        "raw_mass_error", "clipping_tv", "log_partition"
    )}
    status = ("numerical_failed_stop" if not _numeric_ok(direct)
              else "balance_accepted_validate" if _accepted(masses)
              else "balance_failed_continue")
    result = {"experiment": "wp11ac_balance_screen", "ledger_sha256": LEDGER_SHA256,
              "index": index, "pair": entry["pair"], "cardinalities": entry["cardinalities"],
              "detection_sha256": entry["detection_sha256"], "masses": masses,
              "numerical": numeric, "cpu_seconds": time.process_time() - started,
              "status": status}
    _save(output, result)
    return result


def validate(ledger_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    ledger = _validated_ledger(ledger_path)
    entries = _eligible(ledger)
    selected = None
    for entry in entries:
        record = json.loads((run_dir / f"{entry['index']}.json").read_text(encoding="utf-8"))
        if (record["index"] != entry["index"] or record["ledger_sha256"] != LEDGER_SHA256
                or record["detection_sha256"] != entry["detection_sha256"]):
            raise ValueError("screen does not match frozen ledger")
        if record["status"] == "balance_accepted_validate":
            selected = entry
            break
        if record["status"] != "balance_failed_continue":
            raise ValueError("numerical screen failure: stop")
    if selected is None:
        raise ValueError("no candidate passed frozen balance rule")
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["ledger_sha256"] != LEDGER_SHA256 or saved["selected_index"] != (
            selected["index"]
        ):
            raise ValueError("reference checkpoint differs from frozen selection")
        if saved["status"] == "validated_reference_no_sampler":
            return saved
        if saved["status"] != "reference_in_progress":
            raise ValueError("invalid reference checkpoint status")
    _limit(VALIDATION_CPU_CAP)
    case = _checked_case(selected)
    toy = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    screen_record = json.loads((run_dir / f"{selected['index']}.json").read_text())
    checks = {}
    for beta in (0.0, .5, 1.0):
        direct, paired = computer.compute(beta), computer.compute(beta, "paired")
        d, p = np.asarray(direct["joint"]), np.asarray(paired["joint"])
        comparison = {"log_partition_absolute_difference": abs(
            direct["log_partition"] - paired["log_partition"]),
            "joint_tv": float(np.abs(d - p).sum() / 2),
            "maximum_cell_absolute_difference": float(np.max(np.abs(d - p)))}
        if beta == 1.0 and any(abs(a - b) > NUMERIC_TV_TOL for a, b in zip(
            _masses(d)["alive_bands"], screen_record["masses"]["alive_bands"], strict=True
        )):
            raise ValueError("validated candidate differs from frozen beta-one screen")
        checks[str(beta)] = {"direct": direct, "paired_numerical": {
            **comparison, **{key: paired[key] for key in (
                "minimum_raw_cell", "maximum_relative_imaginary_residual",
                "raw_mass_error", "clipping_tv", "cpu_seconds")}}}
        if (not _numeric_ok(direct) or not _numeric_ok(paired)
                or comparison["log_partition_absolute_difference"] > NUMERIC_LOGZ_TOL
                or comparison["joint_tv"] > NUMERIC_TV_TOL
                or comparison["maximum_cell_absolute_difference"] > NUMERIC_CELL_TOL
                or beta == 0 and abs(direct["log_partition"]
                                     - case.model.base_log_normalizer) > NUMERIC_LOGZ_TOL):
            raise ValueError(f"selected numerical validation failed at beta={beta}")
        _save(output, {"experiment": "wp11ac_independent_exact_reference",
                       "ledger_sha256": LEDGER_SHA256, "selected_index": selected["index"],
                       "selected_detection_sha256": selected["detection_sha256"],
                       "toy_validations": toy, "selected_checks": checks,
                       "status": "reference_in_progress"})
    result = {"experiment": "wp11ac_independent_exact_reference",
              "ledger_sha256": LEDGER_SHA256, "selected_index": selected["index"],
              "selected_detection_sha256": selected["detection_sha256"],
              "toy_validations": toy, "selected_checks": checks,
              "base_log_normalizer": case.model.base_log_normalizer,
              "status": "validated_reference_no_sampler"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry", "screen", "validate"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--ledger", type=Path, default=Path("results/wp11ac_geometry_ledger.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11ac_screens"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ac_independent_exact_reference.json"))
    args = parser.parse_args()
    if args.case == "geometry":
        result = geometry(args.ledger)
    elif args.case == "screen":
        if args.index is None:
            parser.error("--index required for screen")
        result = screen(args.ledger, args.run_dir, args.index)
    else:
        result = validate(args.ledger, args.run_dir, args.output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
