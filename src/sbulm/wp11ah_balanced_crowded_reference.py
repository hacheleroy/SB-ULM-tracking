"""Prospectively frozen two-frame decoy search for a crowded exact joint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .data import DetectionSequence
from .slot_existence import SlotExistenceMaternModel
from .wp5_scaling_calibration import ROI_LOWER, ROI_UPPER
from .wp7_convergence_audit import _case_metadata, _save
from .wp8_generalization import STAGGERED_TRUTH_MASK, GeneralizationRegime, make_case
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11v_tempering_mode_audit import (
    NUMERIC_CELL_TOL,
    NUMERIC_LOGZ_TOL,
    NUMERIC_TV_TOL,
    _ExactComputer,
    _numeric_ok,
    _toy_validations,
)
from .wp11w_balanced_exact_reference import _accepted, _limit, _masses

# This entire fresh ladder and the transformations are fixed before its
# first posterior is computed. Geometry eligibility alone chooses bases.
PAIRS = tuple((2411 + 11 * i, 2417 + 11 * i) for i in range(100))
SIXTH_OFFSETS = (.010, .016, .022)
LAST_OFFSETS = (.004, .008, .012)
MAX_SCREENS = 18
LEDGER_SHA256 = "a5978ee5e8e98ed537159ecc98888dd19e04e5745e7afb8a79f3ec064822d2d3"
SCREEN_CPU_CAP = 450
VALIDATION_CPU_CAP = 1_800


def _regime(index: int) -> GeneralizationRegime:
    trajectory, corruption = PAIRS[index]
    return GeneralizationRegime(
        name=f"wp11ah_base_{index}", detection_probability=.68,
        expected_clutter_per_frame=.28, central_separation=.007,
        trajectory_seed=trajectory, corruption_seed=corruption,
        max_clutter_per_frame=2, truth_mask=STAGGERED_TRUTH_MASK,
    )


def _protocol() -> dict[str, Any]:
    return {
        "pairs": [list(pair) for pair in PAIRS],
        "sixth_offsets_mm": list(SIXTH_OFFSETS),
        "last_offsets_mm": list(LAST_OFFSETS),
        "regime": {"detection_probability": .68, "expected_clutter_per_frame": .28,
                   "central_separation": .007, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "eligibility": "17..18 total detections, frame six 2..3 true and zero "
                       "clutter, final frame exactly 2 true and zero clutter, "
                       "frames 3..5 >=6, seventh frame >=1",
        "decoys": "largest nonnegative reference ID at frame six and final frame; "
                  "each y moves toward zero by its prespecified offset; labels become -1",
        "order": "eligible bases by increasing detection count then seed index; "
                 "round-robin across bases for each sixth-then-final offset pair",
        "max_screens": MAX_SCREENS,
        "generator_or_roi_error": "record and skip without replacement",
        "selection": "first numerically valid beta-one joint with low/high alive "
                     ">=.18, middle >=.05, two clutter categories >=.15 "
                     "at both frame six and final frame; stop on any numeric failure",
        "numeric": {"minimum_raw_cell": -1e-10, "max_imaginary_residual": 1e-8,
                    "raw_mass_error": 1e-8, "direct_paired_logz": NUMERIC_LOGZ_TOL,
                    "direct_paired_tv": NUMERIC_TV_TOL,
                    "direct_paired_max_cell": NUMERIC_CELL_TOL,
                    "base_logz": NUMERIC_LOGZ_TOL},
        "screen_cpu_cap": SCREEN_CPU_CAP,
        "validation_cpu_cap": VALIDATION_CPU_CAP,
        "no_sampler_or_holdout": True,
    }


def _base_eligible(case: Any) -> bool:
    sequence = case.model.sequence
    cards = sequence.cardinalities
    sixth, last = sequence.reference_ids[5], sequence.reference_ids[7]
    return (17 <= sum(cards) <= 18 and len(sixth) in (2, 3)
            and len(last) == 2 and np.all(sixth >= 0) and np.all(last >= 0)
            and sum(cards[2:5]) >= 6 and cards[6] >= 1)


def _decoy_case(base: Any, sixth_offset: float,
                last_offset: float) -> tuple[Any, dict[str, Any]]:
    if not _base_eligible(base):
        raise ValueError("base case is outside frozen geometry eligibility")
    sequence = base.model.sequence
    frames = list(sequence.detections)
    labels = list(sequence.reference_ids)
    changes = {}
    for frame_index, offset, name in ((5, sixth_offset, "sixth"),
                                      (7, last_offset, "last")):
        source_ids = labels[frame_index]
        index = int(np.argmax(source_ids))
        frame = frames[frame_index].copy()
        original_y = float(frame[index, 1])
        frame[index, 1] -= (1 if original_y >= 0 else -1) * offset
        if np.any(frame[index] < ROI_LOWER) or np.any(frame[index] > ROI_UPPER):
            raise ValueError(f"{name} decoy is outside the fixed ROI")
        frames[frame_index] = frame
        changed_ids = source_ids.copy()
        changed_ids[index] = -1
        labels[frame_index] = changed_ids
        changes[name] = {"index": index, "source_id": int(source_ids[index]),
                         "original_y": original_y, "decoy_y": float(frame[index, 1])}
    modified = DetectionSequence(
        times=sequence.times, detections=tuple(frames), reference_ids=tuple(labels)
    )
    model = SlotExistenceMaternModel(
        modified, n_slots=base.model.n_slots,
        localization_std=base.localization_std,
        coordinate_std=base.coordinate_std, length_scale=base.length_scale,
        detection_probability=base.model.detection_probability,
        expected_clutter_per_frame=base.model.expected_clutter_per_frame,
        roi_lower=base.model.roi_lower, roi_upper=base.model.roi_upper,
        birth_death_prior=base.model.birth_death_prior,
    )
    return replace(base, model=model), changes


def geometry(output: Path) -> dict[str, Any]:
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["protocol"] != _protocol():
            raise ValueError("geometry protocol changed")
        return saved
    bases, variants = [], []
    for base_index, pair in enumerate(PAIRS):
        try:
            base = make_case(_regime(base_index))
        except ValueError as exc:
            bases.append({"base_index": base_index, "pair": list(pair),
                          "eligible": False, "generator_error": str(exc)})
            continue
        sequence = base.model.sequence
        eligible = bool(_base_eligible(base))
        bases.append({"base_index": base_index, "pair": list(pair),
                      "cardinalities": list(sequence.cardinalities),
                      "base_detection_sha256": _case_metadata(base)["detections_sha256"],
                      "sixth_reference_ids": sequence.reference_ids[5].tolist(),
                      "last_reference_ids": sequence.reference_ids[7].tolist(),
                      "eligible": eligible})
        if not eligible:
            continue
        for sixth_offset in SIXTH_OFFSETS:
            for last_offset in LAST_OFFSETS:
                index = len(variants)
                try:
                    case, changes = _decoy_case(base, sixth_offset, last_offset)
                except ValueError as exc:
                    variants.append({"index": index, "base_index": base_index,
                                     "sixth_offset_mm": sixth_offset,
                                     "last_offset_mm": last_offset,
                                     "eligible": False, "geometry_error": str(exc)})
                    continue
                variants.append({
                    "index": index, "base_index": base_index, "pair": list(pair),
                    "sixth_offset_mm": sixth_offset, "last_offset_mm": last_offset,
                    "eligible": True,
                    "cardinalities": list(case.model.sequence.cardinalities),
                    "detection_sha256": _case_metadata(case)["detections_sha256"],
                    "decoys": changes,
                })
    result = {"experiment": "wp11ah_crowded_geometry_ledger", "protocol": _protocol(),
              "bases": bases, "variants": variants,
              "status": "geometry_only_before_posterior"}
    _save(output, result)
    return result


def _ledger(path: Path) -> dict[str, Any]:
    if LEDGER_SHA256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != (
        LEDGER_SHA256
    ):
        raise ValueError("geometry ledger is unpinned or changed")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["protocol"] != _protocol() or saved["status"] != (
        "geometry_only_before_posterior"
    ):
        raise ValueError("frozen geometry protocol differs")
    return saved


def _ordered(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    bases = sorted((base for base in ledger["bases"] if base["eligible"]),
                   key=lambda base: (sum(base["cardinalities"]), base["base_index"]))
    eligible = [entry for entry in ledger["variants"] if entry["eligible"]]
    by_base = {base["base_index"]: [entry for entry in eligible
                                     if entry["base_index"] == base["base_index"]]
               for base in bases}
    ordered = []
    for sixth_offset in SIXTH_OFFSETS:
        for last_offset in LAST_OFFSETS:
            ordered.extend(entry for base in bases
                           for entry in by_base[base["base_index"]]
                           if entry["sixth_offset_mm"] == sixth_offset
                           and entry["last_offset_mm"] == last_offset)
    return ordered[:MAX_SCREENS]


def _checked_case(entry: dict[str, Any]) -> Any:
    base = make_case(_regime(entry["base_index"]))
    case, changes = _decoy_case(base, entry["sixth_offset_mm"],
                                entry["last_offset_mm"])
    if (_case_metadata(case)["detections_sha256"] != entry["detection_sha256"]
            or list(case.model.sequence.cardinalities) != entry["cardinalities"]
            or changes != entry["decoys"]):
        raise ValueError("observation differs from frozen geometry ledger")
    return case


def screen(ledger_path: Path, run_dir: Path, index: int) -> dict[str, Any]:
    ordered = _ordered(_ledger(ledger_path))
    entry = next((v for v in ordered if v["index"] == index), None)
    if entry is None:
        raise ValueError("variant outside frozen screen order")
    for earlier in ordered[:ordered.index(entry)]:
        path = run_dir / f"{earlier['index']}.json"
        if not path.exists():
            raise ValueError("earlier screen missing, accepted, or failed numerically")
        previous = json.loads(path.read_text())
        if (previous["status"] != "balance_failed_continue"
                or previous["ledger_sha256"] != LEDGER_SHA256
                or previous["index"] != earlier["index"]
                or previous["detection_sha256"] != earlier["detection_sha256"]):
            raise ValueError("earlier screen differs from frozen ledger or failed")
    output = run_dir / f"{index}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved["ledger_sha256"] != LEDGER_SHA256
                or saved["index"] != index
                or saved["detection_sha256"] != entry["detection_sha256"]):
            raise ValueError("screen checkpoint differs")
        return saved
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    _limit(SCREEN_CPU_CAP)
    started = time.process_time()
    direct = _ExactComputer(_checked_case(entry).model, (5, 7)).compute(1.0)
    masses = _masses(np.asarray(direct["joint"]))
    status = ("numerical_failed_stop" if not _numeric_ok(direct)
              else "balance_accepted_validate" if _accepted(masses)
              else "balance_failed_continue")
    result = {
        "experiment": "wp11ah_crowded_screen", "ledger_sha256": LEDGER_SHA256,
        "index": index, "base_index": entry["base_index"],
        "sixth_offset_mm": entry["sixth_offset_mm"],
        "last_offset_mm": entry["last_offset_mm"],
        "detection_sha256": entry["detection_sha256"],
        "masses": masses,
        "numerical": {key: direct[key] for key in (
            "minimum_raw_cell", "maximum_relative_imaginary_residual",
            "raw_mass_error", "clipping_tv", "log_partition"
        )}, "cpu_seconds": time.process_time() - started,
        "status": status,
    }
    _save(output, result)
    return result


def validate(ledger_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    ordered = _ordered(_ledger(ledger_path))
    accepted = None
    for entry in ordered:
        record = json.loads((run_dir / f"{entry['index']}.json").read_text())
        if (record["ledger_sha256"] != LEDGER_SHA256
                or record["index"] != entry["index"]
                or record["detection_sha256"] != entry["detection_sha256"]):
            raise ValueError("screen differs from frozen ledger")
        if record["status"] == "balance_accepted_validate":
            accepted = entry
            break
        if record["status"] != "balance_failed_continue":
            raise ValueError("numerical failure stops validation")
    if accepted is None:
        raise ValueError("frozen ladder contains no accepted reference")
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved["ledger_sha256"] != LEDGER_SHA256
                or saved["index"] != accepted["index"]):
            raise ValueError("reference checkpoint differs")
        if saved["status"] == "validated_reference_no_sampler":
            return saved
        if saved["status"] != "reference_in_progress":
            raise ValueError("invalid reference status")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    _limit(VALIDATION_CPU_CAP)
    started = time.process_time()
    case = _checked_case(accepted)
    toy = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    screen_record = json.loads((run_dir / f"{accepted['index']}.json").read_text())
    checks = {}
    for beta in (0.0, .5, 1.0):
        direct, paired = computer.compute(beta), computer.compute(beta, "paired")
        d, p = np.asarray(direct["joint"]), np.asarray(paired["joint"])
        comparison = {
            "log_partition_absolute_difference": abs(direct["log_partition"]
                                                 - paired["log_partition"]),
            "joint_tv": float(np.abs(d - p).sum() / 2),
            "maximum_cell_absolute_difference": float(np.max(np.abs(d - p))),
        }
        if (beta == 1.0 and any(abs(a - b) > NUMERIC_TV_TOL for a, b in zip(
            _masses(d)["alive_bands"], screen_record["masses"]["alive_bands"],
            strict=True,
        ))):
            raise ValueError("validated posterior differs from screened bands")
        checks[str(beta)] = {"direct": direct, "paired_numerical": {
            **comparison, **{key: paired[key] for key in (
                "minimum_raw_cell", "maximum_relative_imaginary_residual",
                "raw_mass_error", "clipping_tv", "cpu_seconds"
            )}}}
        if (not _numeric_ok(direct) or not _numeric_ok(paired)
                or comparison["log_partition_absolute_difference"] > NUMERIC_LOGZ_TOL
                or comparison["joint_tv"] > NUMERIC_TV_TOL
                or comparison["maximum_cell_absolute_difference"] > NUMERIC_CELL_TOL
                or beta == 0.0 and abs(direct["log_partition"]
                                      - case.model.base_log_normalizer) > NUMERIC_LOGZ_TOL):
            raise ValueError(f"exact numerical validation failed at beta={beta}")
        _save(output, {"experiment": "wp11ah_crowded_exact_reference",
                       "ledger_sha256": LEDGER_SHA256, "index": accepted["index"],
                       "selected_detection_sha256": accepted["detection_sha256"],
                       "toy_validations": toy, "checks": checks,
                       "status": "reference_in_progress"})
    result = {"experiment": "wp11ah_crowded_exact_reference",
              "ledger_sha256": LEDGER_SHA256, "index": accepted["index"],
              "selected_detection_sha256": accepted["detection_sha256"],
              "toy_validations": toy, "checks": checks,
              "base_log_normalizer": case.model.base_log_normalizer,
              "cpu_seconds": time.process_time() - started,
              "status": "validated_reference_no_sampler"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry", "screen", "validate"),
                        required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11ah_geometry_ledger.json"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/wp11ah_screens"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ah_crowded_exact_reference.json"))
    args = parser.parse_args()
    if args.case == "geometry":
        result = geometry(args.ledger)
    elif args.case == "screen":
        if args.index is None:
            parser.error("screen requires --index")
        result = screen(args.ledger, args.run_dir, args.index)
    else:
        result = validate(args.ledger, args.run_dir, args.output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
