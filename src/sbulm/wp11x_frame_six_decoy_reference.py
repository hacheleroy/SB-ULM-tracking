"""Geometry-frozen frame-six decoy reference search, without sampler inference."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from .wp11v_tempering_mode_audit import (
    NUMERIC_CELL_TOL,
    NUMERIC_LOGZ_TOL,
    NUMERIC_TV_TOL,
    _ExactComputer,
    _numeric_ok,
    _toy_validations,
)
from .wp11w_balanced_exact_reference import _accepted, _limit, _masses

PAIRS = ((271, 277), (281, 283), (293, 307), (311, 313),
         (317, 331), (337, 347), (349, 353), (359, 367),
         (373, 379), (383, 389), (397, 401), (409, 419),
         (421, 431), (433, 439), (443, 449), (457, 461),
         (463, 467), (479, 487), (491, 499), (503, 509),
         (521, 523), (541, 547), (557, 563), (569, 571))
OFFSETS = (.010, .016, .022)
MAX_SCREENS = 9
LEDGER_SHA256 = "d1eabae457ac7b4145828d4ee59566ca0811403ac616c4877b0b5f6858e35679"
REFERENCE_SHA256 = "d1834acda7022ba9d50d97a15115975909c6c938409969b8415e913dda62977c"
SCREEN_CPU_CAP = 450
VALIDATION_CPU_CAP = 1_800


def _regime(index: int) -> GeneralizationRegime:
    a, b = PAIRS[index]
    return GeneralizationRegime(
        name=f"wp11x_base_{index}", detection_probability=.68,
        expected_clutter_per_frame=.28, central_separation=.007,
        trajectory_seed=a, corruption_seed=b, max_clutter_per_frame=2,
        truth_mask=STAGGERED_TRUTH_MASK,
    )


def _protocol() -> dict[str, Any]:
    return {
        "pairs": [list(pair) for pair in PAIRS], "offsets_mm": list(OFFSETS),
        "regime": {"detection_probability": .68, "expected_clutter_per_frame": .28,
                   "central_separation": .007, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "eligibility": "17..19 total; frame six 2 or 3 true and zero clutter; "
                       "last >=1; frames 3..5 >=6",
        "decoy": "highest frame-six reference ID becomes clutter; y moves toward zero by offset",
        "errors": "record generator or ROI failure and skip; no replacement",
        "order": "base seed then ascending offset", "max_screens": MAX_SCREENS,
        "acceptance": "low and high >=.18; middle >=.05; two sixth and two last clutter >=.15",
        "numeric": {"minimum_raw_cell": -1e-10, "imaginary_residual": 1e-8,
                    "mass_error": 1e-8, "direct_paired_logz": NUMERIC_LOGZ_TOL,
                    "direct_paired_tv": NUMERIC_TV_TOL,
                    "direct_paired_max_cell": NUMERIC_CELL_TOL},
        "screen_cpu_cap": SCREEN_CPU_CAP, "validation_cpu_cap": VALIDATION_CPU_CAP,
        "no_sampler_or_holdout": True,
    }


def _base_eligible(case: Any) -> bool:
    sequence = case.model.sequence
    cards = sequence.cardinalities
    ids = sequence.reference_ids[5]
    return (17 <= sum(cards) <= 19 and len(ids) in (2, 3)
            and np.all(ids >= 0) and cards[7] >= 1 and sum(cards[2:5]) >= 6)


def _decoy_case(base: Any, offset: float) -> tuple[Any, dict[str, Any]]:
    sequence = base.model.sequence
    ids = sequence.reference_ids[5]
    if not _base_eligible(base):
        raise ValueError("base case is not geometry eligible")
    index = int(np.argmax(ids))
    frames = list(sequence.detections)
    labels = list(sequence.reference_ids)
    frame = frames[5].copy()
    source_y = float(frame[index, 1])
    frame[index, 1] -= (1 if source_y >= 0 else -1) * offset
    if np.any(frame[index] < ROI_LOWER) or np.any(frame[index] > ROI_UPPER):
        raise ValueError("frame-six decoy is outside the fixed ROI")
    frames[5] = frame
    changed_ids = labels[5].copy()
    changed_ids[index] = -1
    labels[5] = changed_ids
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
    return replace(base, model=model), {
        "replaced_frame_six_index": index,
        "replaced_reference_id": int(ids[index]),
        "original_y": source_y, "decoy_y": float(frame[index, 1]),
    }


def geometry(output: Path) -> dict[str, Any]:
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["protocol"] != _protocol():
            raise ValueError("ledger protocol changed")
        return saved
    bases, variants = [], []
    for base_index in range(len(PAIRS)):
        try:
            base = make_case(_regime(base_index))
        except ValueError as exc:
            bases.append({"base_index": base_index, "pair": list(PAIRS[base_index]),
                          "eligible": False, "generator_error": str(exc)})
            continue
        cards = tuple(base.model.sequence.cardinalities)
        entry = {"base_index": base_index, "pair": list(PAIRS[base_index]),
                 "cardinalities": list(cards),
                 "base_detection_sha256": _case_metadata(base)["detections_sha256"],
                 "frame_six_reference_ids": base.model.sequence.reference_ids[5].tolist(),
                 "eligible": bool(_base_eligible(base))}
        bases.append(entry)
        if not entry["eligible"]:
            continue
        for offset in OFFSETS:
            variant_index = len(variants)
            try:
                case, decoy = _decoy_case(base, offset)
            except ValueError as exc:
                variants.append({"variant_index": variant_index,
                                 "base_index": base_index, "offset_mm": offset,
                                 "eligible": False, "geometry_error": str(exc)})
                continue
            variants.append({
                "variant_index": variant_index, "base_index": base_index,
                "pair": list(PAIRS[base_index]), "offset_mm": offset,
                "eligible": True, "cardinalities": list(case.model.sequence.cardinalities),
                "detection_sha256": _case_metadata(case)["detections_sha256"],
                **decoy,
            })
    result = {"experiment": "wp11x_frame_six_geometry", "protocol": _protocol(),
              "bases": bases, "variants": variants,
              "status": "geometry_only_before_posterior"}
    _save(output, result)
    return result


def _ledger(path: Path) -> dict[str, Any]:
    if LEDGER_SHA256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != LEDGER_SHA256:
        raise ValueError("WP11x geometry ledger is unpinned or changed")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["protocol"] != _protocol() or saved["status"] != "geometry_only_before_posterior":
        raise ValueError("WP11x geometry protocol differs from frozen ledger")
    return saved


def _screen_order(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in ledger["variants"] if item["eligible"]][:MAX_SCREENS]


def validated_reference(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("validated WP11x reference checkpoint hash changed")
    saved = json.loads(path.read_text(encoding="utf-8"))
    ledger = _ledger(Path("results/wp11x_geometry_ledger.json"))
    entries = _screen_order(ledger)
    selected = entries[1]
    if (saved["status"] != "validated_reference_no_sampler"
            or saved["ledger_sha256"] != LEDGER_SHA256
            or saved["variant_index"] != selected["variant_index"]
            or saved["selected_detection_sha256"] != selected["detection_sha256"]):
        raise ValueError("WP11x exact reference differs from frozen selection")
    return saved


def _checked_case(entry: dict[str, Any]) -> Any:
    base = make_case(_regime(entry["base_index"]))
    case, decoy = _decoy_case(base, entry["offset_mm"])
    if (_case_metadata(case)["detections_sha256"] != entry["detection_sha256"]
            or tuple(case.model.sequence.cardinalities) != tuple(entry["cardinalities"])
            or any(decoy[key] != entry[key] for key in decoy)):
        raise ValueError("decoy observation differs from pinned ledger")
    return case


def screen(ledger_path: Path, run_dir: Path, variant_index: int) -> dict[str, Any]:
    ordered = _screen_order(_ledger(ledger_path))
    selected = next((item for item in ordered if item["variant_index"] == variant_index), None)
    if selected is None:
        raise ValueError("variant is outside frozen screen order")
    for item in ordered[:ordered.index(selected)]:
        path = run_dir / f"{item['variant_index']}.json"
        if not path.exists():
            raise ValueError("earlier variant has not been screened")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["status"] != "balance_failed_continue":
            raise ValueError("earlier variant accepted or failed numerically: stop")
    output = run_dir / f"{variant_index}.json"
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["ledger_sha256"] != LEDGER_SHA256 or saved["variant_index"] != (
            variant_index
        ):
            raise ValueError("variant checkpoint differs from frozen ledger")
        return saved
    _limit(SCREEN_CPU_CAP)
    case = _checked_case(selected)
    started = time.process_time()
    direct = _ExactComputer(case.model, (5, 7)).compute(1.0)
    masses = _masses(np.asarray(direct["joint"]))
    status = ("numerical_failed_stop" if not _numeric_ok(direct)
              else "balance_accepted_validate" if _accepted(masses)
              else "balance_failed_continue")
    result = {"experiment": "wp11x_decoy_screen", "ledger_sha256": LEDGER_SHA256,
              "variant_index": variant_index, "base_index": selected["base_index"],
              "offset_mm": selected["offset_mm"],
              "detection_sha256": selected["detection_sha256"],
              "masses": masses, "numerical": {key: direct[key] for key in (
                  "minimum_raw_cell", "maximum_relative_imaginary_residual",
                  "raw_mass_error", "clipping_tv", "log_partition")},
              "cpu_seconds": time.process_time() - started, "status": status}
    _save(output, result)
    return result


def validate(ledger_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    ordered = _screen_order(_ledger(ledger_path))
    selected = None
    for item in ordered:
        record = json.loads((run_dir / f"{item['variant_index']}.json").read_text())
        if (record["ledger_sha256"] != LEDGER_SHA256 or record["variant_index"] != (
            item["variant_index"]
        ) or record["detection_sha256"] != item["detection_sha256"]):
            raise ValueError("screen differs from frozen geometry")
        if record["status"] == "balance_accepted_validate":
            selected = item
            break
        if record["status"] != "balance_failed_continue":
            raise ValueError("screen failed numerically: stop")
    if selected is None:
        raise ValueError("frozen decoy ladder has no accepted reference")
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if (saved["ledger_sha256"] != LEDGER_SHA256 or saved["variant_index"] != (
            selected["variant_index"]
        )):
            raise ValueError("reference checkpoint differs from selected geometry")
        if saved["status"] == "validated_reference_no_sampler":
            return saved
        if saved["status"] != "reference_in_progress":
            raise ValueError("invalid reference checkpoint status")
    _limit(VALIDATION_CPU_CAP)
    case = _checked_case(selected)
    screen_record = json.loads((run_dir / f"{selected['variant_index']}.json").read_text())
    toy = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    checks = {}
    for beta in (0.0, .5, 1.0):
        direct, paired = computer.compute(beta), computer.compute(beta, "paired")
        d, p = np.asarray(direct["joint"]), np.asarray(paired["joint"])
        comparison = {"log_partition_absolute_difference": abs(
            direct["log_partition"] - paired["log_partition"]),
            "joint_tv": float(np.abs(d - p).sum() / 2),
            "maximum_cell_absolute_difference": float(np.max(np.abs(d - p)))}
        checks[str(beta)] = {"direct": direct, "paired_numerical": {
            **comparison, **{key: paired[key] for key in (
                "minimum_raw_cell", "maximum_relative_imaginary_residual",
                "raw_mass_error", "clipping_tv", "cpu_seconds")}}}
        if beta == 1.0 and any(abs(a - b) > NUMERIC_TV_TOL for a, b in zip(
            _masses(d)["alive_bands"], screen_record["masses"]["alive_bands"], strict=True
        )):
            raise ValueError("selected beta-one target differs from original screen")
        if (not _numeric_ok(direct) or not _numeric_ok(paired)
                or comparison["log_partition_absolute_difference"] > NUMERIC_LOGZ_TOL
                or comparison["joint_tv"] > NUMERIC_TV_TOL
                or comparison["maximum_cell_absolute_difference"] > NUMERIC_CELL_TOL
                or beta == 0 and abs(direct["log_partition"]
                                     - case.model.base_log_normalizer) > NUMERIC_LOGZ_TOL):
            raise ValueError(f"selected numerical validation failed at beta={beta}")
        _save(output, {"experiment": "wp11x_frame_six_exact_reference",
                       "ledger_sha256": LEDGER_SHA256,
                       "variant_index": selected["variant_index"],
                       "selected_detection_sha256": selected["detection_sha256"],
                       "toy_validations": toy, "selected_checks": checks,
                       "status": "reference_in_progress"})
    result = {"experiment": "wp11x_frame_six_exact_reference",
              "ledger_sha256": LEDGER_SHA256,
              "variant_index": selected["variant_index"],
              "selected_detection_sha256": selected["detection_sha256"],
              "toy_validations": toy, "selected_checks": checks,
              "base_log_normalizer": case.model.base_log_normalizer,
              "status": "validated_reference_no_sampler"}
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry", "screen", "validate"), required=True)
    parser.add_argument("--variant", type=int)
    parser.add_argument("--ledger", type=Path, default=Path("results/wp11x_geometry_ledger.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11x_screens"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11x_frame_six_exact_reference.json"))
    args = parser.parse_args()
    if args.case == "geometry":
        result = geometry(args.ledger)
    elif args.case == "screen":
        if args.variant is None:
            parser.error("--variant required for screen")
        result = screen(args.ledger, args.run_dir, args.variant)
    else:
        result = validate(args.ledger, args.run_dir, args.output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
