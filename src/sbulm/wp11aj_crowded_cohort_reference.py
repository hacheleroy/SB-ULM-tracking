"""Prospective, geometry-ordered crowded absolute reference cohort."""

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

# Frozen before any WP11aj posterior. Eligibility and ordering use geometry only.
PAIRS = tuple((3611 + 13 * i, 3617 + 13 * i) for i in range(240))
SIXTH_OFFSETS = (.010, .016, .022)
LAST_OFFSETS = (.004, .008, .012)
STRATA = ("18", "19")
QUOTAS = {"18": 1, "19": 2}
MAX_SCREENS = {"18": 12, "19": 27}
SCREEN_CPU_CAP = 600
VALIDATION_CPU_CAP = 2_400
LEDGER_SHA256 = "cf3f078aa9bd07db20b1bc3ba02a71e5b331c7ce5c1ede473f94973c4f3ab675"


def _regime(index: int) -> GeneralizationRegime:
    trajectory, corruption = PAIRS[index]
    return GeneralizationRegime(
        name=f"wp11aj_base_{index}", detection_probability=.70,
        expected_clutter_per_frame=.28, central_separation=.007,
        trajectory_seed=trajectory, corruption_seed=corruption,
        max_clutter_per_frame=2, truth_mask=STAGGERED_TRUTH_MASK,
    )


def _protocol() -> dict[str, Any]:
    return {
        "pairs": [list(pair) for pair in PAIRS],
        "regime": {"detection_probability": .70, "expected_clutter_per_frame": .28,
                   "central_separation": .007, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "eligibility": "18 or 19 detections, frame six 2..3 true and zero "
                       "clutter, final exactly 2 true and zero clutter, "
                       "frames 3..5 >=6, seventh >=1, maximum frame count <=4",
        "sixth_offsets_mm": list(SIXTH_OFFSETS),
        "last_offsets_mm": list(LAST_OFFSETS),
        "decoys": "largest nonnegative reference ID in each of frame six and "
                  "final; y toward zero by frozen offset; labels become -1",
        "strata_order": list(STRATA), "quotas": QUOTAS,
        "max_screens_by_stratum": MAX_SCREENS,
        "order": "within each stratum, bases by minimum maximum frame count "
                 "then seed index, round-robin across bases at each "
                 "sixth-then-final offset pair",
        "selection": "first quota numerically sound balanced variants on distinct "
                     "bases per stratum; low/high alive >=.18, middle >=.05, "
                     "two clutter categories >=.15 at both specified frames",
        "stop": "generator/ROI errors ineligible; numerical failure or CPU cap "
                "stops entire cohort; no replacement, threshold change or "
                "partial cohort inference; exhaustion gives no cohort",
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
    return (sum(cards) in (18, 19) and max(cards) <= 4
            and len(sixth) in (2, 3) and len(last) == 2
            and np.all(sixth >= 0) and np.all(last >= 0)
            and sum(cards[2:5]) >= 6 and cards[6] >= 1)


def _decoy_case(base: Any, sixth_offset: float,
                last_offset: float) -> tuple[Any, dict[str, Any]]:
    if not _base_eligible(base):
        raise ValueError("base outside frozen geometry eligibility")
    sequence = base.model.sequence
    frames, labels, changes = list(sequence.detections), list(sequence.reference_ids), {}
    for frame_index, offset, name in ((5, sixth_offset, "sixth"),
                                      (7, last_offset, "last")):
        source_ids = labels[frame_index]
        index = int(np.argmax(source_ids))
        frame = frames[frame_index].copy()
        original_y = float(frame[index, 1])
        frame[index, 1] -= (1 if original_y >= 0 else -1) * offset
        if np.any(frame[index] < ROI_LOWER) or np.any(frame[index] > ROI_UPPER):
            raise ValueError(f"{name} decoy outside the fixed ROI")
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
        saved = json.loads(output.read_text())
        if saved["protocol"] != _protocol():
            raise ValueError("geometry protocol changed")
        return saved
    bases, variants = [], []
    for index, pair in enumerate(PAIRS):
        try:
            base = make_case(_regime(index))
        except ValueError as exc:
            bases.append({"base_index": index, "pair": list(pair),
                          "eligible": False, "generator_error": str(exc)})
            continue
        cards = list(base.model.sequence.cardinalities)
        eligible = bool(_base_eligible(base))
        bases.append({"base_index": index, "pair": list(pair),
                      "cardinalities": cards,
                      "base_detection_sha256": _case_metadata(base)["detections_sha256"],
                      "sixth_reference_ids": base.model.sequence.reference_ids[5].tolist(),
                      "last_reference_ids": base.model.sequence.reference_ids[7].tolist(),
                      "eligible": eligible})
        if not eligible:
            continue
        for sixth_offset in SIXTH_OFFSETS:
            for last_offset in LAST_OFFSETS:
                variant_index = len(variants)
                try:
                    case, changes = _decoy_case(base, sixth_offset, last_offset)
                except ValueError as exc:
                    variants.append({"index": variant_index, "base_index": index,
                                     "sixth_offset_mm": sixth_offset,
                                     "last_offset_mm": last_offset,
                                     "eligible": False, "geometry_error": str(exc)})
                    continue
                variants.append({
                    "index": variant_index, "base_index": index, "pair": list(pair),
                    "sixth_offset_mm": sixth_offset, "last_offset_mm": last_offset,
                    "eligible": True, "cardinalities": cards,
                    "detection_sha256": _case_metadata(case)["detections_sha256"],
                    "decoys": changes,
                })
    result = {"experiment": "wp11aj_crowded_cohort_geometry",
              "protocol": _protocol(), "bases": bases, "variants": variants,
              "status": "geometry_only_before_posterior"}
    _save(output, result)
    return result


def _ledger(path: Path) -> dict[str, Any]:
    if LEDGER_SHA256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != (
        LEDGER_SHA256
    ):
        raise ValueError("geometry ledger is unpinned or changed")
    saved = json.loads(path.read_text())
    if (saved["protocol"] != _protocol()
            or saved["status"] != "geometry_only_before_posterior"):
        raise ValueError("frozen geometry protocol differs")
    return saved


def _ordered(ledger: dict[str, Any], stratum: str) -> list[dict[str, Any]]:
    if stratum not in STRATA:
        raise ValueError("stratum outside frozen selection")
    bases = sorted((base for base in ledger["bases"] if base["eligible"]
                    and sum(base["cardinalities"]) == int(stratum)),
                   key=lambda base: (max(base["cardinalities"]), base["base_index"]))
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
    return ordered


def _checked_case(entry: dict[str, Any]) -> Any:
    base = make_case(_regime(entry["base_index"]))
    case, changes = _decoy_case(base, entry["sixth_offset_mm"],
                                entry["last_offset_mm"])
    if (_case_metadata(case)["detections_sha256"] != entry["detection_sha256"]
            or list(case.model.sequence.cardinalities) != entry["cardinalities"]
            or changes != entry["decoys"]):
        raise ValueError("observation differs from geometry ledger")
    return case


def _selection(ledger: dict[str, Any], run_dir: Path) -> tuple[
    dict[str, Any] | None, dict[str, list[dict[str, Any]]]
]:
    """Return the only allowed next screen and selected distinct bases."""
    selected: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for stratum in STRATA:
        accepted_bases: set[int] = set()
        screened = 0
        for entry in _ordered(ledger, stratum):
            if len(selected[stratum]) == QUOTAS[stratum]:
                break
            if entry["base_index"] in accepted_bases:
                continue
            if screened >= MAX_SCREENS[stratum]:
                break
            path = run_dir / f"{entry['index']}.json"
            if not path.exists():
                return entry, selected
            record = json.loads(path.read_text())
            if (record["ledger_sha256"] != LEDGER_SHA256
                    or record["index"] != entry["index"]
                    or record["detection_sha256"] != entry["detection_sha256"]
                    or record["stratum"] != stratum):
                raise ValueError("screen differs from frozen geometry ledger")
            screened += 1
            if record["status"] == "balance_accepted_validate":
                accepted_bases.add(entry["base_index"])
                selected[stratum].append(entry)
            elif record["status"] != "balance_failed_continue":
                raise ValueError("numerical failure stops entire cohort")
        if len(selected[stratum]) != QUOTAS[stratum]:
            raise ValueError(f"{stratum}-detection frozen ladder exhausted")
    return None, selected


def screen(ledger_path: Path, run_dir: Path, index: int) -> dict[str, Any]:
    entry, _ = _selection(_ledger(ledger_path), run_dir)
    if entry is None or index != entry["index"]:
        raise ValueError("screen is not the next frozen entry")
    _limit(SCREEN_CPU_CAP)
    started = time.process_time()
    direct = _ExactComputer(_checked_case(entry).model, (5, 7)).compute(1.0)
    masses = _masses(np.asarray(direct["joint"]))
    status = ("numerical_failed_stop" if not _numeric_ok(direct)
              else "balance_accepted_validate" if _accepted(masses)
              else "balance_failed_continue")
    result = {"experiment": "wp11aj_crowded_cohort_screen",
              "ledger_sha256": LEDGER_SHA256, "index": index,
              "stratum": str(sum(entry["cardinalities"])),
              "base_index": entry["base_index"],
              "detection_sha256": entry["detection_sha256"], "masses": masses,
              "numerical": {key: direct[key] for key in (
                  "minimum_raw_cell", "maximum_relative_imaginary_residual",
                  "raw_mass_error", "clipping_tv", "log_partition"
              )}, "cpu_seconds": time.process_time() - started, "status": status}
    _save(run_dir / f"{index}.json", result)
    return result


def validate(ledger_path: Path, run_dir: Path, reference_dir: Path,
             index: int) -> dict[str, Any]:
    next_entry, selected = _selection(_ledger(ledger_path), run_dir)
    if next_entry is not None:
        raise ValueError("frozen cohort screening is incomplete")
    entry = next((item for items in selected.values() for item in items
                  if item["index"] == index), None)
    if entry is None:
        raise ValueError("reference is outside selected frozen cohort")
    output = reference_dir / f"{index}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved["ledger_sha256"] != LEDGER_SHA256 or saved["index"] != index):
            raise ValueError("reference checkpoint differs")
        if saved["status"] == "validated_reference_no_sampler":
            return saved
        if saved["status"] != "reference_in_progress":
            raise ValueError("unexpected reference status")
    _limit(VALIDATION_CPU_CAP)
    started = time.process_time()
    case = _checked_case(entry)
    toy = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    screened = json.loads((run_dir / f"{index}.json").read_text())
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
            _masses(d)["alive_bands"], screened["masses"]["alive_bands"],
            strict=True,
        ))):
            raise ValueError("validated posterior differs from frozen screen")
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
        _save(output, {"experiment": "wp11aj_crowded_cohort_reference",
                       "ledger_sha256": LEDGER_SHA256, "index": index,
                       "selected_detection_sha256": entry["detection_sha256"],
                       "toy_validations": toy, "checks": checks,
                       "status": "reference_in_progress"})
    result = {"experiment": "wp11aj_crowded_cohort_reference",
              "ledger_sha256": LEDGER_SHA256, "index": index,
              "selected_detection_sha256": entry["detection_sha256"],
              "toy_validations": toy, "checks": checks,
              "base_log_normalizer": case.model.base_log_normalizer,
              "cpu_seconds": time.process_time() - started,
              "status": "validated_reference_no_sampler"}
    _save(output, result)
    return result


def manifest(ledger_path: Path, run_dir: Path, reference_dir: Path,
             output: Path) -> dict[str, Any]:
    next_entry, selected = _selection(_ledger(ledger_path), run_dir)
    if next_entry is not None:
        raise ValueError("cannot pin an incomplete cohort")
    rows = {}
    for stratum in STRATA:
        rows[stratum] = []
        for entry in selected[stratum]:
            reference_path = reference_dir / f"{entry['index']}.json"
            screen_path = run_dir / f"{entry['index']}.json"
            reference = json.loads(reference_path.read_text())
            if (reference["status"] != "validated_reference_no_sampler"
                    or reference["ledger_sha256"] != LEDGER_SHA256
                    or reference["index"] != entry["index"]
                    or reference["selected_detection_sha256"]
                    != entry["detection_sha256"]
                    or set(reference["checks"]) != {"0.0", "0.5", "1.0"}):
                raise ValueError("reference is incomplete or differs from selection")
            rows[stratum].append({
                "index": entry["index"], "base_index": entry["base_index"],
                "cardinalities": entry["cardinalities"],
                "detection_sha256": entry["detection_sha256"],
                "screen_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "masses": _masses(np.asarray(reference["checks"]["1.0"]["direct"]["joint"])),
            })
    result = {"experiment": "wp11aj_crowded_cohort_manifest",
              "geometry_sha256": LEDGER_SHA256, "protocol": _protocol(),
              "selected": rows,
              "status": "three_independently_validated_references_no_sampler_holdout_closed"}
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("existing frozen manifest changed")
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry", "next", "screen", "validate",
                                           "manifest"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11aj_geometry_ledger.json"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/wp11aj_screens"))
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("results/wp11aj_references"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/wp11aj_crowded_cohort_manifest.json"))
    args = parser.parse_args()
    if args.case == "geometry":
        result = geometry(args.ledger)
        print(f"geometry_only: {sum(b['eligible'] for b in result['bases'])} bases")
    elif args.case == "next":
        entry, selected = _selection(_ledger(args.ledger), args.run_dir)
        print(json.dumps({"next": None if entry is None else entry["index"],
                          "selected": {key: [x["index"] for x in values]
                                       for key, values in selected.items()}}))
    elif args.case == "screen":
        if args.index is None:
            parser.error("screen requires --index")
        result = screen(args.ledger, args.run_dir, args.index)
        print(f"{args.index}: {result['status']}", flush=True)
    elif args.case == "validate":
        if args.index is None:
            parser.error("validate requires --index")
        result = validate(args.ledger, args.run_dir, args.reference_dir, args.index)
        print(f"{args.index}: {result['status']}", flush=True)
    else:
        result = manifest(args.ledger, args.run_dir, args.reference_dir,
                          args.manifest)
        print(result["status"])


if __name__ == "__main__":
    main()
