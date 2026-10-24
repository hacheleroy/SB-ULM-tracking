"""Paired-seed diagnostic ablations of the failed WP8 synthetic realization.

Four counterfactual observation sets change one generative factor at a time.
The unchanged 8,192-particle paired sampler is compared on seeds 100 and 200;
an agreement on this pair is diagnostic only, not the WP8 six-run pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .simulation import DetectionSequence, MultiSlotCrossingConfig, simulate_multislot_crossing
from .slot_existence import SlotExistenceMaternModel
from .wp5_scaling_calibration import (
    REFERENCE_SPEED_MM_S,
    ROI_LOWER,
    ROI_UPPER,
    _sample_case,
    _ScalingCase,
)
from .wp7_convergence_audit import (
    LARGE_TRUTH_MASK,
    STABILITY_THRESHOLD,
    _case_metadata,
    _compare_large,
    _large_run_summary,
    _save,
)
from .wp8_generalization import REGIMES, make_case

N_PARTICLES = 8_192
SEEDS = (100, 200)
BASE = REGIMES[1]
ABLATIONS = (
    "remove_observed_clutter",
    "complete_true_detections",
    "widen_crossing",
    "wp7_existence_schedule",
)
INTERVENTIONS = {
    "remove_observed_clutter": "remove seven clutter points; set Poisson mean 0.45 to 0.15",
    "complete_true_detections": "restore every missed true localization; set detection p 0.65 to 0.95",
    "widen_crossing": "increase midpoint spacing 0.006 to 0.010 mm with identical noise and clutter",
    "wp7_existence_schedule": "switch to the 20-slot-frame WP7 mask; drop now-inactive true detections",
}
# Observation hashes and cardinalities were recorded before any WP9 inference.
EXPECTED_OBSERVATIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "detections_sha256": "afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae",
    },
    "remove_observed_clutter": {
        "detection_cardinalities": [1, 3, 4, 2, 2, 3, 1, 0],
        "detections_sha256": "7317c07164fadf31071f13ade6a689a918c5eef479c8751f3ca814b7eae839c2",
    },
    "complete_true_detections": {
        "detection_cardinalities": [2, 4, 4, 6, 5, 5, 3, 3],
        "detections_sha256": "a2907d08c529f9dfa5f2cf9cf33d828dae92aa39bdaa0a8a3006edbc78618ce9",
    },
    "widen_crossing": {
        "detection_cardinalities": [1, 4, 4, 4, 3, 5, 1, 1],
        "detections_sha256": "cc317188ecf08dfa566176cf00d504c3c910e49afd6b53a760a08f1fadcbf50c",
    },
    "wp7_existence_schedule": {
        "detection_cardinalities": [1, 3, 3, 4, 3, 5, 0, 1],
        "detections_sha256": "b07946e606a671406bd4996869cd96a507e6589aa8d19a07f722c478be5ce564",
    },
}


def _clean_source(separation: float) -> DetectionSequence:
    config = MultiSlotCrossingConfig(
        n_slots=4,
        n_frames=8,
        frame_period=0.001,
        half_span=REFERENCE_SPEED_MM_S * 0.001 * 7 / 2,
        central_separation=separation,
        localization_std=0.005,
    )
    _, clean = simulate_multislot_crossing(config, seed=BASE.trajectory_seed)
    return clean


def _coordinate_lookup(clean: DetectionSequence, frame: int) -> dict[int, np.ndarray]:
    ids = clean.reference_ids
    if ids is None:
        raise RuntimeError("clean synthetic identities are required")
    return {
        int(identity): point
        for identity, point in zip(ids[frame], clean.detections[frame], strict=True)
    }


def make_ablation_case(name: str) -> _ScalingCase:
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation: {name}")
    base = make_case(BASE)
    sequence = base.model.sequence
    if sequence.reference_ids is None:
        raise RuntimeError("the WP8 synthetic reference IDs are required")
    clean = _clean_source(BASE.central_separation)
    widened = _clean_source(0.010) if name == "widen_crossing" else None
    alternate_mask = np.asarray(LARGE_TRUTH_MASK, dtype=bool)
    if not np.all(alternate_mask <= base.truth_mask):
        raise RuntimeError("schedule intervention would require new random detections")

    frames: list[np.ndarray] = []
    ids_by_frame: list[np.ndarray] = []
    for frame_index, (points, references) in enumerate(
        zip(sequence.detections, sequence.reference_ids, strict=True)
    ):
        points = np.array(points, copy=True)
        references = np.array(references, copy=True)
        if name == "remove_observed_clutter":
            keep = references >= 0
            points, references = points[keep], references[keep]
        elif name == "complete_true_detections":
            source = _coordinate_lookup(clean, frame_index)
            missing = [
                slot
                for slot in range(4)
                if base.truth_mask[frame_index, slot] and slot not in references
            ]
            if missing:
                points = np.concatenate((points, np.stack([source[slot] for slot in missing])))
                references = np.concatenate((references, np.asarray(missing, dtype=np.int64)))
        elif name == "widen_crossing":
            assert widened is not None
            source = _coordinate_lookup(clean, frame_index)
            target = _coordinate_lookup(widened, frame_index)
            for index, identity in enumerate(references):
                if identity >= 0:
                    points[index] += target[int(identity)] - source[int(identity)]
        elif name == "wp7_existence_schedule":
            keep = np.asarray(
                [identity < 0 or alternate_mask[frame_index, identity] for identity in references],
                dtype=bool,
            )
            points, references = points[keep], references[keep]
        frames.append(points)
        ids_by_frame.append(references)

    modified = DetectionSequence(
        times=sequence.times,
        detections=tuple(frames),
        reference_ids=tuple(ids_by_frame),
    )
    if any(np.any(frame < ROI_LOWER) or np.any(frame > ROI_UPPER) for frame in frames):
        raise RuntimeError("an ablation moved a localization outside the frozen ROI")
    detection_probability = (
        0.95 if name == "complete_true_detections" else BASE.detection_probability
    )
    clutter_mean = 0.15 if name == "remove_observed_clutter" else BASE.expected_clutter_per_frame
    model = SlotExistenceMaternModel(
        modified,
        n_slots=4,
        localization_std=base.localization_std,
        coordinate_std=base.coordinate_std,
        length_scale=base.length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=clutter_mean,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
        birth_death_prior=base.model.birth_death_prior,
    )
    return _ScalingCase(
        model=model,
        truth_mask=alternate_mask if name == "wp7_existence_schedule" else base.truth_mask,
        localization_std=base.localization_std,
        coordinate_std=base.coordinate_std,
        length_scale=base.length_scale,
    )


def _metadata(name: str, case: _ScalingCase) -> dict[str, Any]:
    references = case.model.sequence.reference_ids
    if references is None:
        raise RuntimeError("synthetic reference IDs were lost")
    return {
        "intervention": INTERVENTIONS[name],
        "model_and_detections": _case_metadata(case),
        "coordinate_std": list(case.coordinate_std),
        "length_scale": case.length_scale,
        "truth_diagnostics": {
            "alive_slot_frames": int(np.count_nonzero(case.truth_mask)),
            "assigned_detections": int(sum(np.count_nonzero(ids >= 0) for ids in references)),
            "clutter_detections": int(sum(np.count_nonzero(ids < 0) for ids in references)),
        },
    }


def _load_or_initialize(output: Path, wp8_output: Path) -> dict[str, Any]:
    if set(EXPECTED_OBSERVATIONS) != set(ABLATIONS) | {"baseline"}:
        raise RuntimeError("ablation observations have not been frozen")
    base = make_case(BASE)
    base_meta = _case_metadata(base)
    if base_meta["detections_sha256"] != EXPECTED_OBSERVATIONS["baseline"]["detections_sha256"]:
        raise RuntimeError("WP8 baseline observations changed")
    wp8 = json.loads(wp8_output.read_text(encoding="utf-8"))
    reference = wp8["results"][BASE.name]
    if (
        wp8["protocol"]["n_particles"] != N_PARTICLES
        or wp8["cases"][BASE.name]["model_and_detections"] != base_meta
        or reference["status"] != "failed_early"
        or reference["cross_run"]["seeds"] != list(SEEDS)
    ):
        raise ValueError("WP8 baseline does not match the frozen failed pair")
    baseline = {
        "detections_sha256": base_meta["detections_sha256"],
        "cross_run": reference["cross_run"],
    }
    cases = {name: _metadata(name, make_ablation_case(name)) for name in ABLATIONS}
    for name in ABLATIONS:
        observed = cases[name]["model_and_detections"]
        expected = EXPECTED_OBSERVATIONS[name]
        if any(observed[key] != value for key, value in expected.items()):
            raise RuntimeError(f"frozen observations changed in {name}")
    protocol = {
        "sampler": "unchanged WP6 likelihood-informed paired mixture",
        "n_particles": N_PARTICLES,
        "seeds_in_order": list(SEEDS),
        "stability_threshold": STABILITY_THRESHOLD,
        "decision": "two-seed diagnostic screen only; below-gate is not a six-run pass",
        "prior": "same truth-informed WP8 hard-case calibration in all four interventions",
    }
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if (
            result["protocol"] != protocol
            or result["baseline"] != baseline
            or result["cases"] != cases
        ):
            raise ValueError("existing checkpoint belongs to a different ablation protocol")
        return result
    return {
        "experiment": "wp9_counterfactual_two_seed_ablation",
        "protocol": protocol,
        "baseline": baseline,
        "cases": cases,
        "results": {
            name: {"status": "in_progress", "runs": {}, "cross_run": None} for name in ABLATIONS
        },
    }


def run_ablation(*, output: Path, wp8_output: Path) -> dict[str, Any]:
    result = _load_or_initialize(output, wp8_output)
    _save(output, result)
    for name in ABLATIONS:
        record = result["results"][name]
        if record["status"] != "in_progress":
            print(f"{name}: reused {record['status']}", flush=True)
            continue
        case = make_ablation_case(name)
        for seed in SEEDS:
            key = str(seed)
            if key not in record["runs"]:
                sampled, runtime = _sample_case(
                    case, method="paired", n_particles=N_PARTICLES, seed=seed
                )
                record["runs"][key] = _large_run_summary(case, sampled, runtime, seed)
                _save(output, result)
                print(
                    f"{name} seed {seed}: {runtime:.1f}s, {sampled.n_tempering_steps} stages",
                    flush=True,
                )
        comparison = _compare_large([record["runs"][str(seed)] for seed in SEEDS])
        record["cross_run"] = comparison
        record["status"] = (
            "diagnostic_pair_below_gate"
            if comparison["stability_passed"]
            else "diagnostic_pair_failed"
        )
        _save(output, result)
        print(
            f"{name}: {record['status']}, worst={max(comparison['metrics'].values()):.4f}",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp9_ablation.json"))
    parser.add_argument("--wp8-output", type=Path, default=Path("results/wp8_generalization.json"))
    args = parser.parse_args()
    result = run_ablation(output=args.output, wp8_output=args.wp8_output)
    print({name: record["status"] for name, record in result["results"].items()})


if __name__ == "__main__":
    main()
