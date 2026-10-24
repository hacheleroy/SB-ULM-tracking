"""Held-out four-slot posterior-stability checks across new synthetic regimes.

The observation cases, paired sampler, six-seed schedule, seven metrics, and
0.10 limit are frozen before inference. A case stops at the first failed
pairwise maximum, because additional runs cannot reverse that failure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import calibrate_matern_from_coordinate_std
from .simulation import (
    DetectionCorruptionConfig,
    MultiSlotCrossingConfig,
    simulate_multislot_crossing,
    simulate_slot_existence_observations,
)
from .slot_existence import SlotExistenceMaternModel
from .wp5_scaling_calibration import (
    DEFAULT_BIRTH_DEATH_PRIOR,
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

SEEDS = (100, 200, 101, 201, 102, 202)
EXPECTED_OBSERVATIONS = {
    "miss_heavy_simultaneous": {
        "detection_cardinalities": [1, 1, 3, 2, 1, 2, 1, 1],
        "detections_sha256": "68ab0a08c14bf656ffc52f904454a87b2cd5afb06b29e5bbd4eac9ffef575d8c",
    },
    "cluttered_staggered_crossing": {
        "detection_cardinalities": [1, 4, 4, 4, 3, 5, 1, 1],
        "detections_sha256": "afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae",
    },
}
STAGGERED_TRUTH_MASK = (
    (True, False, False, True),
    (True, True, False, True),
    (True, True, True, True),
    (True, True, True, True),
    (True, True, True, True),
    (False, True, True, True),
    (False, True, True, True),
    (False, False, True, True),
)


@dataclass(frozen=True)
class GeneralizationRegime:
    name: str
    detection_probability: float
    expected_clutter_per_frame: float
    central_separation: float
    trajectory_seed: int
    corruption_seed: int
    max_clutter_per_frame: int | None
    truth_mask: tuple[tuple[bool, bool, bool, bool], ...]


REGIMES = (
    GeneralizationRegime(
        name="miss_heavy_simultaneous",
        detection_probability=0.50,
        expected_clutter_per_frame=0.15,
        central_separation=0.0,
        trajectory_seed=17,
        corruption_seed=11,
        max_clutter_per_frame=1,
        truth_mask=LARGE_TRUTH_MASK,
    ),
    GeneralizationRegime(
        name="cluttered_staggered_crossing",
        detection_probability=0.65,
        expected_clutter_per_frame=0.45,
        central_separation=0.006,
        trajectory_seed=23,
        corruption_seed=12,
        max_clutter_per_frame=None,
        truth_mask=STAGGERED_TRUTH_MASK,
    ),
)


def make_case(regime: GeneralizationRegime) -> _ScalingCase:
    n_frames = len(regime.truth_mask)
    crossing = MultiSlotCrossingConfig(
        n_slots=4,
        n_frames=n_frames,
        frame_period=0.001,
        half_span=REFERENCE_SPEED_MM_S * 0.001 * (n_frames - 1) / 2.0,
        central_separation=regime.central_separation,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=regime.trajectory_seed)
    truth_mask = np.asarray(regime.truth_mask, dtype=bool)
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=regime.detection_probability,
            expected_clutter_per_frame=regime.expected_clutter_per_frame,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=regime.max_clutter_per_frame,
        ),
        seed=regime.corruption_seed,
    )
    calibration = calibrate_matern_from_coordinate_std(
        tuple(float(value) for value in np.std(truth.positions, axis=(0, 1))),
        rms_speed=REFERENCE_SPEED_MM_S,
    )
    model = SlotExistenceMaternModel(
        detections,
        n_slots=4,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=regime.detection_probability,
        expected_clutter_per_frame=regime.expected_clutter_per_frame,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model,
        truth_mask=truth_mask,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )


def _metadata(regime: GeneralizationRegime, case: _ScalingCase) -> dict[str, Any]:
    reference_ids = case.model.sequence.reference_ids
    if reference_ids is None:
        raise RuntimeError("synthetic reference IDs were lost")
    return {
        "regime": asdict(regime),
        "model_and_detections": _case_metadata(case),
        "truth_diagnostics": {
            "alive_slot_frames": int(np.count_nonzero(case.truth_mask)),
            "assigned_detections": int(sum(np.count_nonzero(ids >= 0) for ids in reference_ids)),
            "clutter_detections": int(sum(np.count_nonzero(ids < 0) for ids in reference_ids)),
        },
    }


def _load_or_initialize(output: Path, n_particles: int) -> dict[str, Any]:
    metadata = json.loads(
        json.dumps({regime.name: _metadata(regime, make_case(regime)) for regime in REGIMES})
    )
    for name, expected in EXPECTED_OBSERVATIONS.items():
        observed = metadata[name]["model_and_detections"]
        if any(observed[key] != value for key, value in expected.items()):
            raise RuntimeError(f"frozen observations changed in {name}")
    protocol = {
        "sampler": "unchanged WP6 likelihood-informed paired mixture",
        "n_particles": n_particles,
        "seeds_in_order": list(SEEDS),
        "stability_threshold": STABILITY_THRESHOLD,
        "gate": "all seven pairwise maximum differences at most 0.10",
        "early_stop": "first failed maximum is irreversible; no further seeds for that case",
        "interpretation": "cross-run stability only, not exact posterior calibration",
    }
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["cases"] != metadata or result["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different frozen protocol")
        return result
    return {
        "experiment": "wp8_heldout_synthetic_generalization",
        "protocol": protocol,
        "cases": metadata,
        "results": {
            regime.name: {
                "status": "in_progress",
                "runs": {},
                "cross_run": None,
            }
            for regime in REGIMES
        },
    }


def run_generalization(*, output: Path, n_particles: int = 8_192) -> dict[str, Any]:
    if n_particles < 2:
        raise ValueError("n_particles must be at least two")
    result = _load_or_initialize(output, n_particles)
    _save(output, result)
    for regime in REGIMES:
        case = make_case(regime)
        record = result["results"][regime.name]
        if record["status"] != "in_progress":
            print(f"{regime.name}: reused {record['status']}", flush=True)
            continue
        for seed in SEEDS:
            key = str(seed)
            if key not in record["runs"]:
                sampled, runtime = _sample_case(
                    case, method="paired", n_particles=n_particles, seed=seed
                )
                record["runs"][key] = _large_run_summary(case, sampled, runtime, seed)
                _save(output, result)
                print(
                    f"{regime.name} seed {seed}: {runtime:.1f}s, "
                    f"{sampled.n_tempering_steps} stages",
                    flush=True,
                )
            else:
                print(f"{regime.name} seed {seed}: reused", flush=True)
            complete = [record["runs"][str(item)] for item in SEEDS if str(item) in record["runs"]]
            if len(complete) >= 2:
                comparison = _compare_large(complete)
                record["cross_run"] = comparison
                if not comparison["stability_passed"]:
                    record["status"] = "failed_early"
                    _save(output, result)
                    print(
                        f"{regime.name}: FAIL after {len(complete)} seeds, "
                        f"worst={max(comparison['metrics'].values()):.4f}",
                        flush=True,
                    )
                    break
            if len(complete) == len(SEEDS):
                record["status"] = "passed_six_runs"
                _save(output, result)
                print(
                    f"{regime.name}: PASS over six seeds, "
                    f"worst={max(record['cross_run']['metrics'].values()):.4f}",
                    flush=True,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp8_generalization.json"))
    parser.add_argument("--n-particles", type=int, default=8_192)
    args = parser.parse_args()
    result = run_generalization(output=args.output, n_particles=args.n_particles)
    print({name: case["status"] for name, case in result["results"].items()})


if __name__ == "__main__":
    main()
