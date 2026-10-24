"""Four-slot exact bridge and fixed-sampler particle-scaling audit.

This does not tune the WP6 proposal or reinterpret its frozen failed gate.
The sparse four-frame oracle is a separate, pre-specified bridge case; the
eight-frame case reproduces the WP6 observation generator exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .calibration import calibrate_matern_from_coordinate_std
from .simulation import (
    DetectionCorruptionConfig,
    MultiSlotCrossingConfig,
    simulate_multislot_crossing,
    simulate_slot_existence_observations,
)
from .slot_existence import (
    ABSENT,
    SlotExistenceMaternModel,
    slot_event_marginals,
    slot_existence_support_size,
)
from .tsbm_reference import compare_tsbm_association_samples
from .wp5_scaling_calibration import (
    DEFAULT_BIRTH_DEATH_PRIOR,
    REFERENCE_SPEED_MM_S,
    ROI_LOWER,
    ROI_UPPER,
    _event_errors,
    _exact_posterior,
    _finite_sample_baseline,
    _histogram,
    _pairwise_histogram_tv,
    _pairwise_maximum,
    _sample_case,
    _ScalingCase,
)

BRIDGE_TRUTH_MASK = (
    (True, False, False, False),
    (True, True, False, False),
    (True, True, True, False),
    (True, True, True, True),
)
LARGE_TRUTH_MASK = (
    *BRIDGE_TRUTH_MASK,
    (True, True, True, True),
    (False, True, True, True),
    (False, False, True, True),
    (False, False, False, True),
)
STABILITY_THRESHOLD = 0.10
EVENT_ERROR_THRESHOLD = 0.06


def _make_case(*, bridge: bool) -> _ScalingCase:
    n_frames = 4 if bridge else 8
    detection_probability = 0.35 if bridge else 0.70
    corruption_seed = 3 if bridge else 5
    crossing = MultiSlotCrossingConfig(
        n_slots=4,
        n_frames=n_frames,
        frame_period=0.001,
        half_span=REFERENCE_SPEED_MM_S * 0.001 * (n_frames - 1) / 2.0,
        central_separation=0.0,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=9)
    truth_mask = np.asarray(BRIDGE_TRUTH_MASK if bridge else LARGE_TRUTH_MASK, dtype=bool)
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=detection_probability,
            expected_clutter_per_frame=0.15,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=corruption_seed,
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
        detection_probability=detection_probability,
        expected_clutter_per_frame=0.15,
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


def _case_metadata(case: _ScalingCase) -> dict[str, Any]:
    sequence = case.model.sequence
    digest = hashlib.sha256()
    digest.update(np.asarray(sequence.times, dtype="<f8").tobytes())
    for detections in sequence.detections:
        digest.update(np.asarray(detections.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(detections, dtype="<f8").tobytes())
    return {
        "n_slots": case.model.n_slots,
        "n_frames": sequence.n_frames,
        "detection_cardinalities": list(sequence.cardinalities),
        "detections_sha256": digest.hexdigest(),
        "detection_probability": case.model.detection_probability,
        "expected_clutter_per_frame": case.model.expected_clutter_per_frame,
        "birth_death_prior": asdict(case.model.birth_death_prior),
    }


def _load_or_initialize(output_path: Path) -> dict[str, Any]:
    bridge = _make_case(bridge=True)
    large = _make_case(bridge=False)
    if slot_existence_support_size(bridge.model) != 282_469:
        raise RuntimeError("the frozen four-slot oracle support has changed")
    if bridge.model.sequence.cardinalities != (1, 0, 1, 1):
        raise RuntimeError("the frozen four-slot oracle observations have changed")
    if large.model.sequence.cardinalities != (0, 2, 4, 3, 4, 2, 1, 1):
        raise RuntimeError("the WP6 observation cardinalities have changed")
    metadata = {
        "bridge": _case_metadata(bridge),
        "large": _case_metadata(large),
    }
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result["cases"] != metadata:
            raise ValueError("existing audit belongs to a different observation case")
        return result
    return {
        "experiment": "wp7_fixed_sampler_convergence_audit",
        "protocol": {
            "sampler_method": "paired",
            "sampler_configuration": (
                "unchanged WP6 paired mixture, ESS target 0.80, eight sweeps "
                "per stage and eight final sweeps"
            ),
            "exact_bridge": (
                "four slots, four frames, detection p=0.35, clutter 0.15, "
                "simulator seed 9, corruption seed 3"
            ),
            "large_case": "unchanged WP6 four-slot/eight-frame case",
            "event_error_threshold": EVENT_ERROR_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "thresholds_are_not_reestimated_from_large_case": True,
        },
        "cases": metadata,
        "bridge": {},
        "large_runs": {},
        "large_comparisons": {},
    }


def _save(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)


def run_bridge(
    result: dict[str, Any],
    output_path: Path,
    *,
    n_particles: int,
    seeds: tuple[int, ...],
    baseline_replicates: int = 100,
) -> None:
    case = _make_case(bridge=True)
    started = time.perf_counter()
    exact, log_evidence = _exact_posterior(case.model)
    print(
        f"exact bridge: {exact.probabilities.size:,} states in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    baseline = _finite_sample_baseline(
        exact,
        n_samples=n_particles,
        n_replicates=baseline_replicates,
        seed=10_003,
    )
    thresholds = {
        "total_variation": min(1.0, float(baseline["total_variation_q95"]) + 0.03),
        "jensen_shannon": min(float(np.log(2.0)), float(baseline["jensen_shannon_q95"]) + 0.02),
        "exact_mass_covered": max(0.0, float(baseline["exact_mass_covered_q05"]) - 0.03),
        "credible_set_recall_95": max(0.0, float(baseline["credible_set_recall_95_q05"]) - 0.03),
        "maximum_event_error": EVENT_ERROR_THRESHOLD,
    }
    key = str(n_particles)
    saved = result["bridge"].get(key)
    if saved is not None and (
        saved["baseline_replicates"] != baseline_replicates or saved["thresholds"] != thresholds
    ):
        raise ValueError("existing bridge baseline configuration differs")
    if saved is None:
        saved = {
            "n_particles": n_particles,
            "baseline_replicates": baseline_replicates,
            "exact_support_size": int(exact.probabilities.size),
            "exact_log_evidence": log_evidence,
            "effective_support": float(1.0 / np.sum(exact.probabilities**2)),
            "finite_sample_baseline": baseline,
            "thresholds": thresholds,
            "runs": {},
        }
        result["bridge"][key] = saved
        _save(output_path, result)
    exact_events = slot_event_marginals(
        exact.associations,
        case.model.sequence,
        probabilities=exact.probabilities,
    )
    for seed in seeds:
        if str(seed) in saved["runs"]:
            print(f"bridge {n_particles} seed {seed}: reused", flush=True)
            continue
        sampled, runtime = _sample_case(case, method="paired", n_particles=n_particles, seed=seed)
        comparison = compare_tsbm_association_samples(sampled.associations, exact)
        sampled_events = slot_event_marginals(sampled.associations, case.model.sequence)
        errors = _event_errors(exact_events, sampled_events)
        checks = {
            "valid_support": comparison.invalid_sample_fraction == 0.0,
            "total_variation": (
                comparison.total_variation_distance <= thresholds["total_variation"]
            ),
            "jensen_shannon": (
                comparison.jensen_shannon_divergence <= thresholds["jensen_shannon"]
            ),
            "exact_mass_covered": (
                comparison.exact_mass_covered_by_empirical_support
                >= thresholds["exact_mass_covered"]
            ),
            "credible_set_recall_95": (
                comparison.credible_set_recall_95 >= thresholds["credible_set_recall_95"]
            ),
            "event_marginals": max(errors.values()) <= EVENT_ERROR_THRESHOLD,
        }
        saved["runs"][str(seed)] = {
            "runtime_seconds": runtime,
            "comparison": comparison.to_dict(),
            "event_errors": errors,
            "maximum_event_error": max(errors.values()),
            "log_evidence_error": sampled.log_evidence_estimate - log_evidence,
            "final_unique_state_count": sampled.stages[-1].unique_state_count,
            "checks": checks,
            "passed": all(checks.values()),
        }
        _save(output_path, result)
        print(
            f"bridge {n_particles} seed {seed}: "
            f"{'PASS' if all(checks.values()) else 'FAIL'}, "
            f"max event error {max(errors.values()):.4f}, "
            f"TV {comparison.total_variation_distance:.4f}, {runtime:.1f}s",
            flush=True,
        )


def _large_run_summary(
    case: _ScalingCase, sampled: Any, runtime: float, seed: int
) -> dict[str, Any]:
    states = sampled.associations
    events = slot_event_marginals(states, case.model.sequence)
    alive_counts = np.count_nonzero(states != ABSENT, axis=(1, 2))
    assigned_counts = np.count_nonzero(states >= 0, axis=(1, 2))
    alive_by_frame = np.count_nonzero(states != ABSENT, axis=2)
    assigned_by_frame = np.count_nonzero(states >= 0, axis=2)
    signatures = np.concatenate((alive_by_frame, assigned_by_frame), axis=1)
    signatures_unique, counts = np.unique(signatures, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:5]
    signature_probabilities = counts / states.shape[0]
    stages = sampled.stages
    return {
        "seed": seed,
        "n_particles": int(states.shape[0]),
        "runtime_seconds": runtime,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_estimate": sampled.log_evidence_estimate,
        "minimum_stage_ess_fraction": min(stage.effective_sample_size for stage in stages)
        / states.shape[0],
        "mean_unique_parent_fraction": mean(stage.unique_parent_count for stage in stages)
        / states.shape[0],
        "final_unique_state_fraction": stages[-1].unique_state_count / states.shape[0],
        "mean_paired_acceptance_rate": mean(stage.paired_slot_acceptance_rate for stage in stages),
        "mean_paired_state_change_rate": mean(
            stage.paired_slot_state_change_rate for stage in stages
        ),
        "alive_count_histogram": _histogram(alive_counts, 32).tolist(),
        "assigned_count_histogram": _histogram(assigned_counts, 32).tolist(),
        "mean_alive_slot_frames": float(np.mean(alive_counts)),
        "mean_assigned_detections": float(np.mean(assigned_counts)),
        "event_marginals": {
            "existence": events.existence_probabilities.tolist(),
            "missed": events.missed_probabilities.tolist(),
            "clutter": events.expected_clutter.tolist(),
            "birth": events.birth_time_probabilities.tolist(),
            "death": events.death_time_probabilities.tolist(),
        },
        "unlabelled_count_signature": {
            "unique_count": int(counts.size),
            "entropy_nats": float(
                -np.sum(signature_probabilities * np.log(signature_probabilities))
            ),
            "top_five": [
                {
                    "per_frame_alive_then_assigned": signatures_unique[index].tolist(),
                    "mass": float(signature_probabilities[index]),
                }
                for index in order
            ],
        },
    }


def _compare_large(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("a stability comparison requires independent runs")
    event_names = {
        "existence": "maximum_existence_marginal_difference",
        "missed": "maximum_missed_marginal_difference",
        "clutter": "maximum_clutter_marginal_difference",
        "birth": "maximum_birth_marginal_difference",
        "death": "maximum_death_marginal_difference",
    }
    metrics = {
        output_name: _pairwise_maximum([np.asarray(run["event_marginals"][name]) for run in runs])
        for name, output_name in event_names.items()
    }
    metrics["maximum_alive_count_tv"] = _pairwise_histogram_tv(
        [np.asarray(run["alive_count_histogram"]) for run in runs]
    )
    metrics["maximum_assigned_count_tv"] = _pairwise_histogram_tv(
        [np.asarray(run["assigned_count_histogram"]) for run in runs]
    )
    return {
        "seeds": [run["seed"] for run in runs],
        "metrics": metrics,
        "checks": {name: value <= STABILITY_THRESHOLD for name, value in metrics.items()},
        "stability_passed": all(value <= STABILITY_THRESHOLD for value in metrics.values()),
        "mean_runtime_seconds": mean(run["runtime_seconds"] for run in runs),
    }


def run_large(
    result: dict[str, Any],
    output_path: Path,
    *,
    n_particles: int,
    seed_groups: tuple[tuple[int, ...], ...],
) -> None:
    case = _make_case(bridge=False)
    key = str(n_particles)
    result["large_runs"].setdefault(key, {})
    for seed in dict.fromkeys(seed for group in seed_groups for seed in group):
        if str(seed) in result["large_runs"][key]:
            print(f"large {n_particles} seed {seed}: reused", flush=True)
            continue
        sampled, runtime = _sample_case(case, method="paired", n_particles=n_particles, seed=seed)
        result["large_runs"][key][str(seed)] = _large_run_summary(case, sampled, runtime, seed)
        _save(output_path, result)
        print(
            f"large {n_particles} seed {seed}: "
            f"{runtime:.1f}s, stages {sampled.n_tempering_steps}, "
            f"unique {sampled.stages[-1].unique_state_count}/{n_particles}",
            flush=True,
        )
    result["large_comparisons"].setdefault(key, {})
    for group in seed_groups:
        label = ",".join(map(str, group))
        runs = [result["large_runs"][key][str(seed)] for seed in group]
        result["large_comparisons"][key][label] = _compare_large(runs)
    if len(seed_groups) > 1:
        all_seeds = list(dict.fromkeys(seed for group in seed_groups for seed in group))
        label = ",".join(map(str, all_seeds))
        runs = [result["large_runs"][key][str(seed)] for seed in all_seeds]
        result["large_comparisons"][key][label] = _compare_large(runs)
    _save(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp7_convergence_audit.json"))
    parser.add_argument("--bridge-only", action="store_true")
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument("--bridge-particles", type=int, default=2_048)
    parser.add_argument("--bridge-seeds", type=int, nargs="+", default=[300, 301, 302])
    parser.add_argument("--large-particles", type=int, nargs="+", default=[4_096])
    parser.add_argument("--large-seed-starts", type=int, nargs="+", default=[100, 200])
    parser.add_argument(
        "--large-seeds", type=int, nargs="+", help="one explicit diagnostic seed group"
    )
    parser.add_argument("--large-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.bridge_only and args.large_only:
        parser.error("choose at most one of --bridge-only and --large-only")
    if args.bridge_particles < 2 or args.large_repeats < 2:
        parser.error("particle and repeat counts are too small")
    if args.large_seeds is not None and len(set(args.large_seeds)) < 2:
        parser.error("--large-seeds requires at least two distinct seeds")
    result = _load_or_initialize(args.output)
    if not args.large_only:
        run_bridge(
            result,
            args.output,
            n_particles=args.bridge_particles,
            seeds=tuple(args.bridge_seeds),
        )
    if not args.bridge_only:
        groups = (
            (tuple(args.large_seeds),)
            if args.large_seeds is not None
            else tuple(
                tuple(range(start, start + args.large_repeats)) for start in args.large_seed_starts
            )
        )
        for n_particles in args.large_particles:
            run_large(result, args.output, n_particles=n_particles, seed_groups=groups)


if __name__ == "__main__":
    main()
