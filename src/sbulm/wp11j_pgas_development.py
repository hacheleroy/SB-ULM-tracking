"""Frozen two-seed fresh-case screen of exact four-path PG ancestor sampling."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .existence_smc import sample_slot_existence_posterior
from .wp7_convergence_audit import _compare_large, _large_run_summary, _save
from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls

REGIME = GeneralizationRegime(
    name="pgas_ancestor_development", detection_probability=0.64,
    expected_clutter_per_frame=0.50, central_separation=0.005,
    trajectory_seed=73, corruption_seed=67, max_clutter_per_frame=3,
    truth_mask=STAGGERED_TRUTH_MASK,
)
OBSERVATION_SHA256 = "07f83b43fdd7555921378b43bb05e75d9abc896d8e268384e6e6e3d4e164adc7"
CARDINALITIES = (2, 3, 4, 5, 1, 2, 2, 2)
SEEDS = (530, 531)
ARMS = ("pg", "pgas")
N_PARTICLES = 512
CPU_LIMIT_SECONDS = 500
MAX_CPU_RATIO = 1.50
MIN_SELECTION_RATIO = 0.80
MAX_SELECTION_RATIO = 1.20
OTHER_METRIC_TOLERANCE = 0.02


def _controls(seed: int, arm: str):
    common = replace(_candidate_controls(seed), n_particles=N_PARTICLES)
    if arm == "pg":
        return common
    if arm == "pgas":
        return replace(common, paired_slot_pg_ancestor_sampling=True)
    raise ValueError("unknown development arm")


def _protocol() -> dict[str, Any]:
    return {
        "case": json.loads(json.dumps(asdict(REGIME))),
        "observation_sha256": OBSERVATION_SHA256,
        "cardinalities": list(CARDINALITIES),
        "seeds": list(SEEDS), "arms": list(ARMS),
        "arm_order_by_seed": {"530": ["pg", "pgas"],
                              "531": ["pgas", "pg"]},
        "n_particles": N_PARTICLES,
        "control": "unchanged WP11g 10% four-path PG mixture",
        "candidate": "same 10% four-path PG mixture with one exact full-suffix ancestor MH step",
        "thread_env": {name: "1" for name in THREAD_ENV},
        "max_process_cpu_seconds_per_run": CPU_LIMIT_SECONDS,
        "maximum_runs": 4,
        "maximum_candidate_control_cpu_ratio_each_seed": MAX_CPU_RATIO,
        "pg_selection_ratio_range_each_seed": [MIN_SELECTION_RATIO, MAX_SELECTION_RATIO],
        "other_metric_tolerance": OTHER_METRIC_TOLERANCE,
        "gate": (
            "both seeds CPU ratio <=1.50 and weighted PG-selection ratio in [.80,1.20]; "
            "cross-seed clutter and alive-count TV each lower than constant PG; "
            "each of five other metrics <= control + .02"
        ),
        "interpretation": "new synthetic development only, not a holdout or calibration",
    }


def _finish(result: dict[str, Any]) -> None:
    if any(set(result["runs"][str(seed)]) != set(ARMS) for seed in SEEDS):
        return
    comparisons = {
        arm: _compare_large([result["runs"][str(seed)][arm]["summary"] for seed in SEEDS])
        for arm in ARMS
    }
    control, candidate = (comparisons[arm]["metrics"] for arm in ARMS)
    target_metrics = ("maximum_clutter_marginal_difference", "maximum_alive_count_tv")
    other_metrics = set(control) - set(target_metrics)
    seed_checks = {}
    for seed in SEEDS:
        runs = result["runs"][str(seed)]
        cpu_ratio = runs["pgas"]["cpu_seconds"] / runs["pg"]["cpu_seconds"]
        selection_ratio = (
            runs["pgas"]["weighted_pg_selection_fraction"]
            / runs["pg"]["weighted_pg_selection_fraction"]
        )
        seed_checks[str(seed)] = {
            "candidate_control_cpu_ratio": cpu_ratio,
            "candidate_control_pg_selection_ratio": selection_ratio,
            "passed": bool(cpu_ratio <= MAX_CPU_RATIO
                           and MIN_SELECTION_RATIO <= selection_ratio <= MAX_SELECTION_RATIO),
        }
    result["cross_seed"] = comparisons
    result["seed_checks"] = seed_checks
    result["target_metrics_improved"] = all(candidate[key] < control[key]
                                            for key in target_metrics)
    result["other_metrics_within_tolerance"] = all(
        candidate[key] <= control[key] + OTHER_METRIC_TOLERANCE
        for key in other_metrics
    )
    result["status"] = (
        "development_screen_passed" if (
            all(check["passed"] for check in seed_checks.values())
            and result["target_metrics_improved"]
            and result["other_metrics_within_tolerance"]
        ) else "stopped_before_high_particle_test"
    )


def run_one(*, output: Path, seed: int, arm: str) -> dict[str, Any]:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("run must use a frozen seed and arm")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to 1")
    case = make_case(REGIME)
    observed = _metadata(REGIME, case)["model_and_detections"]
    if (observed["detections_sha256"] != OBSERVATION_SHA256
            or tuple(observed["detection_cardinalities"]) != CARDINALITIES):
        raise ValueError("development observation changed")
    protocol = _protocol()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing checkpoint has a different frozen protocol")
    else:
        result = {
            "experiment": "wp11j_pgas_development", "protocol": protocol,
            "runs": {str(s): {} for s in SEEDS}, "cross_seed": None,
            "status": "in_progress",
        }
        _save(output, result)
    if arm in result["runs"][str(seed)]:
        print(f"seed {seed} {arm}: reused", flush=True)
        return result
    if result["status"] != "in_progress":
        raise ValueError("development screen already completed")
    ordered = [(s, a) for s in SEEDS for a in protocol["arm_order_by_seed"][str(s)]]
    completed = [(s, a) for s, a in ordered if a in result["runs"][str(s)]]
    if (seed, arm) != ordered[len(completed)]:
        raise ValueError("run the frozen seed and arm order")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_LIMIT_SECONDS + 5:
        raise RuntimeError("system CPU hard limit is below frozen per-run budget")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS + 5))
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std, coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior, config=_controls(seed, arm),
    )
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    stages = sampled.stages
    selection_fraction = (
        sum(stage.paired_slot_pg_selected_fraction * stage.mh_sweeps for stage in stages)
        / sum(stage.mh_sweeps for stage in stages)
    )
    result["runs"][str(seed)][arm] = {
        "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
        "tempering_stages": sampled.n_tempering_steps,
        "weighted_pg_selection_fraction": selection_fraction,
        "summary": _large_run_summary(case, sampled, wall_seconds, seed),
    }
    _finish(result)
    _save(output, result)
    print(f"seed {seed} {arm}: {cpu_seconds:.1f}s CPU, "
          f"{sampled.n_tempering_steps} stages; status={result['status']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/wp11j_pgas_development.json"))
    args = parser.parse_args()
    run_one(output=args.output, seed=args.seed, arm=args.arm)


if __name__ == "__main__":
    main()
