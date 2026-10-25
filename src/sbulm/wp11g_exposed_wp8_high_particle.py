"""Two-seed high-particle development test of the frozen four-path PG mixture.

This reuses the already-exposed WP8 difficult observation and its saved
control runs. It does not open another holdout or establish exact calibration.
Run each seed in a separate process to enforce a per-run CPU-time limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .existence_smc import sample_slot_existence_posterior
from .wp5_scaling_calibration import _controls
from .wp7_convergence_audit import STABILITY_THRESHOLD, _compare_large, _large_run_summary, _save
from .wp8_generalization import EXPECTED_OBSERVATIONS, REGIMES, _metadata, make_case
from .wp11f_interleaved_cost import THREAD_ENV

CASE_NAME = "cluttered_staggered_crossing"
SEEDS = (100, 200)
N_PARTICLES = 8_192
N_PATH_PARTICLES = 4
PG_FRACTION = 0.10
CPU_LIMIT_SECONDS = 1_800
CONTROL_SHA256 = "c1c1b8f00fd6bd43ac5b0ab25858b505010b113cc40e2d2cdc8cb2f9f6ea0e53"


def _candidate_controls(seed: int):
    return replace(
        _controls("paired", n_particles=N_PARTICLES, seed=seed),
        paired_slot_scope="conditional_smc",
        paired_slot_path_particles=N_PATH_PARTICLES,
        paired_slot_pg_fraction=PG_FRACTION,
    )


def _protocol() -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "observation_sha256": EXPECTED_OBSERVATIONS[CASE_NAME]["detections_sha256"],
        "control_file_sha256": CONTROL_SHA256,
        "control": "unchanged, saved WP8 paired SMC runs (no rerun)",
        "candidate": "WP6 paired mixture with 10% of paired proposals replaced by four-path PG",
        "candidate_controls_seed_100": asdict(_candidate_controls(100)),
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "maximum_process_cpu_seconds_per_candidate_run": CPU_LIMIT_SECONDS,
        "maximum_candidate_runs": len(SEEDS),
        "stability_threshold": STABILITY_THRESHOLD,
        "gate": "all seven inherited pairwise label-free differences <= 0.10",
        "early_stop": "evaluate after seed 200; no new observation or seed after a failure",
        "interpretation": "two-seed exposed-case diagnostic, not six-run stability or exact calibration",
    }


def _frozen_inputs(control_path: Path):
    if hashlib.sha256(control_path.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("WP8 control checkpoint hash changed")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    regime = next(regime for regime in REGIMES if regime.name == CASE_NAME)
    case = make_case(regime)
    metadata = json.loads(json.dumps(_metadata(regime, case)))
    if metadata != control["cases"][CASE_NAME]:
        raise ValueError("observation/model metadata differs from frozen WP8 control")
    if metadata["model_and_detections"]["detections_sha256"] != _protocol()["observation_sha256"]:
        raise ValueError("frozen difficult observation changed")
    if control["protocol"]["n_particles"] != N_PARTICLES:
        raise ValueError("control particle count differs")
    if control["protocol"]["sampler"] != "unchanged WP6 likelihood-informed paired mixture":
        raise ValueError("control sampler differs")
    record = control["results"][CASE_NAME]
    if record["status"] != "failed_early" or set(record["runs"]) != {str(s) for s in SEEDS}:
        raise ValueError("expected the original failed two-seed control")
    runs = [record["runs"][str(seed)] for seed in SEEDS]
    if any(run["seed"] != seed or run["n_particles"] != N_PARTICLES
           for seed, run in zip(SEEDS, runs)):
        raise ValueError("control seeds or particle counts differ")
    if _compare_large(runs) != record["cross_run"]:
        raise ValueError("saved control seven-metric comparison differs")
    return case, record, metadata


def _finish(result: dict[str, Any]) -> None:
    runs = result["candidate_runs"]
    if any(str(seed) not in runs for seed in SEEDS):
        return
    comparison = _compare_large([runs[str(seed)]["summary"] for seed in SEEDS])
    original = result["control_cross_run"]
    comparison["delta_from_control"] = {
        metric: value - original["metrics"][metric]
        for metric, value in comparison["metrics"].items()
    }
    result["candidate_cross_run"] = comparison
    result["status"] = (
        "two_seed_exposed_case_passed" if comparison["stability_passed"]
        else "failed_early_before_new_holdout"
    )


def run_seed(*, output: Path, control_path: Path, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed must belong to the frozen WP8 first pair")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread environment variables to 1")
    case, control, metadata = _frozen_inputs(control_path)
    protocol = _protocol()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["case"] != metadata:
            raise ValueError("existing checkpoint belongs to a different frozen protocol")
    else:
        result = {
            "experiment": "wp11g_exposed_wp8_high_particle",
            "protocol": protocol,
            "case": metadata,
            "control_runs": {str(s): control["runs"][str(s)] for s in SEEDS},
            "control_cross_run": control["cross_run"],
            "candidate_runs": {},
            "candidate_cross_run": None,
            "status": "in_progress",
        }
        _save(output, result)
    if str(seed) in result["candidate_runs"]:
        print(f"seed {seed}: reused", flush=True)
        return result
    if result["status"] != "in_progress":
        raise ValueError("frozen diagnostic has already completed")
    if seed == SEEDS[1] and str(SEEDS[0]) not in result["candidate_runs"]:
        raise ValueError("run seed 100 first")
    controls = _candidate_controls(seed)
    # CPU is a hard per-process budget, so each candidate seed is a separate invocation.
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_LIMIT_SECONDS + 5:
        raise RuntimeError("system hard CPU limit is below the frozen per-run budget")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS + 5))
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence,
        n_slots=case.model.n_slots,
        localization_std=case.localization_std,
        coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower,
        roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior,
        config=controls,
    )
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    summary = _large_run_summary(case, sampled, wall_seconds, seed)
    result["candidate_runs"][str(seed)] = {
        "summary": summary,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "mean_stage_pg_selection_fraction": sum(
            stage.paired_slot_pg_selected_fraction for stage in sampled.stages
        ) / len(sampled.stages),
        "mean_stage_paired_count_change_rate": sum(
            stage.paired_slot_count_signature_change_rate for stage in sampled.stages
        ) / len(sampled.stages),
    }
    _finish(result)
    _save(output, result)
    print(f"seed {seed}: {cpu_seconds:.1f}s CPU, {wall_seconds:.1f}s wall, "
          f"{sampled.n_tempering_steps} stages; status={result['status']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--control", type=Path, default=Path("results/wp8_generalization.json"))
    parser.add_argument("--output", type=Path, default=Path("results/wp11g_exposed_wp8_high_particle.json"))
    args = parser.parse_args()
    result = run_seed(output=args.output, control_path=args.control, seed=args.seed)
    if result["candidate_cross_run"] is not None:
        print(result["candidate_cross_run"]["metrics"], flush=True)


if __name__ == "__main__":
    main()
