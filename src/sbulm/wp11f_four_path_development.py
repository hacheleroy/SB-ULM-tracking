"""New development screen for sparse four-path conditional pair PG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from .wp5_scaling_calibration import _sample_case
from .wp7_convergence_audit import _save
from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)
from .wp11d_wp8_development import _cross_seed, _summary
from .wp11f_interleaved_cost import THREAD_ENV

REGIME = GeneralizationRegime(
    name="four_path_development", detection_probability=0.58,
    expected_clutter_per_frame=0.30, central_separation=0.005,
    trajectory_seed=53, corruption_seed=37, max_clutter_per_frame=2,
    truth_mask=STAGGERED_TRUTH_MASK,
)
EXPECTED_OBSERVATION_SHA256 = "7c211e3cfeac33e2e2487f5bcbaec6c204148668954bddbfffbfaec0537c1dd4"
EXPECTED_CARDINALITIES = (0, 3, 2, 3, 2, 3, 2, 1)
SEEDS = (410, 411)
N_PARTICLES = 512
N_SWEEPS = 2
N_REPEATS = 2
PG_FRACTION = 0.10
N_PATH_PARTICLES = 4
MAX_MEDIAN_CPU_RATIO = 2.0
ARMS = ("sparse_pg4", "local")


def _protocol() -> dict[str, Any]:
    return {
        "case": json.loads(json.dumps(asdict(REGIME))),
        "observation_sha256": EXPECTED_OBSERVATION_SHA256,
        "cardinalities": list(EXPECTED_CARDINALITIES),
        "baseline": "WP6 paired annealed SMC, 512 particles, 8 stage and final sweeps",
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "pair_sweeps": N_SWEEPS, "repeats": N_REPEATS,
        "order_by_seed": {"410": ["local", "sparse_pg4"],
                          "411": ["sparse_pg4", "local"]},
        "rng": "same seed+10000+repeat per arm from identical baseline copies",
        "path_particles": N_PATH_PARTICLES,
        "pg_fraction": PG_FRACTION,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "maximum_median_cpu_ratio_per_seed": MAX_MEDIAN_CPU_RATIO,
        "gate": (
            "each seed: mean count-change rate >= local; PG fraction in [.07,.13]; "
            "median paired CPU ratio <= 2"
        ),
        "interpretation": "new synthetic development only, not a holdout or exact eight-frame calibration",
    }


def _controls(arm: str) -> BirthDeathSMCConfig:
    common = {
        "n_particles": N_PARTICLES, "mh_sweeps": N_SWEEPS,
        "final_mh_sweeps": 0, "association_move_probability": 0,
        "interval_move_probability": 0, "slot_swap_probability": 0,
        "blocked_slot_move_probability": 0, "paired_slot_move_probability": 1,
    }
    if arm == "local":
        return BirthDeathSMCConfig(**common)
    if arm == "sparse_pg4":
        return BirthDeathSMCConfig(
            **common, paired_slot_scope="conditional_smc",
            paired_slot_path_particles=N_PATH_PARTICLES,
            paired_slot_pg_fraction=PG_FRACTION,
        )
    raise ValueError("unknown development arm")


def _run_arm(baseline: Any, model: Any, arm: str, rng_seed: int) -> dict[str, Any]:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    sampled = _metropolis_birth_death_rejuvenation(
        baseline.associations.copy(), baseline.log_marginal_densities.copy(),
        baseline.log_count_densities.copy(), baseline.log_existence_priors.copy(),
        model, beta=1.0, n_sweeps=N_SWEEPS, config=_controls(arm),
        rng=np.random.default_rng(rng_seed),
    )
    cpu_seconds = time.process_time() - start_cpu
    wall_seconds = time.perf_counter() - start_wall
    model._validate_associations(sampled[0])
    return {
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "state_change_rate": sampled[-1][6],
        "count_signature_change_rate": sampled[-1][9],
        "actual_pg_fraction": sampled[-1][10],
        "pg_count_signature_change_rate": sampled[-1][11],
        "summary": _summary(sampled[0], model),
    }


def _mean_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        field: np.mean([summary[field] for summary in summaries], axis=0).tolist()
        for field in (
            "alive_count_histograms", "assigned_count_histograms",
            "expected_clutter_by_frame",
        )
    }


def run_development(output: Path) -> dict[str, Any]:
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread environment variables to 1")
    case = make_case(REGIME)
    observed = _metadata(REGIME, case)["model_and_detections"]
    protocol = _protocol()
    if (
        observed["detections_sha256"] != EXPECTED_OBSERVATION_SHA256
        or tuple(observed["detection_cardinalities"]) != EXPECTED_CARDINALITIES
    ):
        raise RuntimeError("frozen development observation changed")
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different protocol")
        if result["status"] != "in_progress":
            return result
    else:
        result = {
            "experiment": "wp11f_new_four_path_development",
            "protocol": protocol, "runs": {str(seed): {} for seed in SEEDS},
            "status": "in_progress",
        }
        _save(output, result)
    for seed in SEEDS:
        baseline, baseline_seconds = _sample_case(
            case, method="paired", n_particles=N_PARTICLES, seed=seed
        )
        digest = hashlib.sha256(baseline.associations.tobytes()).hexdigest()
        if ("baseline" in result["runs"][str(seed)]
                and result["runs"][str(seed)]["baseline"]["association_sha256"] != digest):
            raise ValueError("recomputed baseline associations changed")
        result["runs"][str(seed)]["baseline"] = {
            "association_sha256": digest,
            "runtime_seconds": baseline_seconds,
            "tempering_stages": baseline.n_tempering_steps,
        }
        _save(output, result)
        print(f"seed {seed} baseline: {baseline_seconds:.1f}s", flush=True)
        for repeat in range(N_REPEATS):
            key = str(repeat)
            result["runs"][str(seed)].setdefault(key, {})
            order = protocol["order_by_seed"][str(seed)]
            if repeat % 2:
                order = list(reversed(order))
            for arm in order:
                if arm in result["runs"][str(seed)][key]:
                    continue
                result["runs"][str(seed)][key][arm] = _run_arm(
                    baseline, case.model, arm, seed + 10_000 + repeat
                )
                _save(output, result)
            pair = result["runs"][str(seed)][key]
            print(f"seed {seed} repeat {repeat}: CPU ratio "
                  f"{pair['sparse_pg4']['cpu_seconds'] / pair['local']['cpu_seconds']:.3f}",
                  flush=True)
    result["comparisons"] = {}
    averaged: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        blocks = [result["runs"][str(seed)][str(repeat)] for repeat in range(N_REPEATS)]
        cpu_ratios = [
            block["sparse_pg4"]["cpu_seconds"] / block["local"]["cpu_seconds"]
            for block in blocks
        ]
        local_changes = float(np.mean([
            block["local"]["count_signature_change_rate"] for block in blocks
        ]))
        pg_changes = float(np.mean([
            block["sparse_pg4"]["count_signature_change_rate"] for block in blocks
        ]))
        pg_fraction = float(np.mean([
            block["sparse_pg4"]["actual_pg_fraction"] for block in blocks
        ]))
        for arm in ARMS:
            averaged[arm][str(seed)] = _mean_summary([block[arm]["summary"] for block in blocks])
        result["comparisons"][str(seed)] = {
            "cpu_ratios": cpu_ratios,
            "median_cpu_ratio": float(np.median(cpu_ratios)),
            "mean_local_count_change_rate": local_changes,
            "mean_sparse_count_change_rate": pg_changes,
            "mean_pg_fraction": pg_fraction,
            "gate_passed": (
                np.median(cpu_ratios) <= MAX_MEDIAN_CPU_RATIO
                and pg_changes >= local_changes
                and 0.07 <= pg_fraction <= 0.13
            ),
        }
    result["cross_seed"] = {
        arm: _cross_seed(averaged[arm][str(SEEDS[0])], averaged[arm][str(SEEDS[1])])
        for arm in ARMS
    }
    result["status"] = (
        "four_path_development_screen_passed"
        if all(value["gate_passed"] for value in result["comparisons"].values())
        else "stopped_before_holdout"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp11f_four_path_development.json"))
    result = run_development(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
