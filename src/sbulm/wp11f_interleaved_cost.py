"""Interleaved CPU-time benchmark on the exposed WP11e development case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .existence_smc import _metropolis_birth_death_rejuvenation
from .wp5_scaling_calibration import _sample_case
from .wp7_convergence_audit import _save
from .wp8_generalization import _metadata, make_case
from .wp11e_selective_pg_development import (
    EXPECTED_OBSERVATION_SHA256,
    N_PARTICLES,
    N_SWEEPS,
    REGIME,
    _controls,
)

SEEDS = (310, 311)
N_REPEATS = 4
MAX_MEDIAN_CPU_RATIO = 3.0
MIN_MEAN_PG_FRACTION = 0.07
MAX_MEAN_PG_FRACTION = 0.13
EXPECTED_BASELINE_HASHES = {
    "310": "8dbfe98e5306996f154338ee7675d6c406be9c568828b44d21a025f5606640d1",
    "311": "5361dd1ba25217758c9e32880a919f8d03a7c0397938b1d6b9b43d7cf39bf1e7",
}
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _protocol() -> dict[str, Any]:
    return {
        "case": "unchanged exposed WP11e synthetic development observation",
        "observation_sha256": EXPECTED_OBSERVATION_SHA256,
        "baseline_hashes": EXPECTED_BASELINE_HASHES,
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "pair_sweeps": N_SWEEPS,
        "repeats": N_REPEATS,
        "arm_order": [["local", "sparse_pg"] if index % 2 == 0 else ["sparse_pg", "local"]
                      for index in range(N_REPEATS)],
        "rng": "same seed+10000+repeat in each arm, fresh identical baseline copies",
        "thread_env": {name: "1" for name in THREAD_ENV},
        "max_median_cpu_ratio_per_seed": MAX_MEDIAN_CPU_RATIO,
        "mean_pg_fraction_range": [MIN_MEAN_PG_FRACTION, MAX_MEAN_PG_FRACTION],
        "gate": "for each seed, median CPU ratio <= 3; mean count movement >= local; PG fraction in range",
        "interpretation": "cost/movement diagnostic only; original wall-time gate remains failed",
    }


def _run_arm(initial: Any, model: Any, arm: str, rng_seed: int) -> dict[str, float]:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    result = _metropolis_birth_death_rejuvenation(
        initial.associations.copy(), initial.log_marginal_densities.copy(),
        initial.log_count_densities.copy(), initial.log_existence_priors.copy(),
        model, beta=1.0, n_sweeps=N_SWEEPS, config=_controls(arm),
        rng=np.random.default_rng(rng_seed),
    )
    elapsed_cpu = time.process_time() - start_cpu
    elapsed_wall = time.perf_counter() - start_wall
    model._validate_associations(result[0])
    return {
        "cpu_seconds": elapsed_cpu,
        "wall_seconds": elapsed_wall,
        "state_change_rate": result[-1][6],
        "count_signature_change_rate": result[-1][9],
        "actual_pg_fraction": result[-1][10],
        "pg_count_signature_change_rate": result[-1][11],
    }


def run_benchmark(output: Path) -> dict[str, Any]:
    observed_env = {name: os.environ.get(name) for name in THREAD_ENV}
    if any(value != "1" for value in observed_env.values()):
        raise RuntimeError("set all four BLAS/OMP thread environment variables to 1")
    case = make_case(REGIME)
    observed = _metadata(REGIME, case)["model_and_detections"]
    protocol = _protocol()
    if observed["detections_sha256"] != EXPECTED_OBSERVATION_SHA256:
        raise RuntimeError("frozen WP11e observation changed")
    if output.exists():
        record = json.loads(output.read_text(encoding="utf-8"))
        if record["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different protocol")
        if record["status"] != "in_progress":
            return record
    else:
        record = {
            "experiment": "wp11f_interleaved_cpu_time_on_exposed_wp11e",
            "protocol": protocol, "runs": {str(seed): {} for seed in SEEDS},
            "status": "in_progress",
        }
        _save(output, record)
    for seed in SEEDS:
        baseline, baseline_seconds = _sample_case(
            case, method="paired", n_particles=N_PARTICLES, seed=seed
        )
        digest = hashlib.sha256(baseline.associations.tobytes()).hexdigest()
        if digest != EXPECTED_BASELINE_HASHES[str(seed)]:
            raise ValueError("the baseline states changed under single-threaded execution")
        record["runs"][str(seed)]["baseline"] = {
            "association_sha256": digest, "runtime_seconds": baseline_seconds,
            "tempering_stages": baseline.n_tempering_steps,
        }
        _save(output, record)
        print(f"seed {seed} baseline reproduced: {baseline_seconds:.1f}s", flush=True)
        for repeat, arm_order in enumerate(protocol["arm_order"]):
            key = str(repeat)
            if key in record["runs"][str(seed)] and all(
                arm in record["runs"][str(seed)][key] for arm in arm_order
            ):
                continue
            record["runs"][str(seed)].setdefault(key, {})
            for arm in arm_order:
                if arm in record["runs"][str(seed)][key]:
                    continue
                record["runs"][str(seed)][key][arm] = _run_arm(
                    baseline, case.model, arm, seed + 10_000 + repeat
                )
                _save(output, record)
            local = record["runs"][str(seed)][key]["local"]
            sparse = record["runs"][str(seed)][key]["sparse_pg"]
            print(f"seed {seed} repeat {repeat}: CPU ratio "
                  f"{sparse['cpu_seconds'] / local['cpu_seconds']:.3f}, "
                  f"wall ratio {sparse['wall_seconds'] / local['wall_seconds']:.3f}",
                  flush=True)
    record["comparisons"] = {}
    for seed in SEEDS:
        blocks = [record["runs"][str(seed)][str(index)] for index in range(N_REPEATS)]
        cpu_ratios = np.asarray([
            block["sparse_pg"]["cpu_seconds"] / block["local"]["cpu_seconds"]
            for block in blocks
        ])
        wall_ratios = np.asarray([
            block["sparse_pg"]["wall_seconds"] / block["local"]["wall_seconds"]
            for block in blocks
        ])
        local_changes = float(np.mean([
            block["local"]["count_signature_change_rate"] for block in blocks
        ]))
        sparse_changes = float(np.mean([
            block["sparse_pg"]["count_signature_change_rate"] for block in blocks
        ]))
        pg_fraction = float(np.mean([
            block["sparse_pg"]["actual_pg_fraction"] for block in blocks
        ]))
        record["comparisons"][str(seed)] = {
            "cpu_ratios": cpu_ratios.tolist(),
            "wall_ratios": wall_ratios.tolist(),
            "median_cpu_ratio": float(np.median(cpu_ratios)),
            "median_wall_ratio": float(np.median(wall_ratios)),
            "mean_local_count_change_rate": local_changes,
            "mean_sparse_count_change_rate": sparse_changes,
            "mean_pg_fraction": pg_fraction,
            "gate_passed": (
                np.median(cpu_ratios) <= MAX_MEDIAN_CPU_RATIO
                and sparse_changes >= local_changes
                and MIN_MEAN_PG_FRACTION <= pg_fraction <= MAX_MEAN_PG_FRACTION
            ),
        }
    record["status"] = (
        "controlled_cost_screen_passed"
        if all(comparison["gate_passed"] for comparison in record["comparisons"].values())
        else "controlled_cost_screen_failed"
    )
    _save(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp11f_interleaved_cost.json"))
    result = run_benchmark(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
