"""Frozen exact-oracle pilot for likelihood-guided pair particle Gibbs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from .slot_existence import ABSENT, enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11c_full_history_pilot import _pilot_model

SEEDS = (74, 75, 76)
N_PARTICLES = 512
N_SWEEPS = 2
N_PATH_PARTICLES = 8
MINIMUM_MEAN_COUNT_CHANGE_RATE = 0.01
ARMS = ("conditional_smc", "local")


def _count_signatures(states: np.ndarray) -> np.ndarray:
    """Label-free per-frame alive and assigned counts."""

    alive = np.count_nonzero(states != ABSENT, axis=2)
    assigned = np.count_nonzero(states >= 0, axis=2)
    return np.concatenate((alive, assigned), axis=1)


def _protocol() -> dict[str, Any]:
    model = _pilot_model()
    digest = hashlib.sha256()
    for frame in model.sequence.detections:
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(frame, dtype="<f8").tobytes())
    return {
        "case": "unchanged exact WP11c two-slot three-frame observations",
        "observation_sha256": digest.hexdigest(),
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "sweeps": N_SWEEPS,
        "path_particles": N_PATH_PARTICLES,
        "beta": 1.0,
        "arms": list(ARMS),
        "minimum_mean_count_signature_change_rate": MINIMUM_MEAN_COUNT_CHANGE_RATE,
        "stationarity_threshold": "99% maximum-of-three multinomial count-TV baseline, 1000 replicates",
        "gate": "both mixing and stationary count-TV checks before any larger development case",
    }


def run_pilot(output: Path) -> dict[str, Any]:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    exact_log = model.log_joint_density(support)
    exact = np.exp(exact_log - logsumexp(exact_log))
    _, inverse = np.unique(_count_signatures(support), axis=0, return_inverse=True)
    exact_count = np.bincount(inverse, weights=exact)
    support_lookup = {state.tobytes(): i for i, state in enumerate(support)}
    baseline_rng = np.random.default_rng(12_120)
    maxima = []
    for _ in range(1000):
        counts = baseline_rng.multinomial(N_PARTICLES, exact_count, size=len(SEEDS))
        tv = 0.5 * np.sum(np.abs(counts / N_PARTICLES - exact_count), axis=1)
        maxima.append(float(np.max(tv)))
    stationarity_threshold = float(np.quantile(maxima, 0.99))
    protocol = _protocol()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["exact_support_size"] != len(support):
            raise ValueError("existing checkpoint has a different frozen protocol")
        if result["exact_count_tv_max_q99"] != stationarity_threshold:
            raise ValueError("existing checkpoint has a different exact-count baseline")
    else:
        result = {
            "experiment": "wp11d_conditional_pair_pg_small_exact_pilot",
            "protocol": protocol,
            "exact_support_size": len(support),
            "exact_count_tv_max_q99": stationarity_threshold,
            "runs": {arm: {} for arm in ARMS},
            "status": "in_progress",
        }
        _save(output, result)

    for seed in SEEDS:
        initial_rng = np.random.default_rng(seed)
        initial = support[initial_rng.choice(len(support), size=N_PARTICLES, p=exact)]
        for arm in ARMS:
            if str(seed) in result["runs"][arm]:
                continue
            rng = np.random.default_rng(seed + 10_000)
            config = BirthDeathSMCConfig(
                n_particles=N_PARTICLES, mh_sweeps=N_SWEEPS, final_mh_sweeps=0,
                association_move_probability=0, interval_move_probability=0,
                slot_swap_probability=0, blocked_slot_move_probability=0,
                paired_slot_move_probability=1, paired_slot_scope=arm,
                paired_slot_path_particles=N_PATH_PARTICLES,
            )
            start = time.perf_counter()
            sampled = _metropolis_birth_death_rejuvenation(
                initial.copy(), model.log_marginal_density(initial),
                model.log_count_density(initial, include_poisson_void=False),
                model.log_existence_prior(initial), model,
                beta=1.0, n_sweeps=N_SWEEPS, config=config, rng=rng,
            )
            model._validate_associations(sampled[0])
            selected = np.asarray(
                [support_lookup[state.tobytes()] for state in sampled[0]], dtype=np.int64
            )
            empirical_count = np.bincount(inverse[selected], minlength=len(exact_count)) / N_PARTICLES
            count_tv = 0.5 * float(np.sum(np.abs(empirical_count - exact_count)))
            result["runs"][arm][str(seed)] = {
                "config": asdict(config),
                "state_change_rate": sampled[-1][6],
                "count_signature_change_rate": sampled[-1][9],
                "count_tv_against_exact": count_tv,
                "within_exact_count_tv_q99": count_tv <= stationarity_threshold,
                "runtime_seconds": time.perf_counter() - start,
            }
            _save(output, result)
            print(
                f"{arm} seed {seed}: count change {sampled[-1][9]:.4f}, "
                f"exact-count TV {count_tv:.4f}", flush=True,
            )
    pg = [result["runs"]["conditional_smc"][str(seed)] for seed in SEEDS]
    result["mean_pg_count_change_rate"] = float(np.mean(
        [run["count_signature_change_rate"] for run in pg]
    ))
    result["mixing_gate_passed"] = (
        result["mean_pg_count_change_rate"] >= MINIMUM_MEAN_COUNT_CHANGE_RATE
    )
    result["stationarity_gate_passed"] = all(
        run["within_exact_count_tv_q99"] for run in pg
    )
    result["status"] = (
        "small_case_gate_passed"
        if result["mixing_gate_passed"] and result["stationarity_gate_passed"]
        else "stopped_before_larger_development"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/wp11d_conditional_pair_pilot.json")
    )
    result = run_pilot(parser.parse_args().output)
    print(f"{result['status']}: mean PG count change {result['mean_pg_count_change_rate']:.4f}")


if __name__ == "__main__":
    main()
