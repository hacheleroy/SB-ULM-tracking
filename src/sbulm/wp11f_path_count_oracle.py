"""Frozen four-versus-eight conditional-PG path-particle oracle pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from .slot_existence import enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11c_full_history_pilot import _pilot_model
from .wp11d_conditional_pair_pilot import _count_signatures

SEEDS = (81, 82, 83)
N_PARTICLES = 512
N_SWEEPS = 2
ARMS = (4, 8)
MIN_MEAN_FOUR_COUNT_CHANGE = 0.20
MAX_MEDIAN_CPU_RATIO = 0.75


def _protocol(model: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    for frame in model.sequence.detections:
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(frame, dtype="<f8").tobytes())
    return {
        "case": "unchanged 269-state two-slot/three-frame exact posterior",
        "observation_sha256": digest.hexdigest(),
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "sweeps": N_SWEEPS,
        "beta": 1.0,
        "arms_path_particles": list(ARMS),
        "arm_order": [[4, 8], [8, 4], [4, 8]],
        "finite_sample_envelope": "99% max-of-three count-TV from 2000 exact multinomial triplets, seed 1122",
        "minimum_mean_four_count_change": MIN_MEAN_FOUR_COUNT_CHANGE,
        "maximum_median_four_eight_cpu_ratio": MAX_MEDIAN_CPU_RATIO,
        "gate": "all six count-TV within envelope; N=4 mean change >= .20; median CPU ratio <= .75",
    }


def run_oracle(output: Path) -> dict[str, Any]:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    log_target = model.log_joint_density(support)
    exact = np.exp(log_target - logsumexp(log_target))
    _, inverse = np.unique(_count_signatures(support), axis=0, return_inverse=True)
    exact_count = np.bincount(inverse, weights=exact)
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    rng = np.random.default_rng(1122)
    maxima = np.empty(2000)
    for repeat in range(len(maxima)):
        samples = rng.multinomial(N_PARTICLES, exact_count, size=len(SEEDS))
        maxima[repeat] = np.max(np.sum(np.abs(samples / N_PARTICLES - exact_count), axis=1) / 2)
    envelope = float(np.quantile(maxima, 0.99))
    protocol = _protocol(model)
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if (result["protocol"] != protocol or result["exact_support_size"] != len(support)
                or result["exact_count_tv_max_q99"] != envelope):
            raise ValueError("existing checkpoint belongs to a different oracle protocol")
        if result["status"] != "in_progress":
            return result
    else:
        result = {
            "experiment": "wp11f_path_particle_count_exact_oracle",
            "protocol": protocol, "exact_support_size": len(support),
            "exact_count_tv_max_q99": envelope,
            "runs": {str(size): {} for size in ARMS}, "status": "in_progress",
        }
        _save(output, result)
    for seed, arm_order in zip(SEEDS, protocol["arm_order"], strict=True):
        start_rng = np.random.default_rng(seed)
        initial = support[start_rng.choice(len(support), size=N_PARTICLES, p=exact)]
        for size in arm_order:
            if str(seed) in result["runs"][str(size)]:
                continue
            config = BirthDeathSMCConfig(
                n_particles=N_PARTICLES, mh_sweeps=N_SWEEPS, final_mh_sweeps=0,
                association_move_probability=0, interval_move_probability=0,
                slot_swap_probability=0, blocked_slot_move_probability=0,
                paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
                paired_slot_path_particles=size,
            )
            start_cpu = time.process_time()
            start_wall = time.perf_counter()
            sampled = _metropolis_birth_death_rejuvenation(
                initial.copy(), model.log_marginal_density(initial),
                model.log_count_density(initial, include_poisson_void=False),
                model.log_existence_prior(initial), model,
                beta=1.0, n_sweeps=N_SWEEPS, config=config,
                rng=np.random.default_rng(seed + 10_000),
            )
            cpu_seconds = time.process_time() - start_cpu
            wall_seconds = time.perf_counter() - start_wall
            model._validate_associations(sampled[0])
            indices = np.asarray([lookup[state.tobytes()] for state in sampled[0]])
            histogram = np.bincount(inverse[indices], minlength=len(exact_count)) / N_PARTICLES
            tv = 0.5 * float(np.sum(np.abs(histogram - exact_count)))
            result["runs"][str(size)][str(seed)] = {
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "count_signature_change_rate": sampled[-1][9],
                "count_tv_against_exact": tv,
                "within_exact_count_tv_q99": tv <= envelope,
            }
            _save(output, result)
            print(f"seed {seed} paths {size}: count change {sampled[-1][9]:.3f}, "
                  f"exact TV {tv:.3f}, CPU {cpu_seconds:.1f}s", flush=True)
    cpu_ratios = [
        result["runs"]["4"][str(seed)]["cpu_seconds"]
        / result["runs"]["8"][str(seed)]["cpu_seconds"] for seed in SEEDS
    ]
    result["cpu_ratios_four_to_eight"] = cpu_ratios
    result["median_cpu_ratio_four_to_eight"] = float(np.median(cpu_ratios))
    result["mean_four_count_change_rate"] = float(np.mean([
        result["runs"]["4"][str(seed)]["count_signature_change_rate"] for seed in SEEDS
    ]))
    result["status"] = (
        "small_oracle_path_count_passed"
        if (all(run["within_exact_count_tv_q99"] for arm in result["runs"].values()
                for run in arm.values())
            and result["mean_four_count_change_rate"] >= MIN_MEAN_FOUR_COUNT_CHANGE
            and result["median_cpu_ratio_four_to_eight"] <= MAX_MEDIAN_CPU_RATIO)
        else "stopped_before_new_development_case"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp11f_path_count_oracle.json"))
    result = run_oracle(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
