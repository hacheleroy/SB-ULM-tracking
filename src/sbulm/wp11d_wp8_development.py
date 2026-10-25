"""Paired operational screen on the previously exposed difficult WP8 data.

These are deliberately small, posterior-initialized development runs, not a
new eight-frame holdout and not the inherited 8,192-particle stability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from .slot_existence import ABSENT
from .wp5_scaling_calibration import _sample_case
from .wp7_convergence_audit import _save
from .wp8_generalization import EXPECTED_OBSERVATIONS, REGIMES, _metadata, make_case

SEEDS = (100, 200)
N_PARTICLES = 512
N_SWEEPS = 2
N_PATH_PARTICLES = 8
ARMS = ("conditional_smc", "local")
MAX_RUNTIME_RATIO = 10.0


def _histograms(states: np.ndarray, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    alive = np.count_nonzero(states != ABSENT, axis=2)
    assigned = np.count_nonzero(states >= 0, axis=2)
    alive_hist = np.stack([
        np.bincount(alive[:, t], minlength=n_slots + 1) / len(states)
        for t in range(states.shape[1])
    ])
    assigned_hist = np.stack([
        np.bincount(assigned[:, t], minlength=n_slots + 1) / len(states)
        for t in range(states.shape[1])
    ])
    return alive_hist, assigned_hist


def _summary(states: np.ndarray, model: Any) -> dict[str, Any]:
    alive, assigned = _histograms(states, model.n_slots)
    return {
        "alive_count_histograms": alive.tolist(),
        "assigned_count_histograms": assigned.tolist(),
        "expected_clutter_by_frame": [
            float(cardinality - np.mean(np.count_nonzero(states[:, t] >= 0, axis=1)))
            for t, cardinality in enumerate(model.sequence.cardinalities)
        ],
        "unique_labelled_states": len({state.tobytes() for state in states}),
        "mean_log_joint": float(np.mean(model.log_joint_density(states))),
    }


def _cross_seed(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    alive = np.asarray(first["alive_count_histograms"]) - np.asarray(
        second["alive_count_histograms"]
    )
    assigned = np.asarray(first["assigned_count_histograms"]) - np.asarray(
        second["assigned_count_histograms"]
    )
    clutter = np.asarray(first["expected_clutter_by_frame"]) - np.asarray(
        second["expected_clutter_by_frame"]
    )
    return {
        "max_alive_count_tv": float(np.max(np.sum(np.abs(alive), axis=1) / 2)),
        "max_assigned_count_tv": float(np.max(np.sum(np.abs(assigned), axis=1) / 2)),
        "max_expected_clutter_difference": float(np.max(np.abs(clutter))),
    }


def _protocol() -> dict[str, Any]:
    return {
        "case": "unchanged WP8 cluttered_staggered_crossing",
        "observation_sha256": EXPECTED_OBSERVATIONS[
            "cluttered_staggered_crossing"
        ]["detections_sha256"],
        "baseline_sampler": "WP6 paired annealed SMC, 512 particles, 8 stage and 8 final sweeps",
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "beta": 1.0,
        "post_baseline_pair_sweeps": N_SWEEPS,
        "path_particles": N_PATH_PARTICLES,
        "arms": list(ARMS),
        "max_runtime_ratio": MAX_RUNTIME_RATIO,
        "gate": (
            "each seed must have conditional-PG count changes >= local and "
            "conditional-PG/local pair-update runtime <= 10; no new holdout"
        ),
        "interpretation": "development mixing/cost screen, not 8192-particle stability or exact posterior calibration",
    }


def run_development(output: Path) -> dict[str, Any]:
    regime = next(r for r in REGIMES if r.name == "cluttered_staggered_crossing")
    case = make_case(regime)
    observed = _metadata(regime, case)["model_and_detections"]
    protocol = _protocol()
    if observed["detections_sha256"] != protocol["observation_sha256"]:
        raise RuntimeError("the frozen WP8 observation changed")
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different protocol")
        if result["status"] in (
            "development_screen_passed", "stopped_before_large_inference"
        ):
            if any(set(result["runs"][arm]) != {str(seed) for seed in SEEDS} for arm in ARMS):
                raise ValueError("completed checkpoint lacks a frozen arm or seed")
            return result
    else:
        result = {
            "experiment": "wp11d_unchanged_wp8_development_screen",
            "protocol": protocol,
            "baseline": {},
            "runs": {arm: {} for arm in ARMS},
            "status": "in_progress",
        }
        _save(output, result)
    for seed in SEEDS:
        # A baseline association array is not stored in JSON; deterministic
        # regeneration makes resumed paired arms use exactly the same states.
        initial, baseline_runtime = _sample_case(
            case, method="paired", n_particles=N_PARTICLES, seed=seed
        )
        if str(seed) in result["baseline"] and result["baseline"][str(seed)][
            "association_sha256"
        ] != hashlib.sha256(initial.associations.tobytes()).hexdigest():
            raise ValueError("recomputed baseline associations changed")
        result["baseline"][str(seed)] = {
            "association_sha256": hashlib.sha256(initial.associations.tobytes()).hexdigest(),
            "runtime_seconds": baseline_runtime,
            "tempering_stages": initial.n_tempering_steps,
            "summary": _summary(initial.associations, case.model),
        }
        _save(output, result)
        print(f"baseline seed {seed}: {baseline_runtime:.1f}s, "
              f"{initial.n_tempering_steps} stages", flush=True)
        for arm in ARMS:
            if str(seed) in result["runs"][arm]:
                continue
            config = BirthDeathSMCConfig(
                n_particles=N_PARTICLES, mh_sweeps=N_SWEEPS, final_mh_sweeps=0,
                association_move_probability=0, interval_move_probability=0,
                slot_swap_probability=0, blocked_slot_move_probability=0,
                paired_slot_move_probability=1, paired_slot_scope=arm,
                paired_slot_path_particles=N_PATH_PARTICLES,
            )
            rng = np.random.default_rng(seed + 10_000)
            start = time.perf_counter()
            sampled = _metropolis_birth_death_rejuvenation(
                initial.associations.copy(), initial.log_marginal_densities.copy(),
                initial.log_count_densities.copy(), initial.log_existence_priors.copy(),
                case.model, beta=1.0, n_sweeps=N_SWEEPS, config=config, rng=rng,
            )
            case.model._validate_associations(sampled[0])
            result["runs"][arm][str(seed)] = {
                "config": asdict(config),
                "state_change_rate": sampled[-1][6],
                "count_signature_change_rate": sampled[-1][9],
                "runtime_seconds": time.perf_counter() - start,
                "summary": _summary(sampled[0], case.model),
            }
            _save(output, result)
            print(f"{arm} seed {seed}: count change {sampled[-1][9]:.4f}, "
                  f"{result['runs'][arm][str(seed)]['runtime_seconds']:.1f}s", flush=True)
    result["seed_comparisons"] = {}
    result["cross_seed"] = {}
    for arm in ARMS:
        result["cross_seed"][arm] = _cross_seed(
            result["runs"][arm][str(SEEDS[0])]["summary"],
            result["runs"][arm][str(SEEDS[1])]["summary"],
        )
    for seed in SEEDS:
        pg = result["runs"]["conditional_smc"][str(seed)]
        local = result["runs"]["local"][str(seed)]
        ratio = pg["runtime_seconds"] / local["runtime_seconds"]
        result["seed_comparisons"][str(seed)] = {
            "runtime_ratio": ratio,
            "count_change_advantage": pg["count_signature_change_rate"]
            - local["count_signature_change_rate"],
            "gate_passed": ratio <= MAX_RUNTIME_RATIO and (
                pg["count_signature_change_rate"] >= local["count_signature_change_rate"]
            ),
        }
    result["status"] = (
        "development_screen_passed"
        if all(x["gate_passed"] for x in result["seed_comparisons"].values())
        else "stopped_before_large_inference"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/wp11d_wp8_development.json")
    )
    result = run_development(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
