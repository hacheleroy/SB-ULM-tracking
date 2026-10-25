"""Oracle-backed early-stop pilot for full-history two-slot rejuvenation.

The earlier WP11b eight-frame realization is not reused as a validation gate.
This pilot starts at the exactly enumerable posterior, measures actual
count-signature transitions, and blocks a new held-out case if the full-history
kernel cannot move between count modes at the predeclared small-case rate.
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
from scipy.special import logsumexp

from .data import DetectionSequence
from .existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from .slot_existence import ABSENT, SlotExistenceMaternModel, enumerate_slot_existence_support
from .wp7_convergence_audit import _save

SEEDS = (74, 75, 76)
N_PARTICLES = 512
N_SWEEPS = 2
MINIMUM_MEAN_COUNT_CHANGE_RATE = 0.01
ARMS = ("full_sequence", "local")


def _pilot_model() -> SlotExistenceMaternModel:
    sequence = DetectionSequence(
        times=np.arange(3, dtype=float) * 0.001,
        detections=(
            np.asarray([[-0.008, 0.0]]),
            np.empty((0, 2)),
            np.asarray([[0.008, 0.0], [-0.007, 0.002]]),
        ),
    )
    return SlotExistenceMaternModel(
        sequence,
        n_slots=2,
        localization_std=0.005,
        coordinate_std=(0.02, 0.01),
        length_scale=0.003,
        detection_probability=0.7,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )


def _protocol(model: SlotExistenceMaternModel) -> dict[str, Any]:
    digest = hashlib.sha256()
    for frame in model.sequence.detections:
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(frame, dtype="<f8").tobytes())
    return {
        "case": "exact two-slot three-frame miss/clutter posterior",
        "observation_sha256": digest.hexdigest(),
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "sweeps": N_SWEEPS,
        "beta": 1.0,
        "arms": list(ARMS),
        "candidates_per_move": 2,
        "minimum_mean_count_signature_change_rate": MINIMUM_MEAN_COUNT_CHANGE_RATE,
        "gate": "full-history mean across seeds must be >= 0.01; otherwise do not open a new holdout",
    }


def run_pilot(output: Path) -> dict[str, Any]:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    log_scores = model.log_joint_density(support)
    exact = np.exp(log_scores - logsumexp(log_scores))
    base_log = model.log_base_probability(support)
    base = np.exp(base_log - logsumexp(base_log))
    order = np.argsort(exact)[::-1]
    credible_size = int(np.searchsorted(np.cumsum(exact[order]), 0.95) + 1)
    overlap = {
        "total_variation": 0.5 * float(np.sum(np.abs(base - exact))),
        "posterior_credible_set_95_size": credible_size,
        "base_mass_on_posterior_credible_set_95": float(
            np.sum(base[order[:credible_size]])
        ),
    }
    protocol = _protocol(model)
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["exact_support_size"] != len(support):
            raise ValueError("existing checkpoint has a different frozen pilot protocol")
        if "base_posterior_overlap" in result and result["base_posterior_overlap"] != overlap:
            raise ValueError("the exact pilot distribution changed")
    else:
        result = {
            "experiment": "wp11c_full_history_pair_small_case_pilot",
            "protocol": protocol,
            "exact_support_size": len(support),
            "base_posterior_overlap": overlap,
            "runs": {arm: {} for arm in ARMS},
            "status": "in_progress",
        }
        _save(output, result)
    result["base_posterior_overlap"] = overlap
    expected_alive = float(np.sum(
        exact * np.count_nonzero(support != ABSENT, axis=(1, 2))
    ))
    for seed in SEEDS:
        initial_rng = np.random.default_rng(seed)
        initial = support[initial_rng.choice(len(support), size=N_PARTICLES, p=exact)]
        for arm in ARMS:
            if str(seed) in result["runs"][arm]:
                continue
            rng = np.random.default_rng(seed + 10_000)
            config = BirthDeathSMCConfig(
                n_particles=N_PARTICLES,
                mh_sweeps=N_SWEEPS,
                final_mh_sweeps=0,
                association_move_probability=0.0,
                interval_move_probability=0.0,
                slot_swap_probability=0.0,
                blocked_slot_move_probability=0.0,
                paired_slot_move_probability=1.0,
                paired_slot_candidates=2,
                paired_slot_scope=arm,
                seed=seed,
            )
            current = initial.copy()
            start = time.perf_counter()
            sampled = _metropolis_birth_death_rejuvenation(
                current,
                model.log_marginal_density(current),
                model.log_count_density(current, include_poisson_void=False),
                model.log_existence_prior(current),
                model,
                beta=1.0,
                n_sweeps=N_SWEEPS,
                config=config,
                rng=rng,
            )
            model._validate_associations(sampled[0])
            result["runs"][arm][str(seed)] = {
                "config": asdict(config),
                "acceptance_rate": sampled[-1][5],
                "state_change_rate": sampled[-1][6],
                "count_signature_change_rate": sampled[-1][9],
                "mean_alive_slot_frames": float(np.mean(
                    np.count_nonzero(sampled[0] != ABSENT, axis=(1, 2))
                )),
                "initial_mean_alive_slot_frames": float(np.mean(
                    np.count_nonzero(initial != ABSENT, axis=(1, 2))
                )),
                "exact_mean_alive_slot_frames": expected_alive,
                "runtime_seconds": time.perf_counter() - start,
            }
            _save(output, result)
            print(
                f"{arm} seed {seed}: count-mode changes "
                f"{sampled[-1][9]:.4f}, acceptance {sampled[-1][5]:.4f}",
                flush=True,
            )
    full_rates = [
        result["runs"]["full_sequence"][str(seed)]["count_signature_change_rate"]
        for seed in SEEDS
    ]
    result["mean_full_history_count_change_rate"] = float(np.mean(full_rates))
    result["status"] = (
        "small_case_gate_passed"
        if result["mean_full_history_count_change_rate"] >= MINIMUM_MEAN_COUNT_CHANGE_RATE
        else "stopped_before_fresh_holdout"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/wp11c_full_history_pilot.json")
    )
    args = parser.parse_args()
    result = run_pilot(args.output)
    print(
        f"{result['status']}: mean count-mode change "
        f"{result['mean_full_history_count_change_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
