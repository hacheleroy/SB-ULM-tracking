"""Frozen exact-oracle screen for a beta-dependent four-path PG allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .existence_smc import (
    BirthDeathSMCConfig,
    _metropolis_birth_death_rejuvenation,
)
from .slot_existence import enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11c_full_history_pilot import _pilot_model
from .wp11d_conditional_pair_pilot import _count_signatures

SEEDS = (610, 611, 612)
ARMS = ("constant", "late")
N_PARTICLES = 512
N_SWEEPS = 2
N_PATH_PARTICLES = 4
EARLY_FRACTION = 0.025
LATE_FRACTION = 0.25
CONSTANT_FRACTION = 0.10
BETA_SWITCH = 0.75
OBSERVATION_SHA256 = "c645fab464abe15579b843a35081078e62983d6fdfd014c7692a88a368dace72"
EXACT_SUPPORT_SIZE = 269
MAX_COUNT_TV_Q99 = 0.08892790897077366


def _controls(arm: str) -> BirthDeathSMCConfig:
    common = {
        "n_particles": N_PARTICLES, "mh_sweeps": N_SWEEPS, "final_mh_sweeps": 0,
        "association_move_probability": 0, "interval_move_probability": 0,
        "slot_swap_probability": 0, "blocked_slot_move_probability": 0,
        "paired_slot_move_probability": 1, "paired_slot_scope": "conditional_smc",
        "paired_slot_path_particles": N_PATH_PARTICLES,
    }
    if arm == "constant":
        return BirthDeathSMCConfig(**common, paired_slot_pg_fraction=CONSTANT_FRACTION)
    if arm == "late":
        return BirthDeathSMCConfig(
            **common, paired_slot_pg_fraction=LATE_FRACTION,
            paired_slot_pg_beta_switch=BETA_SWITCH,
            paired_slot_pg_early_fraction=EARLY_FRACTION,
        )
    raise ValueError("unknown oracle arm")


def _protocol() -> dict[str, Any]:
    return {
        "observation_sha256": OBSERVATION_SHA256,
        "exact_support_size": EXACT_SUPPORT_SIZE,
        "seeds": list(SEEDS), "arms": list(ARMS),
        "n_particles": N_PARTICLES, "sweeps": N_SWEEPS,
        "internal_paths": N_PATH_PARTICLES,
        "control_pg_fraction": CONSTANT_FRACTION,
        "candidate_early_fraction": EARLY_FRACTION,
        "candidate_late_fraction": LATE_FRACTION,
        "beta_switch": BETA_SWITCH,
        "finite_sample_count_tv_q99": MAX_COUNT_TV_Q99,
        "arm_order_by_seed": {"610": ["constant", "late"],
                              "611": ["late", "constant"],
                              "612": ["constant", "late"]},
        "gate": (
            "all six count TVs <= frozen 99% exact multinomial envelope; "
            "candidate mean count-change rate > constant at beta=1"
        ),
        "interpretation": "exact-target and movement pilot, not an eight-frame stability result",
    }


def run_oracle(output: Path) -> dict[str, Any]:
    model = _pilot_model()
    digest = hashlib.sha256()
    for frame in model.sequence.detections:
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(frame, dtype="<f8").tobytes())
    if digest.hexdigest() != OBSERVATION_SHA256:
        raise ValueError("small oracle observation changed")
    support, _ = enumerate_slot_existence_support(model)
    if len(support) != EXACT_SUPPORT_SIZE:
        raise ValueError("small oracle support changed")
    log_target = model.log_joint_density(support)
    exact = np.exp(log_target - logsumexp(log_target))
    _, inverse = np.unique(_count_signatures(support), axis=0, return_inverse=True)
    exact_count = np.bincount(inverse, weights=exact)
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    protocol = _protocol()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("oracle checkpoint belongs to a different frozen protocol")
        if result["status"] != "in_progress":
            return result
    else:
        result = {
            "experiment": "wp11i_late_pg_oracle", "protocol": protocol,
            "runs": {arm: {} for arm in ARMS}, "status": "in_progress",
        }
        _save(output, result)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        initial = support[rng.choice(len(support), size=N_PARTICLES, p=exact)]
        for arm in protocol["arm_order_by_seed"][str(seed)]:
            if str(seed) in result["runs"][arm]:
                continue
            config = _controls(arm)
            started = time.process_time()
            sampled = _metropolis_birth_death_rejuvenation(
                initial.copy(), model.log_marginal_density(initial),
                model.log_count_density(initial, include_poisson_void=False),
                model.log_existence_prior(initial), model,
                beta=1.0, n_sweeps=N_SWEEPS, config=config,
                rng=np.random.default_rng(seed + 10_000),
            )
            cpu_seconds = time.process_time() - started
            model._validate_associations(sampled[0])
            indices = np.asarray([lookup[state.tobytes()] for state in sampled[0]])
            histogram = np.bincount(inverse[indices], minlength=len(exact_count)) / N_PARTICLES
            count_tv = float(np.abs(histogram - exact_count).sum() / 2)
            result["runs"][arm][str(seed)] = {
                "cpu_seconds": cpu_seconds,
                "count_change_rate": sampled[-1][9],
                "actual_pg_fraction": sampled[-1][10],
                "count_tv_against_exact": count_tv,
                "within_exact_envelope": count_tv <= MAX_COUNT_TV_Q99,
            }
            _save(output, result)
            print(f"seed {seed} {arm}: TV={count_tv:.4f}, "
                  f"count change={sampled[-1][9]:.3f}", flush=True)
    mean_changes = {arm: float(np.mean([
        result["runs"][arm][str(seed)]["count_change_rate"] for seed in SEEDS
    ])) for arm in ARMS}
    result["mean_count_change_rate"] = mean_changes
    result["status"] = (
        "exact_oracle_passed" if (
            all(run["within_exact_envelope"] for arm in result["runs"].values()
                for run in arm.values())
            and mean_changes["late"] > mean_changes["constant"]
        ) else "stopped_before_new_development_case"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp11i_late_pg_oracle.json"))
    result = run_oracle(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
