"""Frozen exact-target and early-path movement screen for four-path PGAS."""

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
from .slot_existence import ABSENT, enumerate_slot_existence_support
from .wp7_convergence_audit import _save
from .wp11c_full_history_pilot import _pilot_model
from .wp11d_conditional_pair_pilot import _count_signatures

SEEDS = (620, 621, 622)
ARMS = ("pg", "pgas")
N_PARTICLES = 512
N_SWEEPS = 2
N_PATH_PARTICLES = 4
OBSERVATION_SHA256 = "c645fab464abe15579b843a35081078e62983d6fdfd014c7692a88a368dace72"
EXACT_SUPPORT_SIZE = 269
MAX_COUNT_TV_Q99 = 0.08892790897077366
MAX_CPU_RATIO = 2.0


def _controls(arm: str) -> BirthDeathSMCConfig:
    if arm not in ARMS:
        raise ValueError("unknown oracle arm")
    return BirthDeathSMCConfig(
        n_particles=N_PARTICLES, mh_sweeps=N_SWEEPS, final_mh_sweeps=0,
        association_move_probability=0, interval_move_probability=0,
        slot_swap_probability=0, blocked_slot_move_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=1, paired_slot_path_particles=N_PATH_PARTICLES,
        paired_slot_pg_ancestor_sampling=(arm == "pgas"),
    )


def _protocol() -> dict[str, Any]:
    return {
        "observation_sha256": OBSERVATION_SHA256,
        "exact_support_size": EXACT_SUPPORT_SIZE,
        "seeds": list(SEEDS), "arms": list(ARMS),
        "n_particles": N_PARTICLES, "sweeps": N_SWEEPS,
        "internal_paths": N_PATH_PARTICLES,
        "beta": 1.0, "pg_fraction": 1.0,
        "ancestor_proposal": "one forced uniform-excluding-current MH step",
        "ancestor_weights": "full exact future suffix, never truncated",
        "finite_sample_count_tv_q99": MAX_COUNT_TV_Q99,
        "maximum_pgas_pg_cpu_ratio_each_seed": MAX_CPU_RATIO,
        "arm_order_by_seed": {"620": ["pg", "pgas"],
                              "621": ["pgas", "pg"],
                              "622": ["pg", "pgas"]},
        "gate": (
            "all six count TVs <= frozen 99% exact multinomial envelope; "
            "PGAS mean first-frame association-change rate > PG; "
            "PGAS/PG process CPU ratio <=2 on every seed"
        ),
        "interpretation": "exact posterior pilot and path movement, not eight-frame stability",
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
            raise ValueError("oracle checkpoint has a different frozen protocol")
        if result["status"] != "in_progress":
            return result
    else:
        result = {"experiment": "wp11j_pgas_oracle", "protocol": protocol,
                  "runs": {arm: {} for arm in ARMS}, "status": "in_progress"}
        _save(output, result)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        initial = support[rng.choice(len(support), size=N_PARTICLES, p=exact)]
        for arm in protocol["arm_order_by_seed"][str(seed)]:
            if str(seed) in result["runs"][arm]:
                continue
            started = time.process_time()
            sampled = _metropolis_birth_death_rejuvenation(
                initial.copy(), model.log_marginal_density(initial),
                model.log_count_density(initial, include_poisson_void=False),
                model.log_existence_prior(initial), model,
                beta=1, n_sweeps=N_SWEEPS, config=_controls(arm),
                rng=np.random.default_rng(seed + 10_000),
            )
            cpu_seconds = time.process_time() - started
            model._validate_associations(sampled[0])
            indices = np.asarray([lookup[state.tobytes()] for state in sampled[0]])
            histogram = np.bincount(inverse[indices], minlength=len(exact_count)) / N_PARTICLES
            count_tv = float(np.abs(histogram - exact_count).sum() / 2)
            early_changed = np.any(sampled[0][:, 0] != initial[:, 0], axis=1)
            first_count_changed = (
                (np.count_nonzero(sampled[0][:, 0] != ABSENT, axis=1)
                 != np.count_nonzero(initial[:, 0] != ABSENT, axis=1))
                | (np.count_nonzero(sampled[0][:, 0] >= 0, axis=1)
                   != np.count_nonzero(initial[:, 0] >= 0, axis=1))
            )
            result["runs"][arm][str(seed)] = {
                "cpu_seconds": cpu_seconds,
                "first_frame_state_change_rate": float(np.mean(early_changed)),
                "first_frame_count_change_rate": float(np.mean(first_count_changed)),
                "total_count_change_rate": sampled[-1][9],
                "count_tv_against_exact": count_tv,
                "within_exact_envelope": bool(count_tv <= MAX_COUNT_TV_Q99),
            }
            _save(output, result)
            print(f"seed {seed} {arm}: TV={count_tv:.4f}, "
                  f"early state change={np.mean(early_changed):.3f}", flush=True)
    mean_changes = {arm: float(np.mean([
        result["runs"][arm][str(seed)]["first_frame_state_change_rate"] for seed in SEEDS
    ])) for arm in ARMS}
    ratios = {str(seed): (result["runs"]["pgas"][str(seed)]["cpu_seconds"]
                          / result["runs"]["pg"][str(seed)]["cpu_seconds"])
              for seed in SEEDS}
    result["mean_first_frame_state_change_rate"] = mean_changes
    result["pgas_pg_cpu_ratios"] = ratios
    result["status"] = (
        "exact_oracle_passed" if (
            all(run["within_exact_envelope"] for arm in result["runs"].values()
                for run in arm.values())
            and mean_changes["pgas"] > mean_changes["pg"]
            and all(ratio <= MAX_CPU_RATIO for ratio in ratios.values())
        ) else "stopped_before_new_development_case"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp11j_pgas_oracle.json"))
    result = run_oracle(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
