"""Frozen selective-pair-PG development case, with no reserved holdout."""

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
from .wp5_scaling_calibration import _sample_case
from .wp7_convergence_audit import _save
from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)
from .wp11d_wp8_development import _cross_seed, _summary

REGIME = GeneralizationRegime(
    name="selective_pg_development",
    detection_probability=0.62,
    expected_clutter_per_frame=0.35,
    central_separation=0.004,
    trajectory_seed=41,
    corruption_seed=22,
    max_clutter_per_frame=None,
    truth_mask=STAGGERED_TRUTH_MASK,
)
EXPECTED_OBSERVATION_SHA256 = "9665d3a3377254df2211d251ad6ae746d8880b7501f1938ae4a2886afcfd3d37"
EXPECTED_CARDINALITIES = (2, 2, 5, 4, 1, 3, 0, 2)
SEEDS = (310, 311)
N_PARTICLES = 512
N_SWEEPS = 2
N_PATH_PARTICLES = 8
PG_FRACTION = 0.10
MAX_PAIR_RUNTIME_RATIO = 3.0
ARMS = ("sparse_pg", "local")


def _protocol() -> dict[str, Any]:
    return {
        "case": json.loads(json.dumps(asdict(REGIME))),
        "observation_sha256": EXPECTED_OBSERVATION_SHA256,
        "cardinalities": list(EXPECTED_CARDINALITIES),
        "baseline": "WP6 paired annealed SMC, 512 particles, 8 stage and 8 final sweeps",
        "seeds": list(SEEDS),
        "n_particles": N_PARTICLES,
        "post_baseline_pair_sweeps": N_SWEEPS,
        "path_particles": N_PATH_PARTICLES,
        "pg_fraction": PG_FRACTION,
        "arms": list(ARMS),
        "max_pair_runtime_ratio": MAX_PAIR_RUNTIME_RATIO,
        "gate": (
            "each seed: sparse-PG count changes >= local, observed PG fraction "
            "in [0.06,0.14], pair-update runtime <= 3x local"
        ),
        "interpretation": "new development observation, not a holdout or exact eight-frame calibration",
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
    if arm == "sparse_pg":
        return BirthDeathSMCConfig(
            **common, paired_slot_scope="conditional_smc",
            paired_slot_path_particles=N_PATH_PARTICLES,
            paired_slot_pg_fraction=PG_FRACTION,
        )
    raise ValueError("unknown development arm")


def run_development(output: Path) -> dict[str, Any]:
    case = make_case(REGIME)
    metadata = _metadata(REGIME, case)["model_and_detections"]
    protocol = _protocol()
    if (
        metadata["detections_sha256"] != EXPECTED_OBSERVATION_SHA256
        or tuple(metadata["detection_cardinalities"]) != EXPECTED_CARDINALITIES
    ):
        raise RuntimeError("frozen development observation changed")
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different protocol")
        if result["status"] in ("development_screen_passed", "stopped_before_holdout"):
            if any(set(result["runs"][arm]) != {str(seed) for seed in SEEDS} for arm in ARMS):
                raise ValueError("completed checkpoint lacks an arm or seed")
            return result
    else:
        result = {
            "experiment": "wp11e_selective_pg_new_synthetic_development",
            "protocol": protocol,
            "baseline": {},
            "runs": {arm: {} for arm in ARMS},
            "status": "in_progress",
        }
        _save(output, result)
    for seed in SEEDS:
        sampled, baseline_runtime = _sample_case(
            case, method="paired", n_particles=N_PARTICLES, seed=seed
        )
        digest = hashlib.sha256(sampled.associations.tobytes()).hexdigest()
        if (str(seed) in result["baseline"]
                and result["baseline"][str(seed)]["association_sha256"] != digest):
            raise ValueError("recomputed baseline associations changed")
        result["baseline"][str(seed)] = {
            "association_sha256": digest,
            "tempering_stages": sampled.n_tempering_steps,
            "runtime_seconds": baseline_runtime,
            "summary": _summary(sampled.associations, case.model),
        }
        _save(output, result)
        print(f"baseline seed {seed}: {baseline_runtime:.1f}s, "
              f"{sampled.n_tempering_steps} stages", flush=True)
        for arm in ARMS:
            if str(seed) in result["runs"][arm]:
                continue
            config = _controls(arm)
            start = time.perf_counter()
            result_states = _metropolis_birth_death_rejuvenation(
                sampled.associations.copy(), sampled.log_marginal_densities.copy(),
                sampled.log_count_densities.copy(), sampled.log_existence_priors.copy(),
                case.model, beta=1, n_sweeps=N_SWEEPS, config=config,
                rng=np.random.default_rng(seed + 10_000),
            )
            case.model._validate_associations(result_states[0])
            result["runs"][arm][str(seed)] = {
                "config": asdict(config),
                "state_change_rate": result_states[-1][6],
                "count_signature_change_rate": result_states[-1][9],
                "actual_pg_fraction": result_states[-1][10],
                "pg_count_signature_change_rate": result_states[-1][11],
                "runtime_seconds": time.perf_counter() - start,
                "summary": _summary(result_states[0], case.model),
            }
            _save(output, result)
            print(f"{arm} seed {seed}: count change {result_states[-1][9]:.4f}, "
                  f"PG fraction {result_states[-1][10]:.3f}, "
                  f"{result['runs'][arm][str(seed)]['runtime_seconds']:.1f}s", flush=True)
    result["cross_seed"] = {
        arm: _cross_seed(
            result["runs"][arm][str(SEEDS[0])]["summary"],
            result["runs"][arm][str(SEEDS[1])]["summary"],
        ) for arm in ARMS
    }
    result["seed_comparisons"] = {}
    for seed in SEEDS:
        pg = result["runs"]["sparse_pg"][str(seed)]
        local = result["runs"]["local"][str(seed)]
        ratio = pg["runtime_seconds"] / local["runtime_seconds"]
        result["seed_comparisons"][str(seed)] = {
            "runtime_ratio": ratio,
            "count_change_advantage": (
                pg["count_signature_change_rate"] - local["count_signature_change_rate"]
            ),
            "gate_passed": (
                pg["count_signature_change_rate"] >= local["count_signature_change_rate"]
                and 0.06 <= pg["actual_pg_fraction"] <= 0.14
                and ratio <= MAX_PAIR_RUNTIME_RATIO
            ),
        }
    result["status"] = (
        "development_screen_passed"
        if all(x["gate_passed"] for x in result["seed_comparisons"].values())
        else "stopped_before_holdout"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/wp11e_selective_pg_development.json")
    )
    result = run_development(parser.parse_args().output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
