"""Frozen matched test of WP11ag on the independent balanced WP11ah joint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .existence_smc import sample_slot_existence_posterior
from .wp7_convergence_audit import _save
from .wp11aa_tempered_suffix_pg import (
    _controls,
    _decision,
    _joint,
    _metrics,
)
from .wp11aa_tempered_suffix_pg import (
    _protocol as _old_protocol,
)
from .wp11ag_global_slot_comparison import (
    MAX_CPU_RATIO,
    MAX_MEDIAN_CPU_RATIO,
    N_PARTICLES,
    _candidate,
)
from .wp11ah_balanced_crowded_reference import (
    LEDGER_SHA256,
    _checked_case,
    _ledger,
    _ordered,
)
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11w_balanced_exact_reference import _accepted, _masses

SEEDS = (1700, 1701, 1702, 1703)
CPU_CAP = 750
SELECTED_INDEX = 27
SCREEN_SHA256 = "866c9649949c7609349b80e54dcc04184ae3ddecb0f5573caba4f02141a582e2"
REFERENCE_SHA256 = "d0bb9e6496ed33cb784c05f8fd110a7a7b23b9b5ec2e84b98f1c15d91c03a6c2"
WP11AG_SHA256 = "727c207b3e9868034cf7bd227fb4a92ca0718d2ee05c242f30183c3a8c423bb4"


def _protocol() -> dict[str, Any]:
    return {
        "observation": "first predeclared balanced WP11ah screen, index 27",
        "geometry_sha256": LEDGER_SHA256,
        "screen_sha256": SCREEN_SHA256,
        "exact_reference_sha256": REFERENCE_SHA256,
        "prior_wp11ag_result_sha256": WP11AG_SHA256,
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "control_configuration_seed_1700": asdict(_controls(1700, "control")),
        "candidate_configuration_seed_1700": asdict(_candidate(1700)),
        "move": "unchanged WP11ag exact whole-slot conditional Gibbs: 20% of "
                "paired proposals at beta >=.65; controls matched on fresh seeds",
        "accuracy_and_replication": dict(_old_protocol()["decoy_gate"]),
        "cost": {"each_candidate_over_control_at_most": MAX_CPU_RATIO,
                 "candidate_median_over_control_median_at_most": MAX_MEDIAN_CPU_RATIO,
                 "per_run_process_cpu_cap": CPU_CAP},
        "thread_env": {key: "1" for key in THREAD_ENV},
        "decision": "complete all four matched control/candidate pairs even if a partial "
                    "result is poor; any cap or input mismatch stops comparison",
        "interpretation": "independent balanced 17-detection crowded development case; "
                          "high-alive margin only .0006983; exposed 19-detection "
                          "sentinel passed in pinned WP11ag result, no broad calibration claim",
        "holdout": "closed; no untouched WP8 inference",
    }


def _pinned(path: Path, sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"frozen input changed: {path}")
    return json.loads(raw)


def _inputs(ledger_path: Path, screen_path: Path, reference_path: Path,
            previous_path: Path) -> tuple[Any, np.ndarray, float]:
    ledger = _ledger(ledger_path)
    selected = _ordered(ledger)[0]
    screen = _pinned(screen_path, SCREEN_SHA256)
    reference = _pinned(reference_path, REFERENCE_SHA256)
    previous = _pinned(previous_path, WP11AG_SHA256)
    if (selected["index"] != SELECTED_INDEX
            or screen["status"] != "balance_accepted_validate"
            or screen["ledger_sha256"] != LEDGER_SHA256
            or screen["index"] != SELECTED_INDEX
            or screen["detection_sha256"] != selected["detection_sha256"]
            or reference["status"] != "validated_reference_no_sampler"
            or reference["ledger_sha256"] != LEDGER_SHA256
            or reference["index"] != SELECTED_INDEX
            or reference["selected_detection_sha256"] != selected["detection_sha256"]
            or set(reference["checks"]) != {"0.0", "0.5", "1.0"}
            or previous["status"] != (
                "exposed_and_independent_transfer_gates_passed_needs_new_crowded_reference"
            ) or set(previous["cases"]) != {"primary_19", "transfer_17"}
            or not all(x["comparison"]["passed"] for x in previous["cases"].values())):
        raise ValueError("frozen reference or previous success differs")
    case = _checked_case(selected)
    endpoint = reference["checks"]["1.0"]["direct"]
    exact = np.asarray(endpoint["joint"])
    cards = case.model.sequence.cardinalities
    if (exact.shape != (33, cards[5] + 1, cards[7] + 1)
            or not np.isclose(exact.sum(), 1.0, atol=1e-12)
            or not _accepted(_masses(exact))
            or any(abs(a - b) > 1e-7 for a, b in zip(
                _masses(exact)["alive_bands"], screen["masses"]["alive_bands"],
                strict=True,
            ))):
        raise ValueError("selected absolute joint differs or lacks frozen balance")
    return case, exact, float(endpoint["log_partition"])


def _path(run_dir: Path, arm: str, seed: int) -> Path:
    return run_dir / f"balanced_17_{arm}_{seed}.json"


def run_one(arm: str, seed: int, run_dir: Path, ledger: Path, screen: Path,
            reference: Path, previous: Path) -> dict[str, Any]:
    if arm not in ("control", "candidate") or seed not in SEEDS:
        raise ValueError("run outside frozen arms and seeds")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    case, exact, logz = _inputs(ledger, screen, reference, previous)
    config = _candidate(seed) if arm == "candidate" else _controls(seed, "control")
    path = _path(run_dir, arm, seed)
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["protocol"] != _protocol() or saved["arm"] != arm
                or saved["seed"] != seed or saved["config"] != asdict(config)):
            raise ValueError("run checkpoint differs from frozen protocol")
        return saved
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen cap")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP + 5))
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std, coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior, config=config,
    )
    joint = _joint(sampled.associations, tuple(case.model.sequence.cardinalities))
    result = {
        "experiment": "wp11ai_balanced_crowded_frozen_run",
        "protocol": _protocol(), "arm": arm, "seed": seed,
        "config": asdict(config),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - logz,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact),
        "stages": [stage.to_dict() for stage in sampled.stages],
    }
    _save(path, result)
    print(f"balanced_17 {arm} {seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"joint TV {result['errors']['full_joint_tv']:.5f}", flush=True)
    return result


def _validated_run(path: Path, arm: str, seed: int,
                   exact: np.ndarray) -> dict[str, Any]:
    saved = json.loads(path.read_text())
    config = _candidate(seed) if arm == "candidate" else _controls(seed, "control")
    if (saved["protocol"] != _protocol() or saved["arm"] != arm
            or saved["seed"] != seed or saved["config"] != asdict(config)
            or saved["cpu_seconds"] > CPU_CAP
            or saved["n_tempering_steps"] != len(saved["stages"])
            or not np.isclose(saved["stages"][-1]["beta"], 1.0, atol=1e-12)):
        raise ValueError("run differs from frozen protocol")
    joint = np.asarray(saved["joint_alive_by_sixth_by_last_clutter"])
    if (joint.shape != exact.shape or not np.isclose(joint.sum(), 1, atol=1e-12)
            or not np.allclose(joint * N_PARTICLES,
                               np.rint(joint * N_PARTICLES), atol=1e-9)):
        raise ValueError("run joint is invalid")
    for key, actual in _metrics(joint, exact).items():
        if not np.allclose(actual, saved["errors"][key], atol=1e-12):
            raise ValueError(f"run metric {key} differs from pinned reference")
    return saved


def aggregate(run_dir: Path, ledger: Path, screen: Path, reference: Path,
              previous: Path, output: Path) -> dict[str, Any]:
    case, exact, _ = _inputs(ledger, screen, reference, previous)
    controls, candidates, rows = [], [], {}
    for seed in SEEDS:
        left = _validated_run(_path(run_dir, "control", seed), "control", seed, exact)
        right = _validated_run(_path(run_dir, "candidate", seed), "candidate", seed,
                               exact)
        control = {"cpu_seconds": left["cpu_seconds"], "errors": left["errors"]}
        candidate = {"cpu_seconds": right["cpu_seconds"], "errors": right["errors"]}
        controls.append(control)
        candidates.append(candidate)
        rows[str(seed)] = {"control": control, "candidate": {
            **candidate, "n_tempering_steps": right["n_tempering_steps"],
            "log_evidence_error": right["log_evidence_error"],
            "joint_alive_by_sixth_by_last_clutter": right[
                "joint_alive_by_sixth_by_last_clutter"
            ],
        }}
    comparison = _decision("decoy_17", controls, candidates)
    comparison["per_seed_modes"] = dict(zip(
        map(str, SEEDS), comparison["per_seed_modes"].values(), strict=True
    ))
    control_cpu = median(row["cpu_seconds"] for row in controls)
    candidate_cpu = median(row["cpu_seconds"] for row in candidates)
    comparison["gates"]["median_cost"] = (
        candidate_cpu <= MAX_MEDIAN_CPU_RATIO * control_cpu
    )
    comparison["median_cpu_seconds"] = {"control": control_cpu,
                                        "candidate": candidate_cpu}
    comparison["passed"] = all(comparison["gates"].values())
    result = {
        "experiment": "wp11ai_balanced_crowded_comparison",
        "protocol": _protocol(), "cardinalities": list(case.model.sequence.cardinalities),
        "runs": rows, "comparison": comparison,
        "status": ("independent_balanced_crowded_gate_passed_holdout_closed"
                   if comparison["passed"] else "balanced_crowded_gate_failed_stop"),
    }
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "run", "aggregate"), required=True)
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11ah_geometry_ledger.json"))
    parser.add_argument("--screen", type=Path,
                        default=Path("results/wp11ah_screens/27.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/wp11ah_crowded_exact_reference.json"))
    parser.add_argument("--previous", type=Path,
                        default=Path("results/wp11ag_global_slot_comparison.json"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/wp11ai_balanced_crowded_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ai_balanced_crowded_comparison.json"))
    args = parser.parse_args()
    inputs = (args.ledger, args.screen, args.reference, args.previous)
    if args.case == "protocol":
        _inputs(*inputs)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "run":
        if args.arm is None or args.seed is None:
            parser.error("run requires arm and seed")
        run_one(args.arm, args.seed, args.run_dir, *inputs)
    else:
        if args.arm is not None or args.seed is not None:
            parser.error("aggregate takes no arm or seed")
        print(aggregate(args.run_dir, *inputs, args.output)["status"])


if __name__ == "__main__":
    main()
