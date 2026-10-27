"""Frozen independent comparison for a selective exact whole-slot Gibbs move."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .existence_smc import BirthDeathSMCConfig, sample_slot_existence_posterior
from .wp7_convergence_audit import _save
from .wp11aa_tempered_suffix_pg import (
    PRIMARY_SEEDS,
    _controls,
    _joint,
    _metrics,
    _previous,
)
from .wp11aa_tempered_suffix_pg import _case as _primary_or_decoy
from .wp11aa_tempered_suffix_pg import _decision as _previous_decision
from .wp11aa_tempered_suffix_pg import _protocol as _previous_protocol
from .wp11af_geometry_transfer_reference import (
    LEDGER_SHA256,
    _selected_case,
)
from .wp11f_interleaved_cost import THREAD_ENV

SWITCH_BETA = .65
WHOLE_SLOT_FRACTION = .20
TRANSFER_SEEDS = (1600, 1601, 1602, 1603)
N_PARTICLES = 2_048
CPU_CAP = 750
MAX_CPU_RATIO = 1.50
MAX_MEDIAN_CPU_RATIO = 1.25
WP11AF_SHA256 = "cd6f20f821a2689cda5a1a0aceb19fa9db7c6232af0e8e05b6262e312469f87b"
WP11AB_SHA256 = "ab05dec063aa142e0504cf4a6c23a3abfb0db58730d569556f1e288251355c4c"


@dataclass(frozen=True)
class WholeSlotSMCConfig(BirthDeathSMCConfig):
    """Opt in without changing the historical control configuration."""

    paired_slot_whole_slot_beta_switch: float = SWITCH_BETA
    paired_slot_whole_slot_fraction: float = WHOLE_SLOT_FRACTION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.paired_slot_scope != "conditional_smc":
            raise ValueError("whole-slot Gibbs requires conditional SMC paired scope")
        if (not np.isfinite(self.paired_slot_whole_slot_beta_switch)
                or not 0 < self.paired_slot_whole_slot_beta_switch <= 1):
            raise ValueError("whole-slot beta switch must lie in (0, 1]")
        early_pg = (self.paired_slot_pg_early_fraction
                    if self.paired_slot_pg_early_fraction is not None
                    else self.paired_slot_pg_fraction)
        if (not np.isfinite(self.paired_slot_whole_slot_fraction)
                or not 0 < self.paired_slot_whole_slot_fraction <= 1
                or max(early_pg, self.paired_slot_pg_fraction)
                + self.paired_slot_suffix_pg_final_fraction
                + self.paired_slot_whole_slot_fraction > 1):
            raise ValueError("whole-slot paired mixture fractions are invalid")


def _candidate(seed: int) -> WholeSlotSMCConfig:
    return WholeSlotSMCConfig(**asdict(_controls(seed, "control")))


def _protocol() -> dict[str, Any]:
    prior = _previous_protocol()
    primary_gate = dict(prior["primary_gate"])
    transfer_gate = dict(prior["decoy_gate"])
    return {
        "references": {"primary_19": prior["references"]["primary_19"],
                       "transfer_17": WP11AF_SHA256},
        "transfer_geometry_sha256": LEDGER_SHA256,
        "archived_primary_control_sha256": prior["controls"]["primary_19"],
        "wp11ab_result_sha256": WP11AB_SHA256,
        "seeds": {"primary_19": list(PRIMARY_SEEDS),
                  "transfer_17": list(TRANSFER_SEEDS)},
        "candidate_configuration_seed_1300": asdict(_candidate(1300)),
        "move": "At beta >= .65, replace 20% of paired proposals by exact whole-slot "
                "conditional Gibbs redraws, with 10% full-path four-particle PG "
                "and 70% local paired retained. Below switch, bitwise control-equivalent. "
                "Other moves, particle count and sweeps unchanged.",
        "n_particles": N_PARTICLES,
        "per_run_process_cpu_cap": CPU_CAP,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "accuracy_and_replication": {
            "primary_19": primary_gate,
            "transfer_17": transfer_gate,
            "interpretation": "same absolute and replication limits as WP11aa, "
                              "but transfer observation was independently selected",
        },
        "cost": {"each_candidate_over_control_at_most": MAX_CPU_RATIO,
                 "candidate_median_over_control_median_at_most": MAX_MEDIAN_CPU_RATIO},
        "decision": "all six primary candidates and four transfer control/candidate "
                    "pairs; both complete gates required; no early stop on poor outcomes",
        "interpretation": "development and independent transfer only; transfer case is not "
                          "a balanced crowded-mode reference or untouched holdout",
        "holdout": "closed; no 8192-particle or untouched WP8 inference",
    }


def _pinned(path: Path, expected: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen input changed: {path}")
    return json.loads(raw)


def _inputs(primary_reference: Path, transfer_reference: Path, ledger: Path,
            wp11ab: Path, primary_control: Path, decoy_control: Path,
            audit: Path) -> tuple[dict[str, Any], Any, np.ndarray, float]:
    previous = _previous(primary_control, decoy_control, audit)
    prior = _pinned(wp11ab, WP11AB_SHA256)
    if (prior["status"] != "two_exposed_case_gate_failed_stop"
            or set(prior["cases"]["primary_19"]["runs"])
            != set(map(str, PRIMARY_SEEDS))):
        raise ValueError("WP11ab failure checkpoint differs")
    reference = _pinned(transfer_reference, WP11AF_SHA256)
    case, selected = _selected_case(ledger)
    if (reference["status"] != "validated_reference_no_sampler"
            or reference["ledger_sha256"] != LEDGER_SHA256
            or reference["selected_index"] != selected["index"]
            or reference["selected_detection_sha256"] != selected["detection_sha256"]
            or set(reference["checks"]) != {"0.0", "0.5", "1.0"}):
        raise ValueError("independent transfer exact reference differs")
    endpoint = reference["checks"]["1.0"]["direct"]
    exact = np.asarray(endpoint["joint"])
    if (exact.shape != (33, case.model.sequence.cardinalities[5] + 1,
                       case.model.sequence.cardinalities[7] + 1)
            or not np.isclose(exact.sum(), 1.0, atol=1e-12)):
        raise ValueError("independent transfer joint is invalid")
    _primary_or_decoy("primary_19", primary_reference, transfer_reference, ledger)
    return previous, case, exact, float(endpoint["log_partition"])


def _run_path(run_dir: Path, name: str, arm: str, seed: int) -> Path:
    return run_dir / f"{name}_{arm}_{seed}.json"


def run_one(name: str, arm: str, seed: int, run_dir: Path,
            primary_reference: Path, transfer_reference: Path, ledger: Path,
            wp11ab: Path, primary_control: Path, decoy_control: Path,
            audit: Path) -> dict[str, Any]:
    seeds = PRIMARY_SEEDS if name == "primary_19" else (
        TRANSFER_SEEDS if name == "transfer_17" else ()
    )
    if seed not in seeds or arm not in ("control", "candidate") or (
        name == "primary_19" and arm == "control"
    ):
        raise ValueError("run outside the frozen arm and seed schedule")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    _, transfer_case, transfer_exact, transfer_logz = _inputs(
        primary_reference, transfer_reference, ledger, wp11ab,
        primary_control, decoy_control, audit,
    )
    if name == "primary_19":
        case, exact, logz = _primary_or_decoy(
            name, primary_reference, transfer_reference, ledger
        )
    else:
        case, exact, logz = transfer_case, transfer_exact, transfer_logz
    config = _candidate(seed) if arm == "candidate" else _controls(seed, "control")
    path = _run_path(run_dir, name, arm, seed)
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["protocol"] != _protocol() or saved["config"] != asdict(config)
                or saved["name"] != name or saved["arm"] != arm
                or saved["seed"] != seed):
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
        "experiment": "wp11ag_whole_slot_frozen_run", "protocol": _protocol(),
        "name": name, "arm": arm, "seed": seed, "config": asdict(config),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - logz,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact),
        "stages": [stage.to_dict() for stage in sampled.stages],
    }
    _save(path, result)
    print(f"{name} {arm} {seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"joint TV {result['errors']['full_joint_tv']:.5f}", flush=True)
    return result


def _validated_run(path: Path, name: str, arm: str, seed: int,
                   exact: np.ndarray) -> dict[str, Any]:
    saved = json.loads(path.read_text())
    config = _candidate(seed) if arm == "candidate" else _controls(seed, "control")
    if (saved["protocol"] != _protocol() or saved["name"] != name
            or saved["arm"] != arm or saved["seed"] != seed
            or saved["config"] != asdict(config)
            or saved["cpu_seconds"] > CPU_CAP
            or saved["n_tempering_steps"] != len(saved["stages"])
            or not np.isclose(saved["stages"][-1]["beta"], 1.0, atol=1e-12)):
        raise ValueError("candidate or control run differs from frozen protocol")
    joint = np.asarray(saved["joint_alive_by_sixth_by_last_clutter"])
    if (joint.shape != exact.shape or not np.isclose(joint.sum(), 1, atol=1e-12)
            or not np.allclose(joint * N_PARTICLES,
                               np.rint(joint * N_PARTICLES), atol=1e-9)):
        raise ValueError("run joint is invalid")
    for key, actual in _metrics(joint, exact).items():
        if not np.allclose(actual, saved["errors"][key], atol=1e-12):
            raise ValueError(f"run metric {key} differs from pinned reference")
    return saved


def aggregate(run_dir: Path, primary_reference: Path, transfer_reference: Path,
              ledger: Path, wp11ab: Path, primary_control: Path,
              decoy_control: Path, audit: Path, output: Path) -> dict[str, Any]:
    previous, transfer_case, transfer_exact, _ = _inputs(
        primary_reference, transfer_reference, ledger, wp11ab,
        primary_control, decoy_control, audit,
    )
    primary_case, primary_exact, _ = _primary_or_decoy(
        "primary_19", primary_reference, transfer_reference, ledger
    )
    result = {"experiment": "wp11ag_whole_slot_comparison", "protocol": _protocol(),
              "cases": {}, "status": "in_progress"}
    for name, seeds, case, exact in (
        ("primary_19", PRIMARY_SEEDS, primary_case, primary_exact),
        ("transfer_17", TRANSFER_SEEDS, transfer_case, transfer_exact),
    ):
        controls, candidates, rows = [], [], {}
        for seed in seeds:
            if name == "primary_19":
                baseline = previous[name]["runs"][str(seed)]["control"]
                control_joint = np.asarray(baseline["joint_alive_by_sixth_by_last_clutter"])
                control = {"cpu_seconds": baseline["cpu_seconds"],
                           "errors": _metrics(control_joint, exact)}
            else:
                saved_control = _validated_run(
                    _run_path(run_dir, name, "control", seed), name, "control", seed, exact
                )
                control = {"cpu_seconds": saved_control["cpu_seconds"],
                           "errors": saved_control["errors"]}
            saved_candidate = _validated_run(
                _run_path(run_dir, name, "candidate", seed), name, "candidate", seed,
                exact,
            )
            candidate = {"cpu_seconds": saved_candidate["cpu_seconds"],
                         "errors": saved_candidate["errors"]}
            controls.append(control)
            candidates.append(candidate)
            rows[str(seed)] = {
                "control": control, "candidate": {
                    **candidate,
                    "n_tempering_steps": saved_candidate["n_tempering_steps"],
                    "log_evidence_error": saved_candidate["log_evidence_error"],
                    "joint_alive_by_sixth_by_last_clutter": saved_candidate[
                        "joint_alive_by_sixth_by_last_clutter"
                    ],
                    "late_whole_slot_selection_fraction": float(np.mean([
                        stage["paired_slot_whole_slot_selected_fraction"]
                        for stage in saved_candidate["stages"] if stage["beta"] >= SWITCH_BETA
                    ])),
                },
            }
        comparison = _previous_decision(
            "primary_19" if name == "primary_19" else "decoy_17", controls, candidates
        )
        if name == "transfer_17":
            comparison["per_seed_modes"] = dict(zip(
                map(str, seeds), comparison["per_seed_modes"].values(), strict=True
            ))
        median_control = median(row["cpu_seconds"] for row in controls)
        median_candidate = median(row["cpu_seconds"] for row in candidates)
        comparison["gates"]["cost"] = all(
            ratio <= MAX_CPU_RATIO for ratio in comparison["per_seed_cpu_ratios"]
        )
        comparison["gates"]["median_cost"] = (
            median_candidate <= MAX_MEDIAN_CPU_RATIO * median_control
        )
        comparison["median_cpu_seconds"] = {"control": median_control,
                                            "candidate": median_candidate}
        comparison["passed"] = all(comparison["gates"].values())
        result["cases"][name] = {
            "cardinalities": list(case.model.sequence.cardinalities),
            "runs": rows, "comparison": comparison,
        }
    result["status"] = (
        "exposed_and_independent_transfer_gates_passed_needs_new_crowded_reference"
        if all(case["comparison"]["passed"] for case in result["cases"].values())
        else "combined_development_gate_failed_stop"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "run", "aggregate"), required=True)
    parser.add_argument("--observation", choices=("primary_19", "transfer_17"))
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--primary-reference", type=Path,
                        default=Path("results/wp11t_subset_exact_primary.json"))
    parser.add_argument("--transfer-reference", type=Path,
                        default=Path("results/wp11af_geometry_transfer_exact.json"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11af_geometry_transfer_ledger.json"))
    parser.add_argument("--wp11ab", type=Path,
                        default=Path("results/wp11ab_late_pair_window.json"))
    parser.add_argument("--primary-control", type=Path,
                        default=Path("results/wp11u_late_suffix_pg.json"))
    parser.add_argument("--decoy-control", type=Path,
                        default=Path("results/wp11y_independent_absolute_sampler.json"))
    parser.add_argument("--stage-audit", type=Path,
                        default=Path("results/wp11z_balanced_stage_summary.json"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/wp11ag_global_slot_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ag_global_slot_comparison.json"))
    args = parser.parse_args()
    inputs = (args.primary_reference, args.transfer_reference, args.ledger,
              args.wp11ab, args.primary_control, args.decoy_control, args.stage_audit)
    if args.case == "protocol":
        _inputs(*inputs)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "run":
        if args.observation is None or args.arm is None or args.seed is None:
            parser.error("run requires observation, arm and seed")
        run_one(args.observation, args.arm, args.seed, args.run_dir, *inputs)
    else:
        if any(value is not None for value in (args.observation, args.arm, args.seed)):
            parser.error("aggregate takes no observation, arm or seed")
        print(aggregate(args.run_dir, *inputs, args.output)["status"])


if __name__ == "__main__":
    main()
