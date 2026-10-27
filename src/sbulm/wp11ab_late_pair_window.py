"""Frozen two-case comparison of a late, geometry-targeted local pair move."""

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
    DECOY_SEEDS,
    N_PARTICLES,
    PRIMARY_SEEDS,
    _case,
    _controls,
    _joint,
    _metrics,
    _previous,
)
from .wp11aa_tempered_suffix_pg import _decision as _previous_decision
from .wp11aa_tempered_suffix_pg import _protocol as _previous_protocol
from .wp11f_interleaved_cost import THREAD_ENV

WP11AA_SHA256 = "620b5db1603af7626933000b70b38ba40b5239805ccf1a9ea247e5d55292a1ba"
SWITCH_BETA = .75
FOCUS_SHARE = .50
SIXTH_SHARE = .75
START_FRAMES = (5, 7)
CPU_CAP = 500
MAX_PER_SEED_CONTROL_CPU_RATIO = 1.25
MAX_MEDIAN_WP11AA_CPU_RATIO = .80


@dataclass(frozen=True)
class LateWindowSMCConfig(BirthDeathSMCConfig):
    """Opt in without changing the serialized historical control config."""

    paired_slot_late_window_beta_switch: float = SWITCH_BETA
    paired_slot_late_window_focus_share: float = FOCUS_SHARE
    paired_slot_late_window_sixth_share: float = SIXTH_SHARE
    paired_slot_late_window_start_frames: tuple[int, int] = START_FRAMES

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.paired_slot_scope != "conditional_smc":
            raise ValueError("late local pair windows require conditional SMC scope")
        if (not np.isfinite(self.paired_slot_late_window_beta_switch)
                or not 0 < self.paired_slot_late_window_beta_switch <= 1):
            raise ValueError("late window beta switch must lie in (0, 1]")
        if (not np.isfinite(self.paired_slot_late_window_focus_share)
                or not 0 <= self.paired_slot_late_window_focus_share <= 1
                or not np.isfinite(self.paired_slot_late_window_sixth_share)
                or not 0 <= self.paired_slot_late_window_sixth_share <= 1):
            raise ValueError("late window shares must lie in [0, 1]")
        if (len(self.paired_slot_late_window_start_frames) != 2
                or not 0 <= self.paired_slot_late_window_start_frames[0]
                < self.paired_slot_late_window_start_frames[1]):
            raise ValueError("late window starts must be ordered non-negative frames")


def _candidate(seed: int) -> LateWindowSMCConfig:
    return LateWindowSMCConfig(**asdict(_controls(seed, "control")))


def _config_dict(config: LateWindowSMCConfig) -> dict[str, Any]:
    """Use JSON's list representation for the two frozen window indices."""

    value = asdict(config)
    value["paired_slot_late_window_start_frames"] = list(
        config.paired_slot_late_window_start_frames
    )
    return value


def _protocol() -> dict[str, Any]:
    prior = _previous_protocol()
    primary_gate = dict(prior["primary_gate"])
    decoy_gate = dict(prior["decoy_gate"])
    primary_gate["max_per_seed_cpu_ratio"] = MAX_PER_SEED_CONTROL_CPU_RATIO
    decoy_gate["max_per_seed_cpu_ratio"] = MAX_PER_SEED_CONTROL_CPU_RATIO
    return {
        "references": prior["references"], "controls": prior["controls"],
        "wp11aa_result_sha256": WP11AA_SHA256,
        "seeds": {"primary_19": list(PRIMARY_SEEDS), "decoy_17": list(DECOY_SEEDS)},
        "candidate_configuration_seed_1300": _config_dict(_candidate(1300)),
        "move": "At beta >= .75, half of the existing local paired proposals select "
                "a start frame from frame six (75%) or the last frame (25%); "
                "the other half retain a uniform start. The existing 10% full-path "
                "four-particle PG, local candidate count, and sweeps stay fixed. "
                "Earlier stages are bit-for-bit control-equivalent.",
        "n_particles": N_PARTICLES,
        "maximum_process_cpu_seconds_per_candidate": CPU_CAP,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "accuracy_and_replication": {
            "primary": primary_gate, "decoy": decoy_gate,
        },
        "cost": {
            "each_candidate_over_archived_control_at_most": MAX_PER_SEED_CONTROL_CPU_RATIO,
            "candidate_median_over_wp11aa_median_at_most": MAX_MEDIAN_WP11AA_CPU_RATIO,
        },
        "decision": "both complete exposed-case gates pass; run all ten fixed seeds",
        "interpretation": "development only, no post-result retuning; a pass needs a new "
                          "independently selected observation",
        "holdout": "closed; no 8192-particle or untouched WP8 run",
    }


def _inputs(primary_control: Path, decoy_control: Path, audit: Path,
            wp11aa_result: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = _previous(primary_control, decoy_control, audit)
    if hashlib.sha256(wp11aa_result.read_bytes()).hexdigest() != WP11AA_SHA256:
        raise ValueError("WP11aa comparison bytes changed")
    prior = json.loads(wp11aa_result.read_text(encoding="utf-8"))
    if (prior["protocol"] != _previous_protocol()
            or prior["status"] != "two_exposed_case_gate_failed_stop"):
        raise ValueError("WP11aa comparison or gate differs from frozen checkpoint")
    for name, seeds in (("primary_19", PRIMARY_SEEDS), ("decoy_17", DECOY_SEEDS)):
        if set(prior["cases"][name]["runs"]) != set(map(str, seeds)):
            raise ValueError("WP11aa seed schedule differs from frozen checkpoint")
    return previous, prior


def _run_path(run_dir: Path, name: str, seed: int) -> Path:
    return run_dir / f"{name}_{seed}.json"


def run_one(name: str, seed: int, run_dir: Path, primary_reference: Path,
            decoy_reference: Path, ledger: Path, primary_control: Path,
            decoy_control: Path, audit: Path, wp11aa_result: Path) -> dict[str, Any]:
    if seed not in (PRIMARY_SEEDS if name == "primary_19" else DECOY_SEEDS
                    if name == "decoy_17" else ()):
        raise ValueError("seed outside frozen schedule")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    _inputs(primary_control, decoy_control, audit, wp11aa_result)
    case, exact, log_z = _case(name, primary_reference, decoy_reference, ledger)
    config = _candidate(seed)
    path = _run_path(run_dir, name, seed)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["protocol"] != _protocol() or saved["case"] != name
                or saved["seed"] != seed or saved["config"] != _config_dict(config)):
            raise ValueError("candidate checkpoint differs from frozen protocol")
        return saved
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen candidate cap")
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
        "experiment": "wp11ab_late_pair_window_candidate", "protocol": _protocol(),
        "case": name, "seed": seed, "config": _config_dict(config),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - log_z,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact),
        "stages": [stage.to_dict() for stage in sampled.stages],
    }
    _save(path, result)
    print(f"{name} {seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"joint TV {result['errors']['full_joint_tv']:.5f}", flush=True)
    return result


def aggregate(run_dir: Path, primary_reference: Path, decoy_reference: Path,
              ledger: Path, primary_control: Path, decoy_control: Path,
              audit: Path, wp11aa_result: Path, output: Path) -> dict[str, Any]:
    previous, prior = _inputs(primary_control, decoy_control, audit, wp11aa_result)
    result: dict[str, Any] = {
        "experiment": "wp11ab_late_pair_window", "protocol": _protocol(),
        "cases": {}, "status": "in_progress",
    }
    for name, seeds in (("primary_19", PRIMARY_SEEDS), ("decoy_17", DECOY_SEEDS)):
        case, exact, _ = _case(name, primary_reference, decoy_reference, ledger)
        controls, candidates, rows = [], [], {}
        for seed in seeds:
            baseline = (previous[name]["runs"][str(seed)]["control"]
                        if name == "primary_19" else previous[name]["runs"][str(seed)])
            control = {"cpu_seconds": baseline["cpu_seconds"],
                       "errors": _metrics(np.asarray(
                           baseline["joint_alive_by_sixth_by_last_clutter"]), exact)}
            saved = json.loads(_run_path(run_dir, name, seed).read_text(encoding="utf-8"))
            if (saved["protocol"] != _protocol() or saved["case"] != name
                    or saved["seed"] != seed or saved["config"] != _config_dict(_candidate(seed))
                    or saved["cpu_seconds"] > CPU_CAP
                    or saved["n_tempering_steps"] != len(saved["stages"])
                    or not np.isclose(saved["stages"][-1]["beta"], 1.0, atol=1e-12)):
                raise ValueError("candidate run differs from frozen comparison")
            joint = np.asarray(saved["joint_alive_by_sixth_by_last_clutter"])
            if (joint.shape != exact.shape or not np.isclose(joint.sum(), 1, atol=1e-12)
                    or not np.allclose(joint * N_PARTICLES,
                                       np.rint(joint * N_PARTICLES), atol=1e-9)):
                raise ValueError("invalid candidate joint")
            for key, value in _metrics(joint, exact).items():
                if not np.allclose(value, saved["errors"][key], atol=1e-12):
                    raise ValueError(f"saved candidate metric {key} changed")
            candidate = {"cpu_seconds": saved["cpu_seconds"], "errors": saved["errors"]}
            controls.append(control)
            candidates.append(candidate)
            rows[str(seed)] = {"control": control, "candidate": {
                **candidate, "n_tempering_steps": saved["n_tempering_steps"],
                "log_evidence_error": saved["log_evidence_error"],
                "joint_alive_by_sixth_by_last_clutter": saved[
                    "joint_alive_by_sixth_by_last_clutter"],
            }}
        comparison = _previous_decision(name, controls, candidates)
        ratios = comparison["per_seed_cpu_ratios"]
        prior_median_cpu = median(prior["cases"][name]["runs"][str(seed)]["candidate"][
            "cpu_seconds"] for seed in seeds)
        candidate_median_cpu = median(row["cpu_seconds"] for row in candidates)
        comparison["gates"]["cost"] = all(
            ratio <= MAX_PER_SEED_CONTROL_CPU_RATIO for ratio in ratios
        )
        comparison["gates"]["cheaper_than_wp11aa"] = (
            candidate_median_cpu <= MAX_MEDIAN_WP11AA_CPU_RATIO * prior_median_cpu
        )
        comparison["median_candidate_cpu_seconds"] = candidate_median_cpu
        comparison["median_wp11aa_cpu_seconds"] = prior_median_cpu
        comparison["passed"] = all(comparison["gates"].values())
        result["cases"][name] = {
            "cardinalities": list(case.model.sequence.cardinalities),
            "runs": rows, "comparison": comparison,
        }
    result["status"] = (
        "two_exposed_case_gate_passed_needs_new_observation"
        if all(row["comparison"]["passed"] for row in result["cases"].values())
        else "two_exposed_case_gate_failed_stop"
    )
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "run", "aggregate"), required=True)
    parser.add_argument("--observation", choices=("primary_19", "decoy_17"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--primary-reference", type=Path,
                        default=Path("results/wp11t_subset_exact_primary.json"))
    parser.add_argument("--decoy-reference", type=Path,
                        default=Path("results/wp11x_frame_six_exact_reference.json"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11x_geometry_ledger.json"))
    parser.add_argument("--primary-control", type=Path,
                        default=Path("results/wp11u_late_suffix_pg.json"))
    parser.add_argument("--decoy-control", type=Path,
                        default=Path("results/wp11y_independent_absolute_sampler.json"))
    parser.add_argument("--stage-audit", type=Path,
                        default=Path("results/wp11z_balanced_stage_summary.json"))
    parser.add_argument("--wp11aa-result", type=Path,
                        default=Path("results/wp11aa_tempered_suffix_pg.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11ab_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ab_late_pair_window.json"))
    args = parser.parse_args()
    if args.case == "protocol":
        _inputs(args.primary_control, args.decoy_control, args.stage_audit,
                args.wp11aa_result)
        _case("primary_19", args.primary_reference, args.decoy_reference, args.ledger)
        _case("decoy_17", args.primary_reference, args.decoy_reference, args.ledger)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "aggregate":
        if args.observation is not None or args.seed is not None:
            parser.error("aggregate takes no observation or seed")
        print(aggregate(args.run_dir, args.primary_reference, args.decoy_reference,
                        args.ledger, args.primary_control, args.decoy_control,
                        args.stage_audit, args.wp11aa_result, args.output)["status"])
    else:
        if args.observation is None or args.seed is None:
            parser.error("run requires observation and seed")
        run_one(args.observation, args.seed, args.run_dir, args.primary_reference,
                args.decoy_reference, args.ledger, args.primary_control,
                args.decoy_control, args.stage_audit, args.wp11aa_result)


if __name__ == "__main__":
    main()
