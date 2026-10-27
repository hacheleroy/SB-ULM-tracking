"""Frozen two-case comparison of an earlier conditional suffix PG schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .existence_smc import BirthDeathSMCConfig, sample_slot_existence_posterior
from .slot_existence import ABSENT
from .wp7_convergence_audit import _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11u_late_suffix_pg import (
    REFERENCE_SHA256 as PRIMARY_REFERENCE_SHA256,
)
from .wp11u_late_suffix_pg import (
    _protocol as _wp11u_protocol,
)
from .wp11u_late_suffix_pg import (
    _reference as _primary_reference,
)
from .wp11x_frame_six_decoy_reference import REFERENCE_SHA256 as DECOY_REFERENCE_SHA256
from .wp11y_independent_absolute_sampler import (
    _inputs as _decoy_inputs,
)
from .wp11y_independent_absolute_sampler import (
    _protocol as _wp11y_protocol,
)

PRIMARY_CONTROL_SHA256 = "cc3a478913938537e04a498e87c65506ae0da42eb1fcfe90ce7feb380aee9962"
DECOY_CONTROL_SHA256 = "3af1e36390c5700d150291e5493608009f319498f44f9ec4164d89654c795f50"
STAGE_AUDIT_SHA256 = "805ba60c16bdb768d1236cba6d6cb48491639555874b79033bb4eba987640ab9"
PRIMARY_SEEDS = (1300, 1301, 1302, 1303, 1304, 1305)
DECOY_SEEDS = (1500, 1501, 1502, 1503)
N_PARTICLES = 2_048
CPU_CAP = 750
SWITCH_BETA = .75
SUFFIX_FRACTION = .40
SUFFIX_START = 5
MAX_CPU_RATIO = 1.50
PRIMARY_MIN_WINS = 4
DECOY_MIN_WINS = 3
MAX_MEDIAN_FULL_TV = .12
MAX_MEDIAN_BAND_TV = .10
MAX_MEDIAN_BAND_MASS = .08
MAX_MEDIAN_CELL = .04
MAX_PRIMARY_RELATIVE = .80
MAX_DECOY_RELATIVE_FULL = .90
MAX_OTHER_REGRESSION = .020
MIN_DECOY_MODES = 3
DECOY_CLUTTER_ONE_CAP = .06
OTHER_METRICS = ("alive_total_tv", "maximum_joint_cell_error",
                 "sixth_clutter_mean_absolute_error", "last_clutter_mean_absolute_error")


@dataclass(frozen=True)
class TemperedSuffixSMCConfig(BirthDeathSMCConfig):
    """Opt-in schedule without changing serialized historical controls."""

    paired_slot_suffix_pg_beta_switch: float = SWITCH_BETA

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not np.isfinite(self.paired_slot_suffix_pg_beta_switch)
                or not 0 < self.paired_slot_suffix_pg_beta_switch <= 1
                or self.paired_slot_suffix_pg_final_fraction <= 0):
            raise ValueError("tempered suffix PG requires a positive fraction and switch")
        earlier = (self.paired_slot_pg_early_fraction
                   if self.paired_slot_pg_early_fraction is not None
                   else self.paired_slot_pg_fraction)
        if max(earlier, self.paired_slot_pg_fraction) + (
            self.paired_slot_suffix_pg_final_fraction
        ) > 1:
            raise ValueError("full and suffix PG fractions exceed one")


def _controls(seed: int, arm: str) -> BirthDeathSMCConfig:
    control = replace(_candidate_controls(seed), n_particles=N_PARTICLES)
    if arm == "control":
        return control
    if arm != "candidate":
        raise ValueError("unknown frozen arm")
    options = asdict(control)
    options.update(paired_slot_suffix_pg_final_fraction=SUFFIX_FRACTION,
                   paired_slot_suffix_pg_start_frame=SUFFIX_START,
                   paired_slot_suffix_pg_beta_switch=SWITCH_BETA)
    return TemperedSuffixSMCConfig(**options)


def _protocol() -> dict[str, Any]:
    return {
        "references": {"primary_19": PRIMARY_REFERENCE_SHA256,
                       "decoy_17": DECOY_REFERENCE_SHA256},
        "controls": {"primary_19": PRIMARY_CONTROL_SHA256,
                     "decoy_17": DECOY_CONTROL_SHA256},
        "stage_audit_sha256": STAGE_AUDIT_SHA256,
        "seeds": {"primary_19": list(PRIMARY_SEEDS), "decoy_17": list(DECOY_SEEDS)},
        "control_configuration_seed_1300": asdict(_controls(1300, "control")),
        "candidate_configuration_seed_1300": asdict(_controls(1300, "candidate")),
        "move": "four-particle pair suffix PG on frames 6..8; 40% of paired proposals "
                "at beta >=.75, 10% full-path PG retained and 50% local paired; "
                "otherwise unchanged control, no extra sweeps",
        "n_particles": N_PARTICLES,
        "maximum_process_cpu_seconds_per_candidate": CPU_CAP,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "primary_gate": {
            "max_per_seed_cpu_ratio": MAX_CPU_RATIO,
            "minimum_paired_full_joint_tv_wins": PRIMARY_MIN_WINS,
            "median_full_joint_tv_at_most": MAX_MEDIAN_FULL_TV,
            "median_band_by_sixth_tv_at_most": MAX_MEDIAN_BAND_TV,
            "median_maximum_alive_band_error_at_most": MAX_MEDIAN_BAND_MASS,
            "candidate_control_median_ratio_for_each_target_at_most": MAX_PRIMARY_RELATIVE,
            "maximum_median_regression_on_each_other_metric": MAX_OTHER_REGRESSION,
        },
        "decoy_gate": {
            "max_per_seed_cpu_ratio": MAX_CPU_RATIO,
            "minimum_paired_full_joint_tv_wins": DECOY_MIN_WINS,
            "median_full_joint_tv_at_most": MAX_MEDIAN_FULL_TV,
            "median_band_by_sixth_tv_at_most": MAX_MEDIAN_BAND_TV,
            "median_maximum_alive_band_error_at_most": MAX_MEDIAN_BAND_MASS,
            "median_maximum_cell_error_at_most": MAX_MEDIAN_CELL,
            "candidate_control_median_full_joint_tv_ratio_at_most": MAX_DECOY_RELATIVE_FULL,
            "minimum_runs_meeting_mode_limits": MIN_DECOY_MODES,
            "per_run_mode_limits": {"band_by_sixth_tv": MAX_MEDIAN_BAND_TV,
                                    "maximum_alive_band_error": MAX_MEDIAN_BAND_MASS,
                                    "absolute_sixth_clutter_one_error": DECOY_CLUTTER_ONE_CAP,
                                    "absolute_last_clutter_one_error": DECOY_CLUTTER_ONE_CAP},
            "maximum_median_regression_on_each_other_metric": MAX_OTHER_REGRESSION,
        },
        "decision": "both exposed-case gates must pass; run all ten regardless of early result",
        "interpretation": "development only; new independent observation required after a pass",
        "holdout": "closed; no 8192-particle or untouched WP8 run",
    }


def _load_pinned(path: Path, expected: str) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"frozen checkpoint bytes changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _previous(primary_path: Path, decoy_path: Path, audit_path: Path) -> dict[str, Any]:
    primary = _load_pinned(primary_path, PRIMARY_CONTROL_SHA256)
    decoy = _load_pinned(decoy_path, DECOY_CONTROL_SHA256)
    audit = _load_pinned(audit_path, STAGE_AUDIT_SHA256)
    if (primary["protocol"] != _wp11u_protocol()
            or primary["status"] != "exposed_development_gate_failed_stop"
            or decoy["protocol"] != _wp11y_protocol()
            or decoy["status"] != "independent_17_detection_baseline_failed_limits"
            or audit["status"] != "balanced_stage_diagnostic_completed_no_move_gate"
            or set(primary["runs"]) != set(map(str, PRIMARY_SEEDS))
            or set(decoy["runs"]) != set(map(str, DECOY_SEEDS))):
        raise ValueError("frozen control or stage audit differs from exposed checkpoint")
    return {"primary_19": primary, "decoy_17": decoy}


def _case(name: str, primary_reference: Path, decoy_reference: Path,
          ledger: Path) -> tuple[Any, np.ndarray, float]:
    if name == "primary_19":
        case, checkpoint = _primary_reference(primary_reference)
        return (case, np.asarray(checkpoint["joint_alive_by_sixth_by_last_clutter"]),
                float(checkpoint["log_evidence"]))
    if name == "decoy_17":
        return _decoy_inputs(decoy_reference, ledger)
    raise ValueError("case outside frozen comparison")


def _joint(states: np.ndarray, cards: tuple[int, ...]) -> np.ndarray:
    if states.shape != (N_PARTICLES, 8, 4):
        raise ValueError("unexpected particle shape")
    shape = (33, cards[5] + 1, cards[7] + 1)
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    sixth = cards[5] - np.count_nonzero(states[:, 5] >= 0, axis=1)
    last = cards[7] - np.count_nonzero(states[:, 7] >= 0, axis=1)
    cell = np.ravel_multi_index((alive, sixth, last), shape)
    return np.bincount(cell, minlength=int(np.prod(shape))).reshape(shape) / N_PARTICLES


def _metrics(joint: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    if joint.shape != exact.shape or joint.shape[0] != 33:
        raise ValueError("joint shape differs from exact reference")
    difference = joint - exact
    band_delta = np.stack((difference[:21].sum(axis=0), difference[21],
                           difference[22:].sum(axis=0)))
    sixth = difference.sum(axis=(0, 2))
    last = difference.sum(axis=(0, 1))
    return {
        "full_joint_tv": float(np.abs(difference).sum() / 2),
        "band_by_sixth_tv": float(np.abs(band_delta.sum(axis=2)).sum() / 2),
        "maximum_alive_band_error": float(np.max(np.abs(band_delta.sum(axis=(1, 2))))),
        "alive_total_tv": float(np.abs(difference.sum(axis=(1, 2))).sum() / 2),
        "maximum_joint_cell_error": float(np.max(np.abs(difference))),
        "alive_band_signed_errors": band_delta.sum(axis=(1, 2)).tolist(),
        "sixth_clutter_signed_errors": sixth.tolist(),
        "last_clutter_signed_errors": last.tolist(),
        "sixth_clutter_mean_absolute_error": abs(float(np.dot(np.arange(len(sixth)), sixth))),
        "last_clutter_mean_absolute_error": abs(float(np.dot(np.arange(len(last)), last))),
    }


def _run_path(run_dir: Path, name: str, seed: int) -> Path:
    return run_dir / f"{name}_{seed}.json"


def run_one(name: str, seed: int, run_dir: Path, primary_reference: Path,
            decoy_reference: Path, ledger: Path, primary_control: Path,
            decoy_control: Path, audit: Path) -> dict[str, Any]:
    if seed not in (PRIMARY_SEEDS if name == "primary_19" else DECOY_SEEDS
                    if name == "decoy_17" else ()):
        raise ValueError("seed outside frozen case schedule")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    prior = _previous(primary_control, decoy_control, audit)
    case, exact, log_z = _case(name, primary_reference, decoy_reference, ledger)
    cards = tuple(case.model.sequence.cardinalities)
    control = (prior[name]["runs"][str(seed)]["control"] if name == "primary_19"
               else prior[name]["runs"][str(seed)])
    if name == "primary_19" and control["config"] != asdict(_controls(seed, "control")):
        raise ValueError("archived primary control configuration changed")
    controls = _controls(seed, "candidate")
    path = _run_path(run_dir, name, seed)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["protocol"] != _protocol() or saved["case"] != name
                or saved["seed"] != seed or saved["config"] != asdict(controls)):
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
        birth_death_prior=case.model.birth_death_prior, config=controls,
    )
    joint = _joint(sampled.associations, cards)
    stages = [stage.to_dict() for stage in sampled.stages]
    result = {
        "experiment": "wp11aa_tempered_suffix_pg_candidate", "protocol": _protocol(),
        "case": name, "seed": seed, "config": asdict(controls),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - log_z,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact), "stages": stages,
    }
    _save(path, result)
    print(f"{name} {seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"joint TV {result['errors']['full_joint_tv']:.5f}", flush=True)
    return result


def _medians(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("full_joint_tv", "band_by_sixth_tv", "maximum_alive_band_error",
            *OTHER_METRICS)
    return {key: float(median(row["errors"][key] for row in rows)) for key in keys}


def _decision(name: str, controls: list[dict[str, Any]],
              candidates: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = PRIMARY_SEEDS if name == "primary_19" else DECOY_SEEDS
    control_medians, candidate_medians = _medians(controls), _medians(candidates)
    cpu_ratios = [candidate["cpu_seconds"] / control["cpu_seconds"]
                  for control, candidate in zip(controls, candidates, strict=True)]
    wins = sum(candidate["errors"]["full_joint_tv"]
               < control["errors"]["full_joint_tv"]
               for control, candidate in zip(controls, candidates, strict=True))
    targets = {"full_joint_tv": MAX_MEDIAN_FULL_TV,
               "band_by_sixth_tv": MAX_MEDIAN_BAND_TV,
               "maximum_alive_band_error": MAX_MEDIAN_BAND_MASS}
    gates = {
        "cost": all(ratio <= MAX_CPU_RATIO for ratio in cpu_ratios),
        "paired_full_joint_wins": wins >= (PRIMARY_MIN_WINS if name == "primary_19"
                                           else DECOY_MIN_WINS),
        "absolute_targets": all(candidate_medians[key] <= cap
                                for key, cap in targets.items()),
        "other_guardrails": all(candidate_medians[key] - control_medians[key]
                                <= MAX_OTHER_REGRESSION for key in OTHER_METRICS),
    }
    if name == "primary_19":
        gates["relative_targets"] = all(
            candidate_medians[key] <= MAX_PRIMARY_RELATIVE * control_medians[key]
            for key in targets
        )
        modes = None
    else:
        gates["absolute_cell"] = candidate_medians["maximum_joint_cell_error"] <= (
            MAX_MEDIAN_CELL
        )
        gates["relative_full_joint"] = candidate_medians["full_joint_tv"] <= (
            MAX_DECOY_RELATIVE_FULL * control_medians["full_joint_tv"]
        )
        modes = {str(seed): (
            candidate["errors"]["band_by_sixth_tv"] <= MAX_MEDIAN_BAND_TV
            and candidate["errors"]["maximum_alive_band_error"] <= MAX_MEDIAN_BAND_MASS
            and abs(candidate["errors"]["sixth_clutter_signed_errors"][1]) <= (
                DECOY_CLUTTER_ONE_CAP
            ) and abs(candidate["errors"]["last_clutter_signed_errors"][1]) <= (
                DECOY_CLUTTER_ONE_CAP
            )
        ) for seed, candidate in zip(seeds, candidates, strict=True)}
        gates["replicated_modes"] = sum(modes.values()) >= MIN_DECOY_MODES
    return {"control_medians": control_medians, "candidate_medians": candidate_medians,
            "per_seed_cpu_ratios": cpu_ratios, "paired_full_joint_wins": wins,
            "per_seed_modes": modes, "gates": gates, "passed": all(gates.values())}


def aggregate(run_dir: Path, primary_reference: Path, decoy_reference: Path,
              ledger: Path, primary_control: Path, decoy_control: Path,
              audit: Path, output: Path) -> dict[str, Any]:
    previous = _previous(primary_control, decoy_control, audit)
    result = {"experiment": "wp11aa_tempered_suffix_pg", "protocol": _protocol(),
              "cases": {}, "status": "in_progress"}
    for name, seeds in (("primary_19", PRIMARY_SEEDS), ("decoy_17", DECOY_SEEDS)):
        case, exact, _ = _case(name, primary_reference, decoy_reference, ledger)
        controls, candidates, rows = [], [], {}
        for seed in seeds:
            baseline = (previous[name]["runs"][str(seed)]["control"]
                        if name == "primary_19" else previous[name]["runs"][str(seed)])
            old = np.asarray(baseline["joint_alive_by_sixth_by_last_clutter"])
            control = {"cpu_seconds": baseline["cpu_seconds"],
                       "errors": _metrics(old, exact)}
            saved = json.loads(_run_path(run_dir, name, seed).read_text(encoding="utf-8"))
            if (saved["protocol"] != _protocol() or saved["case"] != name
                    or saved["seed"] != seed
                    or saved["config"] != asdict(_controls(seed, "candidate"))
                    or saved["cpu_seconds"] > CPU_CAP
                    or saved["n_tempering_steps"] != len(saved["stages"])
                    or not np.isclose(saved["stages"][-1]["beta"], 1.0, atol=1e-12)):
                raise ValueError("candidate run differs from frozen comparison")
            joint = np.asarray(saved["joint_alive_by_sixth_by_last_clutter"])
            if (joint.shape != exact.shape or not np.isclose(joint.sum(), 1, atol=1e-12)
                    or not np.allclose(joint * N_PARTICLES,
                                       np.rint(joint * N_PARTICLES), atol=1e-9)):
                raise ValueError("invalid candidate final joint")
            for key, value in _metrics(joint, exact).items():
                if not np.allclose(value, saved["errors"][key], atol=1e-12):
                    raise ValueError(f"saved candidate metric {key} changed")
            activated = [stage for stage in saved["stages"] if stage["beta"] >= SWITCH_BETA]
            early = [stage for stage in saved["stages"] if stage["beta"] < SWITCH_BETA]
            if any(stage["paired_slot_suffix_pg_selected_fraction"] != 0 for stage in early):
                raise ValueError("suffix move activated before frozen beta switch")
            candidate = {"cpu_seconds": saved["cpu_seconds"], "errors": saved["errors"]}
            controls.append(control)
            candidates.append(candidate)
            rows[str(seed)] = {
                "control": control, "candidate": {
                    **candidate, "n_tempering_steps": saved["n_tempering_steps"],
                    "log_evidence_error": saved["log_evidence_error"],
                    "joint_alive_by_sixth_by_last_clutter": saved[
                        "joint_alive_by_sixth_by_last_clutter"
                    ],
                    "mean_active_suffix_selection_fraction": float(np.mean([
                        stage["paired_slot_suffix_pg_selected_fraction"]
                        for stage in activated
                    ])),
                    "mean_active_suffix_count_change_rate": float(np.mean([
                        stage["paired_slot_suffix_pg_count_signature_change_rate"]
                        for stage in activated
                    ])),
                    "first_active_beta": activated[0]["beta"],
                },
            }
        result["cases"][name] = {
            "cardinalities": list(case.model.sequence.cardinalities),
            "runs": rows, "comparison": _decision(name, controls, candidates),
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
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11aa_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11aa_tempered_suffix_pg.json"))
    args = parser.parse_args()
    if args.case == "protocol":
        _previous(args.primary_control, args.decoy_control, args.stage_audit)
        _case("primary_19", args.primary_reference, args.decoy_reference, args.ledger)
        _case("decoy_17", args.primary_reference, args.decoy_reference, args.ledger)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "aggregate":
        if args.observation is not None or args.seed is not None:
            parser.error("aggregate takes no observation or seed")
        print(aggregate(args.run_dir, args.primary_reference, args.decoy_reference,
                        args.ledger, args.primary_control, args.decoy_control,
                        args.stage_audit, args.output)["status"])
    else:
        if args.observation is None or args.seed is None:
            parser.error("run requires observation and seed")
        run_one(args.observation, args.seed, args.run_dir, args.primary_reference,
                args.decoy_reference, args.ledger, args.primary_control,
                args.decoy_control, args.stage_audit)


if __name__ == "__main__":
    main()
