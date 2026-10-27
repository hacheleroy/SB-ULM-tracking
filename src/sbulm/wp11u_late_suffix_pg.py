"""Frozen paired development screen of final-stage conditional suffix PG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .existence_smc import sample_slot_existence_posterior
from .slot_existence import ABSENT
from .wp7_convergence_audit import _save
from .wp8_generalization import _metadata, make_case
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11k_variance_calibration import BANK_HASHES, REGIMES

REFERENCE_SHA256 = "52f13725b9f7dd8851b44ee97025afebf98f042791b111cf508dcbd5c0c0c14c"
SEEDS = (1300, 1301, 1302, 1303, 1304, 1305)
N_PARTICLES = 2_048
CPU_CAP = 750
PRIMARY = next(regime for regime in REGIMES if regime.name == "wp11k_primary")
SUFFIX_START = 5  # zero-based: frames six through eight
SUFFIX_FRACTION = 0.40  # additional to the unchanged 0.10 full-path PG
FINAL_EXTRA_SWEEPS = 24  # instead of 8; total final-stage sweeps 32
MAX_CPU_RATIO = 1.50
MAX_MEDIAN_TV = 0.12
MAX_MEDIAN_BAND_TV = 0.10
MAX_MEDIAN_BAND_MASS_ERROR = 0.08
MAX_RELATIVE_MEDIAN = 0.80
MAX_OTHER_REGRESSION = 0.020
MIN_PAIRED_TV_WINS = 4
OTHER_METRICS = ("alive_total_tv", "maximum_3d_cell_error",
                 "sixth_clutter_mean_error", "last_clutter_mean_error")


def _controls(seed: int, arm: str):
    control = replace(_candidate_controls(seed), n_particles=N_PARTICLES)
    if arm == "control":
        return control
    if arm == "candidate":
        return replace(
            control, final_mh_sweeps=FINAL_EXTRA_SWEEPS,
            paired_slot_suffix_pg_final_fraction=SUFFIX_FRACTION,
            paired_slot_suffix_pg_start_frame=SUFFIX_START,
        )
    raise ValueError("unknown frozen arm")


def _protocol() -> dict[str, Any]:
    return {
        "reference_sha256": REFERENCE_SHA256,
        "observation_sha256": BANK_HASHES[PRIMARY.name],
        "case": "exposed WP11k primary, four slots, eight frames, 19 detections",
        "control": asdict(_controls(SEEDS[0], "control")),
        "candidate": asdict(_controls(SEEDS[0], "candidate")),
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "arm_order": {str(seed): (["control", "candidate"] if i % 2 == 0
                                   else ["candidate", "control"])
                      for i, seed in enumerate(SEEDS)},
        "maximum_process_cpu_seconds_per_arm": CPU_CAP,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "selection_gate": {
            "max_per_seed_candidate_control_cpu_ratio": MAX_CPU_RATIO,
            "minimum_paired_full_joint_tv_wins": MIN_PAIRED_TV_WINS,
            "maximum_candidate_median_full_joint_tv": MAX_MEDIAN_TV,
            "maximum_candidate_median_band_by_sixth_tv": MAX_MEDIAN_BAND_TV,
            "maximum_candidate_median_band_mass_error": MAX_MEDIAN_BAND_MASS_ERROR,
            "maximum_candidate_control_median_ratio_on_each_target": MAX_RELATIVE_MEDIAN,
            "maximum_absolute_median_regression_on_each_other_metric": MAX_OTHER_REGRESSION,
            "iid_q95_only_descriptive": True,
        },
        "interpretation": "exposed-case development; separate observation replication required",
        "holdout": "closed; no 8192-particle or untouched WP8 run",
    }


def _reference(path: Path) -> tuple[Any, dict[str, Any]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("WP11t exact checkpoint bytes changed")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (checkpoint["status"] != "absolute_primary_diagnostic_signal"
            or not checkpoint["reference"]["valid"]):
        raise ValueError("validated WP11t exact checkpoint required")
    case = make_case(PRIMARY)
    if json.loads(json.dumps(_metadata(PRIMARY, case))) != checkpoint["case"]:
        raise ValueError("frozen WP11k observation/model changed")
    return case, checkpoint["reference"]


def _errors(joint: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    truth = np.asarray(exact["joint_alive_by_sixth_by_last_clutter"])
    delta = joint - truth
    collapsed = joint.sum(axis=2)
    bands = np.stack((collapsed[:21].sum(axis=0), collapsed[21],
                      collapsed[22:].sum(axis=0)))
    band_delta = bands - np.asarray(exact["band_by_sixth_clutter"])
    return {
        "full_3d_joint_tv": float(np.abs(delta).sum() / 2),
        "band_by_sixth_joint_tv": float(np.abs(band_delta).sum() / 2),
        "maximum_band_mass_error": float(np.max(np.abs(band_delta.sum(axis=1)))),
        "alive_total_tv": float(np.abs(delta.sum(axis=(1, 2))).sum() / 2),
        "maximum_3d_cell_error": float(np.max(np.abs(delta))),
        "sixth_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[1]), delta.sum(axis=(0, 2))
        )),
        "last_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[2]), delta.sum(axis=(0, 1))
        )),
    }


def _run_path(run_dir: Path, seed: int, arm: str) -> Path:
    return run_dir / f"{seed}_{arm}.json"


def run_one(run_dir: Path, reference_path: Path, seed: int, arm: str) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed outside frozen schedule")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    case, exact = _reference(reference_path)
    controls = _controls(seed, arm)
    path = _run_path(run_dir, seed, arm)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["protocol"] != _protocol() or record["config"] != asdict(controls):
            raise ValueError("run checkpoint differs from frozen protocol")
        return record
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap is below the frozen run cap")
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
    cpu = time.process_time() - started_cpu
    states = sampled.associations
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    sixth = 2 - np.count_nonzero(states[:, 5] >= 0, axis=1)
    last = 1 - np.count_nonzero(states[:, 7] >= 0, axis=1)
    joint = (np.bincount(6 * alive + 2 * sixth + last, minlength=198)
             .reshape(33, 3, 2) / N_PARTICLES)
    record = {
        "protocol": _protocol(), "seed": seed, "arm": arm,
        "config": asdict(controls), "cpu_seconds": cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - exact["log_evidence"],
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _errors(joint, exact),
        "stages": [stage.to_dict() for stage in sampled.stages],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _save(path, record)
    print(f"{seed} {arm}: CPU {cpu:.1f}s, joint TV "
          f"{record['errors']['full_3d_joint_tv']:.5f}", flush=True)
    return record


def _decision(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    controls = [runs[str(seed)]["control"] for seed in SEEDS]
    candidates = [runs[str(seed)]["candidate"] for seed in SEEDS]
    cpu_ratios = [candidate["cpu_seconds"] / control["cpu_seconds"]
                  for control, candidate in zip(controls, candidates, strict=True)]
    keys = ("full_3d_joint_tv", "band_by_sixth_joint_tv",
            "maximum_band_mass_error", *OTHER_METRICS)
    medians = {arm: {key: float(median(abs(run["errors"][key]) for run in arm_runs))
                     for key in keys}
               for arm, arm_runs in (("control", controls), ("candidate", candidates))}
    target_caps = {
        "full_3d_joint_tv": MAX_MEDIAN_TV,
        "band_by_sixth_joint_tv": MAX_MEDIAN_BAND_TV,
        "maximum_band_mass_error": MAX_MEDIAN_BAND_MASS_ERROR,
    }
    wins = sum(candidate["errors"]["full_3d_joint_tv"]
               < control["errors"]["full_3d_joint_tv"]
               for control, candidate in zip(controls, candidates, strict=True))
    gates = {
        "cost": all(ratio <= MAX_CPU_RATIO for ratio in cpu_ratios),
        "paired_joint_wins": wins >= MIN_PAIRED_TV_WINS,
        "absolute_targets": all(medians["candidate"][key] <= cap
                                for key, cap in target_caps.items()),
        "relative_targets": all(
            medians["candidate"][key] <= MAX_RELATIVE_MEDIAN * medians["control"][key]
            for key in target_caps
        ),
        "other_guardrails": all(
            medians["candidate"][key] - medians["control"][key] <= MAX_OTHER_REGRESSION
            for key in OTHER_METRICS
        ),
    }
    return {"per_seed_cpu_ratios": cpu_ratios, "paired_joint_wins": wins,
            "median_absolute_errors": medians, "gates": gates,
            "passed": all(gates.values())}


def aggregate(run_dir: Path, reference_path: Path, output: Path) -> dict[str, Any]:
    _, exact = _reference(reference_path)
    runs: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        rows = {}
        for arm in _protocol()["arm_order"][str(seed)]:
            record = json.loads(_run_path(run_dir, seed, arm).read_text(encoding="utf-8"))
            if record["protocol"] != _protocol() or record["config"] != asdict(
                _controls(seed, arm)
            ) or record["seed"] != seed or record["arm"] != arm:
                raise ValueError("run checkpoint does not match frozen protocol")
            joint = np.asarray(record["joint_alive_by_sixth_by_last_clutter"])
            if joint.shape != (33, 3, 2) or not np.isclose(joint.sum(), 1):
                raise ValueError("invalid final joint")
            for key, value in _errors(joint, exact).items():
                if not np.isclose(value, record["errors"][key], atol=1e-12):
                    raise ValueError("saved absolute error changed")
            rows[arm] = record
        runs[str(seed)] = rows
    result = {"experiment": "wp11u_late_suffix_pg", "protocol": _protocol(),
              "reference_log_evidence": exact["log_evidence"], "runs": runs,
              "comparison": _decision(runs)}
    result["status"] = ("exposed_development_gate_passed_needs_replication"
                        if result["comparison"]["passed"]
                        else "exposed_development_gate_failed_stop")
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11u_runs"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/wp11t_subset_exact_primary.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11u_late_suffix_pg.json"))
    args = parser.parse_args()
    if args.aggregate:
        if args.seed is not None or args.arm is not None:
            parser.error("aggregate takes no arm/seed")
        print(aggregate(args.run_dir, args.reference, args.output)["status"], flush=True)
    else:
        if args.seed is None or args.arm is None:
            parser.error("a run requires its frozen arm and seed")
        run_one(args.run_dir, args.reference, args.seed, args.arm)


if __name__ == "__main__":
    main()
