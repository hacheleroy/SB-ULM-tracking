"""Frozen absolute comparison of unchanged WP11g on the WP11x observation."""

from __future__ import annotations

import argparse
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
from .wp7_convergence_audit import _case_metadata, _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11x_frame_six_decoy_reference import (
    REFERENCE_SHA256,
    _checked_case,
    _ledger,
    _screen_order,
    validated_reference,
)

SEEDS = (1500, 1501, 1502, 1503)
N_PARTICLES = 2_048
CPU_CAP = 750
IID_SEED = 45_671
IID_REPLICATES = 1_000
DETECTION_SHA256 = "dd3d3fbb55fd601298ee410539767df24950e2f886d814c30b98f54e89bc0c29"
CARDINALITIES = (1, 2, 2, 2, 2, 3, 3, 2)
FULL_TV_CAP = .12
BAND_TV_CAP = .10
BAND_MASS_CAP = .08
CLUTTER_ONE_CAP = .06
MAX_CELL_MEDIAN_CAP = .04
MIN_REPLICATED_RUNS = 3


def _controls(seed: int):
    return replace(_candidate_controls(seed), n_particles=N_PARTICLES)


def _protocol() -> dict[str, Any]:
    return {
        "reference_sha256": REFERENCE_SHA256,
        "detection_sha256": DETECTION_SHA256,
        "cardinalities": list(CARDINALITIES),
        "reference_cell_shape": [33, 4, 3],
        "sampler": "unchanged WP11g 10% four-path PG paired mixture",
        "configuration_seed_1500": asdict(_controls(SEEDS[0])),
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "maximum_process_cpu_seconds_per_run": CPU_CAP,
        "iid_descriptive": {"seed": IID_SEED, "replicates": IID_REPLICATES,
                            "sample_size": N_PARTICLES, "quantile": .95},
        "adequacy_screen": {
            "median_full_joint_tv_at_most": FULL_TV_CAP,
            "median_band_by_sixth_tv_at_most": BAND_TV_CAP,
            "median_maximum_alive_band_mass_error_at_most": BAND_MASS_CAP,
            "median_maximum_joint_cell_error_at_most": MAX_CELL_MEDIAN_CAP,
            "at_least_runs_meeting_all_mode_limits": MIN_REPLICATED_RUNS,
            "per_run_band_by_sixth_tv_at_most": BAND_TV_CAP,
            "per_run_maximum_alive_band_mass_error_at_most": BAND_MASS_CAP,
            "per_run_absolute_sixth_clutter_one_mass_error_at_most": CLUTTER_ONE_CAP,
            "per_run_absolute_last_clutter_one_mass_error_at_most": CLUTTER_ONE_CAP,
        },
        "interpretation": "independent 17-detection baseline only; no 19-detection calibration claim",
        "holdout": "closed; no new sampler move or high-particle run",
    }


def _inputs(reference_path: Path, ledger_path: Path) -> tuple[Any, np.ndarray, float]:
    reference = validated_reference(reference_path)
    entries = _screen_order(_ledger(ledger_path))
    selected = entries[1]
    if (selected["variant_index"] != 1
            or selected["detection_sha256"] != DETECTION_SHA256
            or tuple(selected["cardinalities"]) != CARDINALITIES):
        raise ValueError("WP11x observation differs from frozen comparison")
    case = _checked_case(selected)
    if (_case_metadata(case)["detections_sha256"] != DETECTION_SHA256
            or tuple(case.model.sequence.cardinalities) != CARDINALITIES):
        raise ValueError("regenerated observation differs from frozen comparison")
    direct = reference["selected_checks"]["1.0"]["direct"]
    if (direct["beta"] != 1.0 or direct["minimum_raw_cell"] < -1e-10
            or direct["maximum_relative_imaginary_residual"] > 1e-8
            or direct["raw_mass_error"] > 1e-8):
        raise ValueError("WP11x beta-one reference is numerically invalid")
    exact = np.asarray(direct["joint"], dtype=float)
    if (exact.shape != (33, 4, 3) or np.min(exact) < 0
            or not np.isclose(exact.sum(), 1, atol=1e-10)):
        raise ValueError("WP11x beta-one reference has invalid cells")
    return case, exact, float(direct["log_partition"])


def _bands(joint: np.ndarray) -> np.ndarray:
    return np.stack((joint[:21].sum(axis=0), joint[21], joint[22:].sum(axis=0)))


def _metrics(joint: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    delta = joint - exact
    band_delta = _bands(joint) - _bands(exact)
    sixth = delta.sum(axis=(0, 2))
    last = delta.sum(axis=(0, 1))
    return {
        "full_joint_tv": float(np.abs(delta).sum() / 2),
        "band_by_sixth_tv": float(np.abs(band_delta.sum(axis=2)).sum() / 2),
        "maximum_alive_band_mass_error": float(np.max(np.abs(band_delta.sum(axis=(1, 2))))),
        "maximum_joint_cell_error": float(np.max(np.abs(delta))),
        "alive_total_tv": float(np.abs(delta.sum(axis=(1, 2))).sum() / 2),
        "alive_band_mass_error": band_delta.sum(axis=(1, 2)).tolist(),
        "band_by_sixth_mass_error": band_delta.sum(axis=2).tolist(),
        "sixth_clutter_mass_error": sixth.tolist(),
        "last_clutter_mass_error": last.tolist(),
        "sixth_clutter_mean_error": float(np.dot(np.arange(4), sixth)),
        "last_clutter_mean_error": float(np.dot(np.arange(3), last)),
    }


def _empirical_joint(states: np.ndarray) -> np.ndarray:
    if states.shape != (N_PARTICLES, 8, 4):
        raise ValueError("unexpected particle shape")
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    sixth = CARDINALITIES[5] - np.count_nonzero(states[:, 5] >= 0, axis=1)
    last = CARDINALITIES[7] - np.count_nonzero(states[:, 7] >= 0, axis=1)
    cells = np.ravel_multi_index((alive, sixth, last), (33, 4, 3))
    return np.bincount(cells, minlength=33 * 4 * 3).reshape(33, 4, 3) / N_PARTICLES


def _run_path(run_dir: Path, seed: int) -> Path:
    return run_dir / f"{seed}.json"


def run_one(run_dir: Path, reference_path: Path, ledger_path: Path, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed outside frozen schedule")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    case, exact, log_z = _inputs(reference_path, ledger_path)
    controls = _controls(seed)
    path = _run_path(run_dir, seed)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["protocol"] != _protocol() or record["seed"] != seed
                or record["config"] != asdict(controls)):
            raise ValueError("run checkpoint differs from frozen protocol")
        return record
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap is below the frozen cap")
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
    joint = _empirical_joint(sampled.associations)
    record = {
        "experiment": "wp11y_independent_absolute_sampler_run",
        "protocol": _protocol(), "seed": seed, "config": asdict(controls),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - log_z,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact),
        "stage_diagnostics": [stage.to_dict() for stage in sampled.stages],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _save(path, record)
    print(f"{seed}: {record['cpu_seconds']:.1f}s CPU, "
          f"joint TV {record['errors']['full_joint_tv']:.5f}", flush=True)
    return record


def _iid_q95(exact: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(IID_SEED)
    draws = rng.multinomial(N_PARTICLES, exact.ravel(), size=IID_REPLICATES)
    values: dict[str, list[float]] = {key: [] for key in (
        "full_joint_tv", "band_by_sixth_tv", "maximum_alive_band_mass_error",
        "maximum_joint_cell_error", "sixth_clutter_one_mass_error",
        "last_clutter_one_mass_error",
    )}
    for draw in draws:
        metrics = _metrics(draw.reshape(exact.shape) / N_PARTICLES, exact)
        for key, series in values.items():
            value = (abs(metrics["sixth_clutter_mass_error"][1])
                     if key == "sixth_clutter_one_mass_error"
                     else abs(metrics["last_clutter_mass_error"][1])
                     if key == "last_clutter_one_mass_error" else metrics[key])
            series.append(value)
    return {key: float(np.quantile(series, .95)) for key, series in values.items()}


def _decision(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [runs[str(seed)]["errors"] for seed in SEEDS]
    keys = ("full_joint_tv", "band_by_sixth_tv", "maximum_alive_band_mass_error",
            "maximum_joint_cell_error")
    medians = {key: float(median(row[key] for row in rows)) for key in keys}
    per_run_mode = {
        str(seed): (row["band_by_sixth_tv"] <= BAND_TV_CAP
                    and row["maximum_alive_band_mass_error"] <= BAND_MASS_CAP
                    and abs(row["sixth_clutter_mass_error"][1]) <= CLUTTER_ONE_CAP
                    and abs(row["last_clutter_mass_error"][1]) <= CLUTTER_ONE_CAP)
        for seed, row in zip(SEEDS, rows, strict=True)
    }
    gates = {
        "median_full_joint": medians["full_joint_tv"] <= FULL_TV_CAP,
        "median_band_by_sixth": medians["band_by_sixth_tv"] <= BAND_TV_CAP,
        "median_maximum_band_mass": medians["maximum_alive_band_mass_error"] <= BAND_MASS_CAP,
        "median_maximum_cell": medians["maximum_joint_cell_error"] <= MAX_CELL_MEDIAN_CAP,
        "replicated_modes": sum(per_run_mode.values()) >= MIN_REPLICATED_RUNS,
    }
    return {"medians": medians, "per_run_mode_within_limits": per_run_mode,
            "gates": gates, "passed": all(gates.values())}


def aggregate(run_dir: Path, reference_path: Path, ledger_path: Path,
              output: Path) -> dict[str, Any]:
    _, exact, log_z = _inputs(reference_path, ledger_path)
    runs = {}
    for seed in SEEDS:
        record = json.loads(_run_path(run_dir, seed).read_text(encoding="utf-8"))
        if (record["protocol"] != _protocol() or record["seed"] != seed
                or record["config"] != asdict(_controls(seed))
                or record["cpu_seconds"] > CPU_CAP
                or record["n_tempering_steps"] != len(record["stage_diagnostics"])):
            raise ValueError("run checkpoint differs from frozen protocol")
        joint = np.asarray(record["joint_alive_by_sixth_by_last_clutter"])
        if (joint.shape != (33, 4, 3) or np.min(joint) < 0
                or not np.isclose(joint.sum(), 1, atol=1e-12)
                or not np.allclose(joint * N_PARTICLES, np.rint(joint * N_PARTICLES),
                                   atol=1e-9)):
            raise ValueError("invalid empirical joint")
        if not np.isclose(record["stage_diagnostics"][-1]["beta"], 1, atol=1e-12):
            raise ValueError("sampler did not reach beta one")
        for key, value in _metrics(joint, exact).items():
            if not np.allclose(value, record["errors"][key], atol=1e-12):
                raise ValueError(f"saved metric {key} changed")
        runs[str(seed)] = record
    decision = _decision(runs)
    result = {
        "experiment": "wp11y_independent_absolute_sampler",
        "protocol": _protocol(), "reference_log_partition": log_z,
        "exact_alive_bands": _bands(exact).sum(axis=(1, 2)).tolist(),
        "exact_sixth_clutter": exact.sum(axis=(0, 2)).tolist(),
        "exact_last_clutter": exact.sum(axis=(0, 1)).tolist(),
        "iid_q95_descriptive": _iid_q95(exact),
        "runs": runs, "comparison": decision,
        "status": ("independent_17_detection_baseline_within_frozen_limits"
                   if decision["passed"] else "independent_17_detection_baseline_failed_limits"),
    }
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "sampler", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference", type=Path,
                        default=Path("results/wp11x_frame_six_exact_reference.json"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11x_geometry_ledger.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11y_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11y_independent_absolute_sampler.json"))
    args = parser.parse_args()
    if args.case == "protocol":
        _inputs(args.reference, args.ledger)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "sampler":
        if args.seed is None:
            parser.error("sampler requires --seed")
        run_one(args.run_dir, args.reference, args.ledger, args.seed)
    else:
        if args.seed is not None:
            parser.error("aggregate takes no seed")
        print(aggregate(args.run_dir, args.reference, args.ledger, args.output)["status"])


if __name__ == "__main__":
    main()
