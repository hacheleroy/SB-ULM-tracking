"""Frozen stage-level absolute audit on the WP11x balanced decoy observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .existence_smc import sample_slot_existence_posterior
from .wp7_convergence_audit import _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11v_tempering_mode_audit import _ExactComputer, _numeric_ok, _StageObserver
from .wp11y_independent_absolute_sampler import (
    CARDINALITIES,
    CPU_CAP,
    DETECTION_SHA256,
    N_PARTICLES,
    REFERENCE_SHA256,
    SEEDS,
    _controls,
    _empirical_joint,
    _inputs,
    _metrics,
)
from .wp11y_independent_absolute_sampler import _protocol as _wp11y_protocol

WP11Y_SHA256 = "3af1e36390c5700d150291e5493608009f319498f44f9ec4164d89654c795f50"
EXACT_CPU_CAP = 1_000
IID_SEED = 56_789
IID_REPLICATES = 1_000
METRICS = ("alive_band", "sixth_clutter_one", "last_clutter_one",
           "band_by_sixth", "band_by_last")
FLOORS = {"alive_band": .06, "sixth_clutter_one": .06,
          "last_clutter_one": .06, "band_by_sixth": .08, "band_by_last": .08}


def _protocol() -> dict[str, Any]:
    return {
        "wp11y_result_sha256": WP11Y_SHA256,
        "wp11x_reference_sha256": REFERENCE_SHA256,
        "detection_sha256": DETECTION_SHA256,
        "cardinalities": list(CARDINALITIES),
        "same_seeds_and_controls_as_wp11y": list(SEEDS),
        "n_particles": N_PARTICLES, "sampler_cpu_cap_per_seed": CPU_CAP,
        "exact_cpu_cap_per_seed": EXACT_CPU_CAP,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "stage_observer": ["prior_unweighted", "likelihood_weighted",
                           "resampled", "rejuvenated"],
        "exact": "direct squarefree four-slot joint at beta zero and every observed stage beta",
        "numerical": {"min_raw_cell": -1e-10, "max_imaginary_residual": 1e-8,
                    "max_mass_error": 1e-8, "endpoint_joint_tv": 1e-7,
                    "endpoint_log_partition_error": 1e-8},
        "iid": {"seed_base": IID_SEED, "replicates": IID_REPLICATES,
                "sample_size": N_PARTICLES, "per_stage_seed": "base + 128*seed + step"},
        "onset": {
            "metrics": list(METRICS), "absolute_floor": FLOORS,
            "threshold": "max(floor, 2 * exact-IID q95 at current beta)",
            "sustained": "first two consecutive rejuvenated stages above threshold",
            "operation": "carried if prior exceeds previous-beta threshold, else first current-beta weighting/resampling/rejuvenation above threshold",
            "no_onset": "report none, peak error and transient exceedances",
        },
        "interpretation": "diagnostic of already exposed WP11y seeds; no new move gate",
        "holdout": "closed; no 19-detection or high-particle inference",
    }


def _wp11y(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != WP11Y_SHA256:
        raise ValueError("WP11y absolute comparison bytes changed")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved["protocol"] != _wp11y_protocol()
            or saved["status"] != "independent_17_detection_baseline_failed_limits"
            or set(saved["runs"]) != {str(seed) for seed in SEEDS}):
        raise ValueError("WP11y baseline is not the frozen four-seed failure")
    return saved


def _run_path(run_dir: Path, seed: int) -> Path:
    return run_dir / f"{seed}.json"


def _exact_path(exact_dir: Path, seed: int) -> Path:
    return exact_dir / f"{seed}.json"


def run_seed(run_dir: Path, reference_path: Path, ledger_path: Path,
             wp11y_path: Path, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed outside frozen schedule")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    previous = _wp11y(wp11y_path)["runs"][str(seed)]
    case, _, log_z = _inputs(reference_path, ledger_path)
    controls = _controls(seed)
    path = _run_path(run_dir, seed)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["protocol"] != _protocol() or saved["seed"] != seed:
            raise ValueError("stage checkpoint differs from frozen protocol")
        return saved
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen sampler cap")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP + 5))
    observer = _StageObserver(CARDINALITIES)
    started = time.process_time()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std, coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior, config=controls,
        observer=observer,
    )
    joint = _empirical_joint(sampled.associations)
    if (not np.array_equal(joint, np.asarray(previous[
        "joint_alive_by_sixth_by_last_clutter"]))
            or not np.isclose(sampled.log_evidence_estimate - log_z,
                              previous["log_evidence_error"], atol=1e-10)
            or sampled.n_tempering_steps != previous["n_tempering_steps"]
            or not np.array_equal(joint, np.asarray(observer.records[-1]["rejuvenated"]))):
        raise ValueError("passive observer changed the pinned WP11y final run")
    result = {
        "experiment": "wp11z_balanced_stage_run", "protocol": _protocol(),
        "seed": seed, "config": asdict(controls),
        "cpu_seconds": time.process_time() - started,
        "stages": [stage.to_dict() for stage in sampled.stages],
        "joint_trace": observer.records,
        "status": "identical_final_joint_to_wp11y",
    }
    _save(path, result)
    print(f"{seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"{len(observer.records)} stage traces, final joint identical", flush=True)
    return result


def _trace(path: Path, seed: int, previous: dict[str, Any]) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if (trace["protocol"] != _protocol() or trace["seed"] != seed
            or trace["config"] != asdict(_controls(seed))
            or trace["status"] != "identical_final_joint_to_wp11y"
            or trace["cpu_seconds"] > CPU_CAP
            or len(trace["joint_trace"]) != len(trace["stages"])
            or len(trace["stages"]) != previous["n_tempering_steps"]):
        raise ValueError("sampler trace differs from frozen WP11y replay")
    final = np.asarray(trace["joint_trace"][-1]["rejuvenated"])
    if not np.array_equal(final, np.asarray(previous[
        "joint_alive_by_sixth_by_last_clutter"])):
        raise ValueError("stage trace final joint differs from WP11y")
    for stage, diagnostic in zip(trace["joint_trace"], trace["stages"], strict=True):
        if (stage["step"] != diagnostic["step"] or stage["beta"] != diagnostic["beta"]
                or not 0 < stage["beta"] <= 1):
            raise ValueError("stage index or beta differs from sampler diagnostics")
        for name in _protocol()["stage_observer"]:
            joint = np.asarray(stage[name])
            if joint.shape != (33, 4, 3) or not np.isclose(joint.sum(), 1, atol=1e-10):
                raise ValueError("stage joint is invalid")
    if not np.isclose(trace["stages"][-1]["beta"], 1, atol=1e-12):
        raise ValueError("sampler did not reach beta one")
    return trace


def run_exact(exact_dir: Path, run_dir: Path, reference_path: Path,
              ledger_path: Path, wp11y_path: Path, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed outside frozen schedule")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    previous = _wp11y(wp11y_path)["runs"][str(seed)]
    trace = _trace(_run_path(run_dir, seed), seed, previous)
    case, _, _ = _inputs(reference_path, ledger_path)
    path = _exact_path(exact_dir, seed)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["protocol"] != _protocol() or saved["seed"] != seed:
            raise ValueError("exact checkpoint differs from frozen protocol")
        return saved
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < EXACT_CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen exact cap")
    resource.setrlimit(resource.RLIMIT_CPU, (EXACT_CPU_CAP, EXACT_CPU_CAP + 5))
    started = time.process_time()
    computer = _ExactComputer(case.model, (5, 7))
    betas = [0.0] + [stage["beta"] for stage in trace["joint_trace"]]
    checks = [computer.compute(beta) for beta in betas]
    for check in checks:
        if not _numeric_ok(check):
            raise ValueError(f"exact numerical envelope failed at beta={check['beta']}")
    pinned = json.loads(reference_path.read_text(encoding="utf-8"))["selected_checks"]
    for index, key in ((0, "0.0"), (-1, "1.0")):
        current = checks[index]
        saved = pinned[key]["direct"]
        if (abs(current["log_partition"] - saved["log_partition"]) > 1e-8
                or np.abs(np.asarray(current["joint"]) - np.asarray(saved["joint"])
                ).sum() / 2 > 1e-7):
            raise ValueError(f"exact endpoint differs from pinned beta {key}")
    result = {
        "experiment": "wp11z_balanced_stage_exact", "protocol": _protocol(),
        "seed": seed, "betas": betas, "checks": checks,
        "cpu_seconds": time.process_time() - started,
        "status": "all_observed_beta_exact_checks_passed",
    }
    _save(path, result)
    print(f"{seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"{len(checks)} exact stage targets", flush=True)
    return result


def _values(observed: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    errors = _metrics(observed, exact)
    d = observed - exact
    bands = np.stack((d[:21].sum(axis=0), d[21], d[22:].sum(axis=0)))
    return {
        "full_joint_tv": errors["full_joint_tv"],
        "alive_band": errors["maximum_alive_band_mass_error"],
        "sixth_clutter_one": abs(errors["sixth_clutter_mass_error"][1]),
        "last_clutter_one": abs(errors["last_clutter_mass_error"][1]),
        "band_by_sixth": errors["band_by_sixth_tv"],
        "band_by_last": float(np.abs(bands.sum(axis=1)).sum() / 2),
        "alive_band_signed": errors["alive_band_mass_error"],
        "sixth_clutter_one_signed": errors["sixth_clutter_mass_error"][1],
        "last_clutter_one_signed": errors["last_clutter_mass_error"][1],
    }


def _iid_q95(joint: np.ndarray, seed: int, step: int) -> dict[str, float]:
    rng = np.random.default_rng(IID_SEED + 128 * seed + step)
    draws = rng.multinomial(N_PARTICLES, joint.ravel(), size=IID_REPLICATES)
    differences = draws.reshape((IID_REPLICATES,) + joint.shape) / N_PARTICLES - joint
    bands = np.stack((differences[:, :21].sum(axis=1), differences[:, 21],
                      differences[:, 22:].sum(axis=1)), axis=1)
    values = {
        "alive_band": np.max(np.abs(bands.sum(axis=(2, 3))), axis=1),
        "sixth_clutter_one": np.abs(differences.sum(axis=(1, 3))[:, 1]),
        "last_clutter_one": np.abs(differences.sum(axis=(1, 2))[:, 1]),
        "band_by_sixth": np.abs(bands.sum(axis=3)).sum(axis=(1, 2)) / 2,
        "band_by_last": np.abs(bands.sum(axis=2)).sum(axis=(1, 2)) / 2,
    }
    return {key: float(np.quantile(value, .95)) for key, value in values.items()}


def _first_sustained(stages: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    for first, second in pairwise(stages):
        if (first["errors"]["rejuvenated"][metric] > first["limits"][metric]
                and second["errors"]["rejuvenated"][metric] > second["limits"][metric]):
            return {"step": first["step"], "beta": first["beta"],
                    "first_above_operation": first["first_above_operation"][metric]}
    return None


STAGE_COLUMNS = ("step", "beta", "full_joint_tv", "alive_band",
                 "sixth_clutter_one_signed", "last_clutter_one_signed",
                 "band_by_sixth", "band_by_last", "effective_sample_size",
                 "unique_parent_count", "unique_state_count",
                 "alive_band_limit", "sixth_clutter_one_limit",
                 "last_clutter_one_limit", "band_by_sixth_limit", "band_by_last_limit")


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    per_seed = {}
    for seed, record in result["per_seed"].items():
        onset_values = {}
        for metric, onset in record["onset"].items():
            if onset is None:
                continue
            stage = record["stages"][onset["step"] - 1]
            onset_values[metric] = {
                "limit": stage["limits"][metric],
                "by_operation": {name: stage["errors"][name][metric] for name in (
                    "prior_unweighted", "likelihood_weighted", "resampled", "rejuvenated"
                )},
            }
        curves = []
        for stage in record["stages"]:
            final = stage["errors"]["rejuvenated"]
            curves.append([
                stage["step"], stage["beta"], final["full_joint_tv"],
                final["alive_band"], final["sixth_clutter_one_signed"],
                final["last_clutter_one_signed"], final["band_by_sixth"],
                final["band_by_last"], stage["effective_sample_size"],
                stage["unique_parent_count"], stage["unique_state_count"],
                *(stage["limits"][key] for key in METRICS),
            ])
        per_seed[seed] = {
            **{key: record[key] for key in (
                "sampler_cpu_seconds", "n_stages", "onset", "peak_rejuvenated",
                "transient_weighted_exceedances", "median_signed_full_tv_change",
                "final_errors"
            )},
            "onset_operation_values": onset_values,
            "stage_curve_columns": list(STAGE_COLUMNS), "stage_curves": curves,
        }
    return {key: result[key] for key in (
        "experiment", "protocol", "exact_numerical", "status"
    )} | {"per_seed": per_seed}


def aggregate(exact_dir: Path, run_dir: Path, reference_path: Path,
              ledger_path: Path, wp11y_path: Path, output: Path,
              detail_output: Path | None = None) -> dict[str, Any]:
    previous = _wp11y(wp11y_path)
    _inputs(reference_path, ledger_path)
    per_seed = {}
    exact_numerical = {}
    for seed in SEEDS:
        trace = _trace(_run_path(run_dir, seed), seed, previous["runs"][str(seed)])
        saved = json.loads(_exact_path(exact_dir, seed).read_text(encoding="utf-8"))
        if (saved["protocol"] != _protocol() or saved["seed"] != seed
                or saved["status"] != "all_observed_beta_exact_checks_passed"
                or saved["cpu_seconds"] > EXACT_CPU_CAP
                or saved["betas"] != [0.0] + [stage["beta"] for stage in trace[
                    "joint_trace"
                ]] or len(saved["checks"]) != len(saved["betas"])):
            raise ValueError("exact stage checkpoint differs from frozen trace")
        exact = [np.asarray(check["joint"]) for check in saved["checks"]]
        if any(not _numeric_ok(check) for check in saved["checks"]):
            raise ValueError("exact stage numerical envelope failed")
        exact_numerical[str(seed)] = {
            "cpu_seconds": saved["cpu_seconds"],
            "max_imaginary_residual": max(check[
                "maximum_relative_imaginary_residual"] for check in saved["checks"]),
            "minimum_raw_cell": min(check["minimum_raw_cell"] for check in saved["checks"]),
            "max_mass_error": max(check["raw_mass_error"] for check in saved["checks"]),
            "max_clipping_tv": max(check["clipping_tv"] for check in saved["checks"]),
        }
        stages = []
        for i, (stage, diagnostic) in enumerate(zip(
            trace["joint_trace"], trace["stages"], strict=True
        )):
            target, prior = exact[i + 1], exact[i]
            q95 = _iid_q95(target, seed, stage["step"])
            previous_q95 = _iid_q95(prior, seed, stage["step"] - 1)
            limits = {key: max(FLOORS[key], 2 * q95[key]) for key in METRICS}
            previous_limits = {key: max(FLOORS[key], 2 * previous_q95[key])
                               for key in METRICS}
            errors = {
                "prior_unweighted": _values(np.asarray(stage["prior_unweighted"]), prior),
                **{name: _values(np.asarray(stage[name]), target) for name in (
                    "likelihood_weighted", "resampled", "rejuvenated"
                )},
            }
            first_above = {}
            for key in METRICS:
                if errors["prior_unweighted"][key] > previous_limits[key]:
                    first_above[key] = "carried_from_previous_stage"
                else:
                    first_above[key] = next((name for name in (
                        "likelihood_weighted", "resampled", "rejuvenated"
                    ) if errors[name][key] > limits[key]), None)
            stages.append({
                "step": stage["step"], "beta": stage["beta"],
                "previous_beta": saved["betas"][i],
                "errors": errors, "iid_q95": q95, "limits": limits,
                "first_above_operation": first_above,
                "effective_sample_size": diagnostic["effective_sample_size"],
                "unique_parent_count": diagnostic["unique_parent_count"],
                "unique_state_count": diagnostic["unique_state_count"],
            })
        onset = {key: _first_sustained(stages, key) for key in METRICS}
        peaks = {key: max(stage["errors"]["rejuvenated"][key] for stage in stages)
                 for key in METRICS}
        transient = {key: sum(stage["errors"]["likelihood_weighted"][key]
                              > stage["limits"][key]
                              and stage["errors"]["rejuvenated"][key]
                              <= stage["limits"][key] for stage in stages)
                     for key in METRICS}
        changes = {operation: float(median(
            stage["errors"][after]["full_joint_tv"]
            - stage["errors"][before]["full_joint_tv"]
            for stage in stages
        )) for operation, before, after in (
            ("weighting", "prior_unweighted", "likelihood_weighted"),
            ("resampling", "likelihood_weighted", "resampled"),
            ("rejuvenation", "resampled", "rejuvenated"),
        )}
        per_seed[str(seed)] = {
            "sampler_cpu_seconds": trace["cpu_seconds"],
            "n_stages": len(stages), "onset": onset, "peak_rejuvenated": peaks,
            "transient_weighted_exceedances": transient,
            "median_signed_full_tv_change": changes,
            "final_errors": stages[-1]["errors"]["rejuvenated"],
            "stages": stages,
        }
    result = {
        "experiment": "wp11z_balanced_stage_audit", "protocol": _protocol(),
        "exact_numerical": exact_numerical, "per_seed": per_seed,
        "status": "balanced_stage_diagnostic_completed_no_move_gate",
    }
    if detail_output is not None:
        _save(detail_output, result)
    summary = _compact(result)
    _save(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "sampler", "exact", "aggregate"),
                        required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference", type=Path,
                        default=Path("results/wp11x_frame_six_exact_reference.json"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11x_geometry_ledger.json"))
    parser.add_argument("--wp11y", type=Path,
                        default=Path("results/wp11y_independent_absolute_sampler.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/wp11z_runs"))
    parser.add_argument("--exact-dir", type=Path, default=Path("results/wp11z_exact"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11z_balanced_stage_summary.json"))
    parser.add_argument("--detail-output", type=Path,
                        default=Path("results/wp11z_balanced_stage_detail.json"))
    args = parser.parse_args()
    if args.case == "protocol":
        _inputs(args.reference, args.ledger)
        _wp11y(args.wp11y)
        print(json.dumps(_protocol(), indent=2))
        return
    if args.case == "aggregate":
        if args.seed is not None:
            parser.error("aggregate takes no seed")
        print(aggregate(args.exact_dir, args.run_dir, args.reference, args.ledger,
                        args.wp11y, args.output, args.detail_output)["status"])
        return
    if args.seed is None:
        parser.error("sampler and exact require --seed")
    if args.case == "sampler":
        run_seed(args.run_dir, args.reference, args.ledger, args.wp11y, args.seed)
    else:
        run_exact(args.exact_dir, args.run_dir, args.reference, args.ledger,
                  args.wp11y, args.seed)


if __name__ == "__main__":
    main()
