"""Bind the frozen WP11al 24-job rule to the validated WP11ap cohort."""

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
from .wp11aa_tempered_suffix_pg import _controls, _decision, _joint, _metrics
from .wp11ag_global_slot_comparison import _candidate
from .wp11ap_positive_joint_cohort import (
    LEDGER_SHA256, _checked_case, _ledger, _protocol as _reference_protocol,
    _selection,
)
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11w_balanced_exact_reference import _accepted, _masses

MANIFEST_SHA256 = "2e2e20acbb84524e5022d9eaa7ab40202420bad660214426dbcf0ecfca000e68"
WP11AL_RULE_SHA256 = "693a34ae0a5a3da8f149e9087c8ee38441b748b1845596905a7a9b5afe2592ad"
CASE_BINDINGS = {
    "crowded_18": {
        "stratum": "18", "index": 36, "base_index": 52,
        "observation_sha256": "fab8353efa279f0f52eee0fc24a70617bc9ba21b58f13a9c8ddee9d00217449e",
        "screen_sha256": "a04f4c1c7c6afa7560242455e7c89b12bad28144e5fb0ec44d989d220f6bfd1a",
        "reference_sha256": "2a4ed5be9e9ce3d41aca73dfe53f9e4044a489ea2394ac784ed4d5d49f6a8f17",
        "seeds": (1800, 1801, 1802, 1803),
    },
    "crowded_19_first": {
        "stratum": "19", "index": 207, "base_index": 424,
        "observation_sha256": "bc8e6e9afc8d5b702556ddaf643c8493feac4fbd9bd6f73e6beaa1817820be30",
        "screen_sha256": "1a0d6c660dad4884fae8302162b96676fc0777dcc7b27a1d3f87f3d133e6d40a",
        "reference_sha256": "6f1e229ee7a00c5b062b58a0f31b2f3b6b41655b40b00be9f132977b25179833",
        "seeds": (1810, 1811, 1812, 1813),
    },
    "crowded_19_second": {
        "stratum": "19", "index": 9, "base_index": 35,
        "observation_sha256": "21a05378a41892e3c6b444c4a67b6851157f16da7164c0b0e24a2d24537f4dfc",
        "screen_sha256": "959105438ab867d04c8783c9c649367327b36e9bae163ff1bd69f4b84ddc205e",
        "reference_sha256": "5ac3af551e941484a8999a57e73ba5789b73603a2106e69e0e8bf25926adce7a",
        "seeds": (1820, 1821, 1822, 1823),
    },
}
N_PARTICLES = 2_048
CPU_CAP = 750
MAX_PER_SEED_CPU_RATIO = 1.50
MAX_MEDIAN_CPU_RATIO = 1.25
MIN_POOLED_WINS = 10
MIN_POOLED_MODES = 10


def _schedule() -> list[tuple[str, str, int]]:
    return [(name, arm, seed) for name, binding in CASE_BINDINGS.items()
            for seed in binding["seeds"] for arm in ("control", "candidate")]


def _config(arm: str, seed: int) -> Any:
    if arm == "control":
        return _controls(seed, "control")
    if arm == "candidate":
        return _candidate(seed)
    raise ValueError("arm outside frozen comparison")


def _protocol() -> dict[str, Any]:
    return {
        "wp11al_rule_sha256": WP11AL_RULE_SHA256,
        "wp11ap_manifest_sha256": MANIFEST_SHA256,
        "wp11ap_geometry_sha256": LEDGER_SHA256,
        "bindings": {name: {**binding, "seeds": list(binding["seeds"])}
                     for name, binding in CASE_BINDINGS.items()},
        "schedule": [list(job) for job in _schedule()],
        "control_configuration_seed_1800": asdict(_config("control", 1800)),
        "candidate_configuration_seed_1800": asdict(_config("candidate", 1800)),
        "n_particles": N_PARTICLES,
        "adaptive_tempering": True,
        "mh_sweeps": 8, "final_mh_sweeps": 8,
        "whole_slot_fraction_at_beta_ge_0_65": .20,
        "process_cpu_cap_seconds_per_job": CPU_CAP,
        "thread_env": {key: "1" for key in THREAD_ENV},
        "case_gate": {
            "full_joint_tv_wins_at_least": 3,
            "median_full_joint_tv_at_most": .12,
            "median_full_joint_tv_over_control_at_most": .90,
            "median_band_by_sixth_tv_at_most": .10,
            "median_maximum_alive_band_error_at_most": .08,
            "median_maximum_joint_cell_error_at_most": .04,
            "median_regression_each_other_metric_at_most": .020,
            "simultaneous_mode_runs_at_least": 3,
            "per_run_mode_limits": {"band_by_sixth_tv": .10,
                                    "maximum_alive_band_error": .08,
                                    "sixth_clutter_one_absolute_error": .06,
                                    "last_clutter_one_absolute_error": .06},
            "each_candidate_over_control_cpu_at_most": MAX_PER_SEED_CPU_RATIO,
            "candidate_median_over_control_cpu_at_most": MAX_MEDIAN_CPU_RATIO,
        },
        "pooled_gate": {"full_joint_tv_wins_at_least": MIN_POOLED_WINS,
                        "simultaneous_mode_runs_at_least": MIN_POOLED_MODES,
                        "matched_pairs": 12},
        "decision": "run all 24 despite poor accuracy; cap, hash, configuration or "
                    "numerical mismatch stops comparison; all three cases and "
                    "both pooled gates must pass",
        "holdout": "WP8 closed; development only",
    }


def _pinned(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen input hash changed: {path}")
    return json.loads(raw)


def _inputs(manifest_path: Path, ledger_path: Path, screen_dir: Path,
            reference_dir: Path, rule_path: Path) -> dict[str, tuple[Any, np.ndarray, float]]:
    if hashlib.sha256(rule_path.read_bytes()).hexdigest() != WP11AL_RULE_SHA256:
        raise ValueError("frozen WP11al rule changed")
    manifest = _pinned(manifest_path, MANIFEST_SHA256)
    ledger = _ledger(ledger_path)
    next_entry, selected = _selection(ledger, screen_dir)
    if (next_entry is not None or manifest["geometry_sha256"] != LEDGER_SHA256
            or manifest["protocol"] != _reference_protocol()
            or manifest["status"] != (
                "three_independently_validated_references_no_sampler_holdout_closed"
            ) or set(manifest["selected"]) != {"18", "19"}
            or len(manifest["selected"]["18"]) != 1
            or len(manifest["selected"]["19"]) != 2):
        raise ValueError("WP11ap manifest is not the complete frozen cohort")
    rows = {name: row for name, row in zip(CASE_BINDINGS, (
        *manifest["selected"]["18"], *manifest["selected"]["19"]
    ), strict=True)}
    cases = {}
    for name, binding in CASE_BINDINGS.items():
        row = rows[name]
        entry = next((entry for entry in selected[binding["stratum"]]
                      if entry["index"] == binding["index"]), None)
        if (entry is None or row["index"] != binding["index"]
                or row["base_index"] != binding["base_index"]
                or row["detection_sha256"] != binding["observation_sha256"]
                or row["screen_sha256"] != binding["screen_sha256"]
                or row["reference_sha256"] != binding["reference_sha256"]
                or row["cardinalities"] != entry["cardinalities"]):
            raise ValueError(f"{name} differs from frozen case binding")
        screen = _pinned(screen_dir / f"{row['index']}.json", row["screen_sha256"])
        reference = _pinned(reference_dir / f"{row['index']}.json",
                            row["reference_sha256"])
        if (screen["status"] != "balance_accepted_validate"
                or screen["detection_sha256"] != row["detection_sha256"]
                or reference["status"] != "validated_reference_no_sampler"
                or reference["ledger_sha256"] != LEDGER_SHA256
                or reference["index"] != row["index"]
                or reference["selected_detection_sha256"]
                != row["detection_sha256"]
                or set(reference["checks"]) != {"0.0", "0.5", "1.0"}):
            raise ValueError(f"{name} screen or reference changed")
        case = _checked_case(entry)
        endpoint = reference["checks"]["1.0"]["paired"]
        exact = np.asarray(endpoint["joint"], dtype=float)
        cards = tuple(case.model.sequence.cardinalities)
        if (exact.shape != (33, cards[5] + 1, cards[7] + 1)
                or not np.isfinite(exact).all() or np.min(exact) < 0
                or abs(float(exact.sum()) - 1) > 1e-12
                or not _accepted(_masses(exact))
                or not np.allclose(
                    np.asarray(row["masses"]["alive_bands"]),
                    _masses(exact)["alive_bands"], atol=1e-12, rtol=0)
                or not np.isfinite(endpoint["log_partition"])):
            raise ValueError(f"{name} exact joint differs from pinned reference")
        cases[name] = (case, exact, float(endpoint["log_partition"]))
    return cases


def _path(run_dir: Path, name: str, arm: str, seed: int) -> Path:
    return run_dir / f"{name}_{arm}_{seed}.json"


def _validated_run(path: Path, name: str, arm: str, seed: int,
                   exact: np.ndarray) -> dict[str, Any]:
    saved = json.loads(path.read_text())
    config = asdict(_config(arm, seed))
    if (saved["protocol"] != _protocol() or saved["name"] != name
            or saved["arm"] != arm or saved["seed"] != seed
            or saved["config"] != config
            or not np.isfinite(saved["cpu_seconds"])
            or not 0 < saved["cpu_seconds"] <= CPU_CAP
            or not np.isfinite(saved["wall_seconds"])
            or not np.isfinite(saved["log_evidence_error"])
            or saved["n_tempering_steps"] != len(saved["stages"])
            or not saved["stages"]):
        raise ValueError(f"{path} differs from frozen run configuration")
    betas = [stage["beta"] for stage in saved["stages"]]
    if (not np.isfinite(betas).all() or any(b < 0 or b > 1 for b in betas)
            or any(a > b for a, b in zip(betas, betas[1:]))
            or abs(betas[-1] - 1) > 1e-12):
        raise ValueError(f"{path} has invalid tempering stages")
    if arm == "candidate" and any(
        stage["paired_slot_whole_slot_selected_fraction"] != 0
        for stage in saved["stages"] if stage["beta"] < .65
    ):
        raise ValueError(f"{path} activated whole-slot move before beta .65")
    joint = np.asarray(saved["joint_alive_by_sixth_by_last_clutter"], dtype=float)
    if (joint.shape != exact.shape or not np.isfinite(joint).all()
            or np.min(joint) < 0 or abs(float(joint.sum()) - 1) > 1e-12
            or not np.allclose(joint * N_PARTICLES,
                               np.rint(joint * N_PARTICLES), atol=1e-9, rtol=0)):
        raise ValueError(f"{path} has invalid full joint")
    for key, actual in _metrics(joint, exact).items():
        reported = np.asarray(saved["errors"][key])
        if (not np.isfinite(reported).all()
                or not np.allclose(actual, reported, atol=1e-12, rtol=0)):
            raise ValueError(f"{path} metric {key} differs from pinned joint")
    return saved


def _next_job(run_dir: Path, cases: dict[str, tuple[Any, np.ndarray, float]]) -> (
        tuple[str, str, int] | None):
    allowed = {_path(run_dir, *job) for job in _schedule()}
    if run_dir.exists() and any(path not in allowed for path in run_dir.glob("*.json")):
        raise ValueError("unlisted run checkpoint in frozen directory")
    pending = None
    for name, arm, seed in _schedule():
        path = _path(run_dir, name, arm, seed)
        if path.exists():
            if pending is not None:
                raise ValueError("run checkpoints are out of frozen order")
            _validated_run(path, name, arm, seed, cases[name][1])
        elif pending is None:
            pending = (name, arm, seed)
    return pending


def run_one(name: str, arm: str, seed: int, run_dir: Path,
            manifest: Path, ledger: Path, screens: Path, references: Path,
            rule: Path) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    cases = _inputs(manifest, ledger, screens, references, rule)
    if (name, arm, seed) != _next_job(run_dir, cases):
        raise ValueError("job is not next in frozen schedule")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_CAP + 5:
        raise RuntimeError("system CPU hard cap below frozen cap")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP + 5))
    case, exact, logz = cases[name]
    config = _config(arm, seed)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std,
        coordinate_std=case.coordinate_std, length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior, config=config,
    )
    joint = _joint(sampled.associations, tuple(case.model.sequence.cardinalities))
    result = {
        "experiment": "wp11aq_bound_wp11al_cohort_run", "protocol": _protocol(),
        "name": name, "arm": arm, "seed": seed, "config": asdict(config),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
        "n_tempering_steps": sampled.n_tempering_steps,
        "log_evidence_error": sampled.log_evidence_estimate - logz,
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": _metrics(joint, exact),
        "stages": [stage.to_dict() for stage in sampled.stages],
    }
    path = _path(run_dir, name, arm, seed)
    _save(path, result)
    _validated_run(path, name, arm, seed, exact)
    print(f"{name} {arm} {seed}: {result['cpu_seconds']:.1f}s CPU, "
          f"joint TV {result['errors']['full_joint_tv']:.5f}", flush=True)
    return result


def _case_decision(seeds: tuple[int, ...], controls: list[dict[str, Any]],
                   candidates: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = _decision("decoy_17", controls, candidates)
    comparison["per_seed_modes"] = dict(zip(
        map(str, seeds), comparison["per_seed_modes"].values(), strict=True
    ))
    control_cpu = median(row["cpu_seconds"] for row in controls)
    candidate_cpu = median(row["cpu_seconds"] for row in candidates)
    comparison["gates"]["median_cost"] = (
        candidate_cpu <= MAX_MEDIAN_CPU_RATIO * control_cpu
    )
    comparison["median_cpu_seconds"] = {"control": control_cpu,
                                        "candidate": candidate_cpu}
    comparison["passed"] = all(comparison["gates"].values())
    return comparison


def _cohort_decision(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["paired_full_joint_wins"] for row in comparisons.values())
    modes = sum(sum(row["per_seed_modes"].values()) for row in comparisons.values())
    gates = {"all_three_cases": all(row["passed"] for row in comparisons.values()),
             "pooled_full_joint_wins": wins >= MIN_POOLED_WINS,
             "pooled_simultaneous_modes": modes >= MIN_POOLED_MODES}
    return {"pooled_full_joint_wins": wins, "pooled_simultaneous_modes": modes,
            "gates": gates, "passed": all(gates.values())}


def aggregate(run_dir: Path, output: Path, manifest: Path, ledger: Path,
              screens: Path, references: Path, rule: Path) -> dict[str, Any]:
    cases = _inputs(manifest, ledger, screens, references, rule)
    if _next_job(run_dir, cases) is not None:
        raise ValueError("all 24 frozen jobs are required before a decision")
    rows, comparisons = {}, {}
    for name, binding in CASE_BINDINGS.items():
        controls, candidates, per_seed = [], [], {}
        for seed in binding["seeds"]:
            pair = {}
            for arm in ("control", "candidate"):
                path = _path(run_dir, name, arm, seed)
                saved = _validated_run(path, name, arm, seed, cases[name][1])
                summary = {"cpu_seconds": saved["cpu_seconds"],
                           "errors": saved["errors"]}
                (controls if arm == "control" else candidates).append(summary)
                pair[arm] = {**summary,
                             "run_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "log_evidence_error": saved["log_evidence_error"],
                             "n_tempering_steps": saved["n_tempering_steps"]}
            per_seed[str(seed)] = pair
        comparison = _case_decision(binding["seeds"], controls, candidates)
        comparisons[name] = comparison
        rows[name] = {"index": binding["index"],
                      "cardinalities": list(cases[name][0].model.sequence.cardinalities),
                      "runs": per_seed, "comparison": comparison}
    cohort = _cohort_decision(comparisons)
    result = {"experiment": "wp11aq_bound_wp11al_cohort_comparison",
              "protocol": _protocol(), "cases": rows, "cohort_comparison": cohort,
              "status": ("frozen_24_job_cohort_gate_passed_holdout_closed"
                         if cohort["passed"] else "frozen_24_job_cohort_gate_failed_stop")}
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("existing aggregate differs from frozen comparison")
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("protocol", "next", "run", "aggregate"),
                        required=True)
    parser.add_argument("--observation", choices=tuple(CASE_BINDINGS))
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/wp11ap_crowded_cohort_manifest.json"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/wp11ap_geometry_ledger.json"))
    parser.add_argument("--screens", type=Path,
                        default=Path("results/wp11ap_screens"))
    parser.add_argument("--references", type=Path,
                        default=Path("results/wp11ap_references"))
    parser.add_argument("--rule", type=Path,
                        default=Path("docs/wp11al_conditional_cohort_gate.md"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/wp11aq_bound_cohort_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11aq_bound_cohort_comparison.json"))
    args = parser.parse_args()
    inputs = (args.manifest, args.ledger, args.screens, args.references, args.rule)
    if args.case == "protocol":
        _inputs(*inputs)
        print(json.dumps(_protocol(), indent=2))
    elif args.case == "next":
        cases = _inputs(*inputs)
        print(json.dumps({"next": _next_job(args.run_dir, cases)}))
    elif args.case == "run":
        if args.observation is None or args.arm is None or args.seed is None:
            parser.error("run requires observation, arm and seed")
        run_one(args.observation, args.arm, args.seed, args.run_dir, *inputs)
    else:
        if any(value is not None for value in (args.observation, args.arm, args.seed)):
            parser.error("aggregate takes no observation, arm or seed")
        print(aggregate(args.run_dir, args.output, *inputs)["status"])


if __name__ == "__main__":
    main()
