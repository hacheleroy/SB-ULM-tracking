"""Frozen eight-frame exact bridge for absolute alive/clutter mode calibration."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .data import DetectionSequence
from .existence_smc import sample_slot_existence_posterior
from .slot_existence import (
    ABSENT,
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
    slot_existence_support_size,
)
from .wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR, _ScalingCase
from .wp7_convergence_audit import _case_metadata, _large_run_summary, _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls

OBSERVATION_SHA256 = "9b99c281f85c2265a6c10d333379e90eb80bc928b273d9d4c76bdf76f735e77e"
SUPPORT_SIZE = 667_989
CARDINALITIES = (1, 0, 2, 2, 1, 2, 1, 0)
CLUTTER_RATES = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
FRAME_INDEX = 5
MIN_MODE_MASS = 0.10
SEEDS = (900, 901, 902, 903, 904, 905)
PARTICLES = (512, 2_048)
IID_REPLICATES = 500
IID_SEED = 30_303
TEMPERED_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
REFERENCE_CPU_CAP = 1_800
RUN_CPU_CAPS = {512: 300, 2048: 900}


def _case(clutter_rate: float) -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(8, dtype=float) * 0.001,
        detections=(
            np.asarray([[-0.018, 0.001]]),
            np.empty((0, 2)),
            np.asarray([[-0.008, 0.002], [0.010, -0.001]]),
            np.asarray([[-0.003, 0.001], [0.005, -0.002]]),
            np.asarray([[0.000, 0.001]]),
            np.asarray([[-0.004, 0.001], [0.004, 0.000]]),
            np.asarray([[0.011, -0.001]]),
            np.empty((0, 2)),
        ),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=2, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.55, expected_clutter_per_frame=clutter_rate,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model, truth_mask=np.zeros((8, 2), dtype=bool),
        localization_std=0.005, coordinate_std=(0.02, 0.01), length_scale=0.003,
    )


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "absolute eight-frame mode-weight calibration; no move-selection gate",
        "sampler": "unchanged WP11g 10% four-path PG mixture",
        "observation_sha256": OBSERVATION_SHA256,
        "support_size": SUPPORT_SIZE,
        "cardinalities": list(CARDINALITIES),
        "clutter_rate_grid": list(CLUTTER_RATES),
        "mode_A": "alive slot-frames <= 10 and sixth-frame clutter >= 1",
        "mode_B": "alive slot-frames >= 11 and sixth-frame clutter == 0",
        "other_mode": "all remaining support",
        "bridge_selection": (
            "maximize min(exact mass A, exact mass B) over the fixed grid; "
            "tie goes to the smaller clutter rate; valid iff both masses >= 0.10"
        ),
        "reference_before_sampler": True,
        "tempered_beta_grid": list(TEMPERED_GRID),
        "iid_replicates_per_particle_count": IID_REPLICATES,
        "iid_seed": IID_SEED,
        "seeds": list(SEEDS), "particles": list(PARTICLES),
        "run_order": "reference, 512 seeds 900..905, then 2048 seeds 900..905",
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap": REFERENCE_CPU_CAP,
        "per_run_cpu_caps": {str(k): v for k, v in RUN_CPU_CAPS.items()},
        "interpretation": (
            "compare absolute mode A/B signed and joint TV errors to an exact target "
            "and iid multinomial baseline; no 8192 run or untouched holdout"
        ),
    }


def _metadata() -> dict[str, Any]:
    case = _case(CLUTTER_RATES[0])
    info = _case_metadata(case)
    if (info["detections_sha256"] != OBSERVATION_SHA256
            or tuple(info["detection_cardinalities"]) != CARDINALITIES
            or slot_existence_support_size(case.model) != SUPPORT_SIZE):
        raise ValueError("the frozen bridge changed")
    return {"observation": info, "models": [
        _case_metadata(_case(rate)) for rate in CLUTTER_RATES
    ]}


def _load(output: Path) -> dict[str, Any]:
    protocol = _protocol()
    metadata = _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != metadata:
            raise ValueError("checkpoint belongs to another frozen bridge")
        return result
    result = {
        "experiment": "wp11l_exact_eight_frame_bridge",
        "protocol": protocol, "cases": metadata,
        "reference_grid": None, "selected_rate": None,
        "reference": None, "runs": {str(n): {} for n in PARTICLES},
        "comparison": None, "status": "in_progress",
    }
    _save(output, result)
    return result


def _counts(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    clutter = CARDINALITIES[FRAME_INDEX] - np.count_nonzero(
        states[:, FRAME_INDEX] >= 0, axis=1
    )
    return alive, clutter


def _mode(states: np.ndarray) -> np.ndarray:
    alive, clutter = _counts(states)
    mode = np.full(len(states), 2, dtype=np.int8)
    mode[(alive <= 10) & (clutter >= 1)] = 0
    mode[(alive >= 11) & (clutter == 0)] = 1
    return mode


def _distributions(states: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    alive, clutter = _counts(states)
    mode = _mode(states)
    return {
        "mode": np.bincount(mode, weights=weights, minlength=3).tolist(),
        "joint_alive_by_clutter": np.bincount(
            3 * alive + clutter, weights=weights, minlength=17 * 3
        ).reshape(17, 3).tolist(),
        "alive": np.bincount(alive, weights=weights, minlength=17).tolist(),
        "clutter": np.bincount(clutter, weights=weights, minlength=3).tolist(),
    }


def _errors(states: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    observed = _distributions(states, np.full(len(states), 1 / len(states)))
    p, q = np.asarray(observed["mode"]), np.asarray(exact["mode"])
    alive_p, alive_q = np.asarray(observed["alive"]), np.asarray(exact["alive"])
    joint_p = np.asarray(observed["joint_alive_by_clutter"])
    joint_q = np.asarray(exact["joint_alive_by_clutter"])
    clutter_p, clutter_q = np.asarray(observed["clutter"]), np.asarray(exact["clutter"])
    return {
        "mode_A_signed_error": float(p[0] - q[0]),
        "mode_B_signed_error": float(p[1] - q[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(p - q))),
        "three_mode_tv": float(np.abs(p - q).sum() / 2),
        "full_joint_tv": float(np.abs(joint_p - joint_q).sum() / 2),
        "alive_total_tv": float(np.abs(alive_p - alive_q).sum() / 2),
        "clutter_mean_error": float(np.dot(np.arange(3), clutter_p - clutter_q)),
    }


def _set_cpu_limit(seconds: int) -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < seconds + 5:
        raise RuntimeError("system hard CPU limit is below frozen cap")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))


def _reference_distribution(
    states: np.ndarray, log_base: np.ndarray, log_marginal: np.ndarray, beta: float
) -> tuple[dict[str, Any], float]:
    scores = log_base + beta * log_marginal
    normalizer = float(logsumexp(scores))
    return _distributions(states, np.exp(scores - normalizer)), normalizer


def run_reference(output: Path) -> dict[str, Any]:
    result = _load(output)
    if result["reference_grid"] is not None:
        print("exact bridge: reused", flush=True)
        return result
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    case = _case(CLUTTER_RATES[0])
    support, _ = enumerate_slot_existence_support(
        case.model, max_hypotheses=SUPPORT_SIZE
    )
    log_marginal = np.empty(len(support))
    log_prior = np.empty(len(support))
    for lo in range(0, len(support), 20_000):
        hi = min(lo + 20_000, len(support))
        block = support[lo:hi]
        log_marginal[lo:hi] = case.model.log_marginal_density(block)
        log_prior[lo:hi] = case.model.log_existence_prior(block)
    grid = []
    for rate in CLUTTER_RATES:
        model = _case(rate).model
        log_base = np.empty(len(support))
        for lo in range(0, len(support), 20_000):
            hi = min(lo + 20_000, len(support))
            log_base[lo:hi] = (
                log_prior[lo:hi]
                + model.log_count_density(support[lo:hi], include_poisson_void=False)
            )
        if not np.isclose(
            logsumexp(log_base) - rate * len(CARDINALITIES),
            model.base_log_normalizer, atol=1e-8,
        ):
            raise ValueError("enumerated base distribution is not normalized")
        distribution, log_normalizer = _reference_distribution(
            support, log_base, log_marginal, 1.0
        )
        grid.append({
            "clutter_rate": rate,
            "mode": distribution["mode"],
            "log_evidence": log_normalizer - rate * len(CARDINALITIES),
        })
    selected = min(grid, key=lambda x: (-min(x["mode"][:2]), x["clutter_rate"]))
    result["reference_grid"] = grid
    if min(selected["mode"][:2]) < MIN_MODE_MASS:
        result["status"] = "bridge_invalid_no_competing_modes"
        _save(output, result)
        print("exact bridge: no predeclared model retained competing modes", flush=True)
        return result
    selected_rate = selected["clutter_rate"]
    selected_model = _case(selected_rate).model
    selected_base = np.empty(len(support))
    for lo in range(0, len(support), 20_000):
        hi = min(lo + 20_000, len(support))
        selected_base[lo:hi] = (
            log_prior[lo:hi] + selected_model.log_count_density(
                support[lo:hi], include_poisson_void=False
            )
        )
    tempered = []
    for beta in TEMPERED_GRID:
        distribution, _ = _reference_distribution(
            support, selected_base, log_marginal, beta
        )
        tempered.append({"beta": beta, "mode": distribution["mode"]})
    exact, _ = _reference_distribution(support, selected_base, log_marginal, 1.0)
    probabilities = np.exp(
        selected_base + log_marginal - logsumexp(selected_base + log_marginal)
    )
    rng = np.random.default_rng(IID_SEED)
    baseline = {}
    for n in PARTICLES:
        draws = {key: [] for key in (
            "maximum_mode_absolute_error", "three_mode_tv", "full_joint_tv",
            "alive_total_tv", "clutter_mean_error",
        )}
        for _ in range(IID_REPLICATES):
            indices = rng.choice(len(support), size=n, p=probabilities)
            errors = _errors(support[indices], exact)
            for key, values in draws.items():
                values.append(abs(errors[key]))
        baseline[str(n)] = {
            key: {"median": float(np.median(vals)),
                  "q95": float(np.quantile(vals, .95))}
            for key, vals in draws.items()
        }
    result["selected_rate"] = selected_rate
    result["reference"] = {
        "support_size": len(support), "exact": exact,
        "log_evidence": selected["log_evidence"],
        "tempered_mode_curve": tempered,
        "iid_baseline": baseline,
        "cpu_seconds": time.process_time() - started,
    }
    _save(output, result)
    print(f"exact bridge: {len(support):,} states, rate {selected_rate}; "
          f"modes {selected['mode'][:2]}; "
          f"{result['reference']['cpu_seconds']:.1f}s CPU", flush=True)
    return result


class _BridgeObserver:
    """RNG-free stage masses and accepted transitions between exact macro modes."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def begin_stage(self, step: int, beta: float) -> None:
        self.records.append({
            "step": step, "beta": float(beta),
            "accepted_A_to_B": {"pg": 0, "other": 0},
            "accepted_B_to_A": {"pg": 0, "other": 0},
        })

    def record_resampling(
        self, associations: np.ndarray, weights: np.ndarray, ancestors: np.ndarray
    ) -> None:
        mode = _mode(associations)
        before = np.bincount(mode, weights=weights, minlength=3)
        after = np.bincount(mode[ancestors], minlength=3) / len(ancestors)
        self.records[-1]["weighted_mode"] = before.tolist()
        self.records[-1]["resampled_mode"] = after.tolist()
        self.records[-1]["resampling_mode_tv"] = float(np.abs(before - after).sum() / 2)

    def record_pair(
        self, before: np.ndarray, proposed: np.ndarray,
        accepted: np.ndarray, pg_selected: np.ndarray,
    ) -> None:
        source, destination = _mode(before), _mode(proposed)
        for label, selected in (("pg", pg_selected), ("other", ~pg_selected)):
            self.records[-1]["accepted_A_to_B"][label] += int(np.count_nonzero(
                accepted & selected & (source == 0) & (destination == 1)
            ))
            self.records[-1]["accepted_B_to_A"][label] += int(np.count_nonzero(
                accepted & selected & (source == 1) & (destination == 0)
            ))

    def end_stage(self, associations: np.ndarray) -> None:
        self.records[-1]["post_mode"] = (
            np.bincount(_mode(associations), minlength=3) / len(associations)
        ).tolist()


def _run_order() -> list[tuple[int, int]]:
    return [(n, seed) for n in PARTICLES for seed in SEEDS]


def run_one(output: Path, *, n_particles: int, seed: int) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set the four BLAS/OMP thread variables to 1")
    if (n_particles, seed) not in _run_order():
        raise ValueError("run is outside the frozen bridge")
    result = _load(output)
    if result["status"] == "bridge_invalid_no_competing_modes":
        raise ValueError("the exact reference did not validate this bridge")
    if result["reference"] is None or result["selected_rate"] is None:
        raise ValueError("run the exact reference before the sampler")
    if str(seed) in result["runs"][str(n_particles)]:
        print(f"bridge {n_particles} seed {seed}: reused", flush=True)
        return result
    completed = [(n, s) for n, s in _run_order() if str(s) in result["runs"][str(n)]]
    if (n_particles, seed) != _run_order()[len(completed)]:
        raise ValueError("run the frozen particle-count and seed order")
    _set_cpu_limit(RUN_CPU_CAPS[n_particles])
    case = _case(result["selected_rate"])
    controls = replace(_candidate_controls(seed), n_particles=n_particles)
    observer = _BridgeObserver()
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std, coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior,
        config=controls, observer=observer,
    )
    cpu = time.process_time() - started_cpu
    wall = time.perf_counter() - started_wall
    record = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "errors": _errors(sampled.associations, result["reference"]["exact"]),
        "stage_mode_trace": observer.records,
    }
    record["errors"]["log_evidence_error"] = (
        sampled.log_evidence_estimate - result["reference"]["log_evidence"]
    )
    result["runs"][str(n_particles)][str(seed)] = record
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"bridge {n_particles} seed {seed}: "
          f"{cpu:.1f}s CPU, mode errors "
          f"{record['errors']['mode_A_signed_error']:+.3f}, "
          f"{record['errors']['mode_B_signed_error']:+.3f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    comparison = {}
    for n in PARTICLES:
        runs = [result["runs"][str(n)][str(seed)] for seed in SEEDS]
        errors = [r["errors"] for r in runs]
        baseline = result["reference"]["iid_baseline"][str(n)]
        comparison[str(n)] = {
            "median_errors": {key: float(median(e[key] for e in errors)) for key in errors[0]},
            "median_absolute_errors": {
                key: float(median(abs(e[key]) for e in errors)) for key in errors[0]
            },
            "error_ranges": {key: [min(e[key] for e in errors), max(e[key] for e in errors)]
                             for key in errors[0]},
            "iid_q95_exceedances": {
                key: sum(abs(e[key]) > stats["q95"] for e in errors)
                for key, stats in baseline.items()
            },
            "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in runs])),
            "mean_final_mode_error": {
                key: float(np.mean([e[key] for e in errors]))
                for key in ("mode_A_signed_error", "mode_B_signed_error")
            },
            "total_accepted_mode_crossings": {
                label: {
                    direction: sum(
                        stage[f"accepted_{direction}"][label]
                        for run in runs for stage in run["stage_mode_trace"]
                    )
                    for direction in ("A_to_B", "B_to_A")
                }
                for label in ("pg", "other")
            },
        }
    result["comparison"] = comparison
    result["status"] = "absolute_bridge_audit_complete_no_move_selected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11l_exact_bridge.json"))
    args = parser.parse_args()
    if args.case == "reference":
        if args.particles is not None or args.seed is not None:
            parser.error("exact reference takes no particles or seed")
        run_reference(args.output)
    else:
        if args.particles is None or args.seed is None:
            parser.error("sampler requires --particles and --seed")
        run_one(args.output, n_particles=args.particles, seed=args.seed)


if __name__ == "__main__":
    main()
