"""Frozen three-slot/eight-frame exact alive-clutter calibration bridge."""

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

OBSERVATION_SHA256 = "005b33179c58d2385cfe6e49e3ff3ddb264e2f6ab41b4f31a56974250416e5d6"
SUPPORT_SIZE = 8_375_839
CARDINALITIES = (1, 0, 1, 0, 1, 2, 1, 0)
CLUTTER_RATES = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
FRAME_INDEX = 5
LOW_ALIVE_MAX = 12
HIGH_ALIVE_MIN = 13
MIN_MODE_MASS = 0.10
MAX_LATE_FRACTION = 0.50
SEEDS = (920, 921, 922, 923, 924, 925)
PARTICLES = (512, 2_048)
IID_REPLICATES = 500
IID_SEED = 31_313
TEMPERED_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
REFERENCE_CPU_CAP = 3_600
RUN_CPU_CAPS = {512: 450, 2048: 1_200}
BATCH_SIZE = 20_000


def _case(clutter_rate: float) -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(8, dtype=float) * 0.001,
        detections=(
            np.asarray([[-0.019, 0.001]]),
            np.empty((0, 2)),
            np.asarray([[-0.007, 0.002]]),
            np.empty((0, 2)),
            np.asarray([[0.001, 0.000]]),
            np.asarray([[-0.005, 0.002], [0.006, -0.001]]),
            np.asarray([[0.012, -0.001]]),
            np.empty((0, 2)),
        ),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=3, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.55, expected_clutter_per_frame=clutter_rate,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model, truth_mask=np.zeros((8, 3), dtype=bool),
        localization_std=0.005, coordinate_std=(0.02, 0.01), length_scale=0.003,
    )


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "absolute three-slot/eight-frame mode-weight diagnostic",
        "sampler": "unchanged WP11g 10% four-path PG mixture",
        "observation_sha256": OBSERVATION_SHA256,
        "support_size": SUPPORT_SIZE,
        "cardinalities": list(CARDINALITIES),
        "clutter_rate_grid": list(CLUTTER_RATES),
        "mode_A": "alive slot-frames <= 12 and frame-six clutter >= 1",
        "mode_B": "alive slot-frames >= 13 and frame-six clutter == 0",
        "other_mode": "all remaining support",
        "bridge_selection": (
            "maximize min(exact mass A, exact mass B) over the frozen rate grid; "
            "tie goes to smaller rate; require both masses >= 0.10 and exact "
            "P(B) at beta 0.75 <= half P(B) at beta one"
        ),
        "reference_before_sampler": True,
        "tempered_beta_grid": list(TEMPERED_GRID),
        "iid_replicates_per_particle_count": IID_REPLICATES,
        "iid_seed": IID_SEED,
        "seeds": list(SEEDS), "particles": list(PARTICLES),
        "run_order": "reference, 512 seeds 920..925, then 2048 seeds 920..925",
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap": REFERENCE_CPU_CAP,
        "per_run_cpu_caps": {str(n): cap for n, cap in RUN_CPU_CAPS.items()},
        "interpretation": (
            "diagnostic exact calibration, not a move-selection gate; "
            "no 8192 run or untouched holdout"
        ),
    }


def _metadata() -> dict[str, Any]:
    case = _case(CLUTTER_RATES[0])
    info = _case_metadata(case)
    if (info["detections_sha256"] != OBSERVATION_SHA256
            or tuple(info["detection_cardinalities"]) != CARDINALITIES
            or slot_existence_support_size(case.model) != SUPPORT_SIZE):
        raise ValueError("the frozen three-slot bridge changed")
    return {"observation": info, "models": [
        _case_metadata(_case(rate)) for rate in CLUTTER_RATES
    ]}


def _load(output: Path) -> dict[str, Any]:
    protocol, cases = _protocol(), _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != cases:
            raise ValueError("checkpoint belongs to another frozen protocol")
        return result
    result = {
        "experiment": "wp11m_three_slot_exact_bridge",
        "protocol": protocol, "cases": cases,
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


def _mode(alive: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    mode = np.full(len(alive), 2, dtype=np.int8)
    mode[(alive <= LOW_ALIVE_MAX) & (clutter >= 1)] = 0
    mode[(alive >= HIGH_ALIVE_MIN) & (clutter == 0)] = 1
    return mode


def _distributions(
    alive: np.ndarray, clutter: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    mode = _mode(alive, clutter)
    return {
        "mode": np.bincount(mode, weights=weights, minlength=3).tolist(),
        "joint_alive_by_clutter": np.bincount(
            3 * alive + clutter, weights=weights, minlength=25 * 3
        ).reshape(25, 3).tolist(),
        "alive": np.bincount(alive, weights=weights, minlength=25).tolist(),
        "clutter": np.bincount(clutter, weights=weights, minlength=3).tolist(),
    }


def _errors_from_counts(
    alive: np.ndarray, clutter: np.ndarray, exact: dict[str, Any]
) -> dict[str, float]:
    observed = _distributions(alive, clutter, np.full(len(alive), 1 / len(alive)))
    p, q = np.asarray(observed["mode"]), np.asarray(exact["mode"])
    jp = np.asarray(observed["joint_alive_by_clutter"])
    jq = np.asarray(exact["joint_alive_by_clutter"])
    ap, aq = np.asarray(observed["alive"]), np.asarray(exact["alive"])
    cp, cq = np.asarray(observed["clutter"]), np.asarray(exact["clutter"])
    return {
        "mode_A_signed_error": float(p[0] - q[0]),
        "mode_B_signed_error": float(p[1] - q[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(p - q))),
        "three_mode_tv": float(np.abs(p - q).sum() / 2),
        "full_joint_tv": float(np.abs(jp - jq).sum() / 2),
        "alive_total_tv": float(np.abs(ap - aq).sum() / 2),
        "clutter_mean_error": float(np.dot(np.arange(3), cp - cq)),
    }


def _set_cpu_limit(seconds: int) -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < seconds + 5:
        raise RuntimeError("system hard CPU limit is below frozen cap")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))


def _reference_at_beta(
    log_base: np.ndarray, log_marginal: np.ndarray, clutter_total: np.ndarray,
    alive: np.ndarray, sixth_clutter: np.ndarray, rate: float, beta: float,
) -> tuple[dict[str, Any], float]:
    scores = log_base + beta * log_marginal + clutter_total * np.log(rate)
    normalizer = float(logsumexp(scores))
    return _distributions(alive, sixth_clutter, np.exp(scores - normalizer)), normalizer


def run_reference(output: Path) -> dict[str, Any]:
    result = _load(output)
    if result["reference_grid"] is not None:
        print("three-slot exact reference: reused", flush=True)
        return result
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    case = _case(1.0)
    support, masks = enumerate_slot_existence_support(
        case.model, max_hypotheses=SUPPORT_SIZE
    )
    del masks
    n = len(support)
    log_marginal = np.empty(n)
    log_base = np.empty(n)
    clutter_total = np.empty(n, dtype=np.uint8)
    alive = np.empty(n, dtype=np.uint8)
    sixth_clutter = np.empty(n, dtype=np.uint8)
    other_models = [_case(rate).model for rate in (CLUTTER_RATES[0], CLUTTER_RATES[-1])]
    for lo in range(0, n, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE, n)
        block = support[lo:hi]
        log_marginal[lo:hi] = case.model.log_marginal_density(block)
        log_base[lo:hi] = (
            case.model.log_existence_prior(block)
            + case.model.log_count_density(block, include_poisson_void=False)
        )
        clutter_total[lo:hi] = sum(CARDINALITIES) - np.count_nonzero(
            block >= 0, axis=(1, 2)
        )
        alive[lo:hi], sixth_clutter[lo:hi] = _counts(block)
        if lo == 0:
            baseline_count = case.model.log_count_density(
                block, include_poisson_void=False
            )
            for model in other_models:
                direct = model.log_count_density(block, include_poisson_void=False)
                expected = baseline_count + clutter_total[lo:hi] * np.log(
                    model.expected_clutter_per_frame
                )
                np.testing.assert_allclose(direct, expected, atol=1e-10, rtol=0)
    grid = []
    for rate in CLUTTER_RATES:
        model = _case(rate).model
        base_normalizer = float(logsumexp(log_base + clutter_total * np.log(rate)))
        if not np.isclose(
            base_normalizer - rate * len(CARDINALITIES),
            model.base_log_normalizer, atol=1e-8,
        ):
            raise ValueError("enumerated base normalizer does not match model")
        distribution, log_norm = _reference_at_beta(
            log_base, log_marginal, clutter_total, alive, sixth_clutter, rate, 1.0
        )
        grid.append({
            "clutter_rate": rate, "mode": distribution["mode"],
            "log_evidence": log_norm - rate * len(CARDINALITIES),
        })
    selected = min(grid, key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]))
    result["reference_grid"] = grid
    rate = selected["clutter_rate"]
    tempered = []
    for beta in TEMPERED_GRID:
        distribution, _ = _reference_at_beta(
            log_base, log_marginal, clutter_total, alive, sixth_clutter, rate, beta
        )
        tempered.append({"beta": beta, "mode": distribution["mode"]})
    if (min(selected["mode"][:2]) < MIN_MODE_MASS
            or tempered[3]["mode"][1] > MAX_LATE_FRACTION * selected["mode"][1]):
        result["status"] = "bridge_invalid_mode_or_late_criterion"
        result["validity_diagnostics"] = {
            "selected_rate": rate, "selected_mode": selected["mode"],
            "tempered_mode_curve": tempered,
        }
        _save(output, result)
        print("three-slot exact reference: frozen validity criteria failed", flush=True)
        return result
    exact, log_norm = _reference_at_beta(
        log_base, log_marginal, clutter_total, alive, sixth_clutter, rate, 1.0
    )
    probabilities = np.exp(
        log_base + log_marginal + clutter_total * np.log(rate) - log_norm
    )
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    rng = np.random.default_rng(IID_SEED)
    baseline = {}
    for particle_count in PARTICLES:
        draws = {key: [] for key in (
            "maximum_mode_absolute_error", "three_mode_tv", "full_joint_tv",
            "alive_total_tv", "clutter_mean_error",
        )}
        for _ in range(IID_REPLICATES):
            indices = np.searchsorted(cdf, rng.random(particle_count))
            errors = _errors_from_counts(alive[indices], sixth_clutter[indices], exact)
            for key, values in draws.items():
                values.append(abs(errors[key]))
        baseline[str(particle_count)] = {
            key: {"median": float(np.median(vals)),
                  "q95": float(np.quantile(vals, .95))}
            for key, vals in draws.items()
        }
    result["selected_rate"] = rate
    result["reference"] = {
        "support_size": n, "exact": exact,
        "log_evidence": selected["log_evidence"],
        "tempered_mode_curve": tempered,
        "iid_baseline": baseline,
        "cpu_seconds": time.process_time() - started,
    }
    _save(output, result)
    print(f"three-slot exact reference: {n:,} states, rate {rate}, "
          f"modes {selected['mode'][:2]}, "
          f"{result['reference']['cpu_seconds']:.1f}s CPU", flush=True)
    return result


class _ModeObserver:
    """Passive stagewise mode masses and accepted A/B transitions."""

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
        modes = _mode(*_counts(associations))
        before = np.bincount(modes, weights=weights, minlength=3)
        after = np.bincount(modes[ancestors], minlength=3) / len(ancestors)
        self.records[-1]["weighted_mode"] = before.tolist()
        self.records[-1]["resampled_mode"] = after.tolist()
        self.records[-1]["resampling_mode_tv"] = float(np.abs(before - after).sum() / 2)

    def record_pair(
        self, before: np.ndarray, proposed: np.ndarray,
        accepted: np.ndarray, pg_selected: np.ndarray,
    ) -> None:
        source, destination = _mode(*_counts(before)), _mode(*_counts(proposed))
        for label, selected in (("pg", pg_selected), ("other", ~pg_selected)):
            self.records[-1]["accepted_A_to_B"][label] += int(np.count_nonzero(
                accepted & selected & (source == 0) & (destination == 1)
            ))
            self.records[-1]["accepted_B_to_A"][label] += int(np.count_nonzero(
                accepted & selected & (source == 1) & (destination == 0)
            ))

    def end_stage(self, associations: np.ndarray) -> None:
        self.records[-1]["post_mode"] = (
            np.bincount(_mode(*_counts(associations)), minlength=3) / len(associations)
        ).tolist()


def _run_order() -> list[tuple[int, int]]:
    return [(n, seed) for n in PARTICLES for seed in SEEDS]


def run_one(output: Path, *, n_particles: int, seed: int) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    if (n_particles, seed) not in _run_order():
        raise ValueError("run is outside the frozen three-slot bridge")
    result = _load(output)
    if result["status"] == "bridge_invalid_mode_or_late_criterion":
        raise ValueError("exact reference did not validate this bridge")
    if result["reference"] is None or result["selected_rate"] is None:
        raise ValueError("run the exact reference first")
    if str(seed) in result["runs"][str(n_particles)]:
        print(f"three-slot bridge {n_particles} seed {seed}: reused", flush=True)
        return result
    completed = [(n, s) for n, s in _run_order() if str(s) in result["runs"][str(n)]]
    if (n_particles, seed) != _run_order()[len(completed)]:
        raise ValueError("run the frozen particle-count and seed order")
    _set_cpu_limit(RUN_CPU_CAPS[n_particles])
    case = _case(result["selected_rate"])
    controls = replace(_candidate_controls(seed), n_particles=n_particles)
    observer = _ModeObserver()
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
    cpu, wall = time.process_time() - started_cpu, time.perf_counter() - started_wall
    alive, clutter = _counts(sampled.associations)
    errors = _errors_from_counts(alive, clutter, result["reference"]["exact"])
    errors["log_evidence_error"] = (
        sampled.log_evidence_estimate - result["reference"]["log_evidence"]
    )
    result["runs"][str(n_particles)][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "errors": errors, "stage_mode_trace": observer.records,
    }
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"three-slot bridge {n_particles} seed {seed}: {cpu:.1f}s CPU, "
          f"mode errors {errors['mode_A_signed_error']:+.3f}, "
          f"{errors['mode_B_signed_error']:+.3f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    comparisons = {}
    for n in PARTICLES:
        runs = [result["runs"][str(n)][str(seed)] for seed in SEEDS]
        errors = [run["errors"] for run in runs]
        baseline = result["reference"]["iid_baseline"][str(n)]
        comparisons[str(n)] = {
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
    result["comparison"] = comparisons
    result["status"] = "three_slot_absolute_audit_complete_no_move_selected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11m_three_slot_bridge.json"))
    args = parser.parse_args()
    if args.case == "reference":
        if args.particles is not None or args.seed is not None:
            parser.error("reference takes no particles or seed")
        run_reference(args.output)
    else:
        if args.particles is None or args.seed is None:
            parser.error("sampler requires --particles and --seed")
        run_one(args.output, n_particles=args.particles, seed=args.seed)


if __name__ == "__main__":
    main()
