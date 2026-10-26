"""Frozen paired four-slot exact references for joint-cell calibration."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .data import DetectionSequence
from .existence_smc import sample_slot_existence_posterior
from .slot_existence import ABSENT, SlotExistenceMaternModel, slot_existence_support_size
from .wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR, _ScalingCase
from .wp7_convergence_audit import _case_metadata, _large_run_summary, _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11n_dense_three_slot_bridge import _set_cpu_limit, _support_blocks

OBSERVATIONS = {
    "primary": (
        ((-0.019, 0.001),), ((-0.011, 0.002),), ((-0.004, 0.001),),
        ((0.003, -0.001),), ((-0.003, 0.002), (0.010, -0.001)),
    ),
    "replica": (
        ((-0.017, -0.002),), ((-0.010, -0.001),), ((-0.001, 0.001),),
        ((0.005, 0.002),), ((-0.006, 0.001), (0.013, 0.000)),
    ),
}
OBSERVATION_HASHES = {
    "primary": "1fae28c9d57ea22d6d0a6a3457dec90d6acddae910541f583f9497d1effbd5c1",
    "replica": "ccc188850719d35edbd3fcfe018eab63f6ad5f5b34c2daabe17e267ec65f6d4f",
}
SUPPORT_SIZE = 24_193_016
CARDINALITIES = (1, 1, 1, 1, 2)
CLUTTER_RATES = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
FRAME_INDEX = 4
LOW_ALIVE_MAX = 9
HIGH_ALIVE_MIN = 10
N_ALIVE = 21
N_CLUTTER = 3
TEMPERED_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
OBSERVATION_ORDER = ("primary", "replica")
SEEDS = {"primary": (1000, 1001, 1002, 1003, 1004, 1005),
         "replica": (1010, 1011, 1012, 1013, 1014, 1015)}
PARTICLES = (512, 2_048)
IID_REPLICATES = 1_000
IID_SEEDS = {"primary": 33_331, "replica": 33_337}
REFERENCE_CPU_CAP = 6_000
RUN_CPU_CAPS = {512: 600, 2048: 1_600}
JOINT_TAIL_MIN_EXCEEDANCES = 2


def _case(observation: str, rate: float) -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(5, dtype=float) * 0.001,
        detections=tuple(np.asarray(frame, dtype=float) for frame in OBSERVATIONS[observation]),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=4, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.55, expected_clutter_per_frame=rate,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model, truth_mask=np.zeros((5, 4), dtype=bool),
        localization_std=0.005, coordinate_std=(0.02, 0.01), length_scale=0.003,
    )


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "paired four-slot absolute joint-cell diagnostic",
        "sampler": "unchanged WP11g 10% four-path PG mixture",
        "observation_hashes": OBSERVATION_HASHES,
        "cardinalities": list(CARDINALITIES), "support_size_each": SUPPORT_SIZE,
        "clutter_rate_grid": list(CLUTTER_RATES),
        "mode_A": "alive slot-frames <= 9 and final-frame clutter >= 1",
        "mode_B": "alive slot-frames >= 10 and final-frame clutter == 0",
        "mode_other": "all remaining states",
        "rate_selection": (
            "maximize min(exact P(A), exact P(B)) independently per observation; "
            "tie to smaller clutter rate; do not exclude a weakly balanced case"
        ),
        "joint_cells": "all 21 alive totals (0..20) by 3 final-frame clutter counts (0..2)",
        "per_run_comparison": (
            "full joint TV and maximum absolute signed cell error against "
            "1000 exact multinomial IID draws at each particle count; "
            "also report mode, alive, clutter and evidence errors"
        ),
        "cell_group_comparison": (
            "mean signed error per cell across six 2048-particle seeds; "
            "familywise q95 of the maximum absolute cell mean from 1000 "
            "independent six-draw exact-posterior groups"
        ),
        "replication_rule": (
            "replicated_joint_tail if both observations have >=2/6 2048-particle "
            "full-joint-TV errors above their own IID q95; replicated_cell_shift "
            "if at least one same (alive, clutter) cell exceeds its observation's "
            "familywise group q95 in both observations with the same signed error; "
            "report both flags, neither selects a sampler move"
        ),
        "reference_before_sampler": True,
        "tempered_beta_grid": list(TEMPERED_GRID),
        "iid_replicates": IID_REPLICATES, "iid_seeds": IID_SEEDS,
        "observation_order": list(OBSERVATION_ORDER),
        "seeds": {key: list(value) for key, value in SEEDS.items()},
        "particles": list(PARTICLES),
        "run_order": (
            "primary reference, replica reference; then primary 512/2048 "
            "seeds 1000..1005 and replica 512/2048 seeds 1010..1015"
        ),
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap_each": REFERENCE_CPU_CAP,
        "per_run_cpu_caps": {str(n): cap for n, cap in RUN_CPU_CAPS.items()},
        "interpretation": (
            "absolute diagnostic on two four-slot/five-frame cases, not the "
            "four-slot/eight-frame primary case; no 8192 or untouched holdout"
        ),
    }


def _metadata() -> dict[str, Any]:
    result = {}
    for observation in OBSERVATION_ORDER:
        model = _case(observation, CLUTTER_RATES[0]).model
        info = _case_metadata(_case(observation, CLUTTER_RATES[0]))
        if (info["detections_sha256"] != OBSERVATION_HASHES[observation]
                or tuple(info["detection_cardinalities"]) != CARDINALITIES
                or slot_existence_support_size(model) != SUPPORT_SIZE):
            raise ValueError(f"frozen {observation} observation changed")
        result[observation] = {
            "observation": info,
            "models": [_case_metadata(_case(observation, rate)) for rate in CLUTTER_RATES],
        }
    return result


def _load(output: Path) -> dict[str, Any]:
    protocol, cases = _protocol(), _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != cases:
            raise ValueError("checkpoint belongs to a different frozen protocol")
        return result
    result = {
        "experiment": "wp11o_four_slot_joint_reference",
        "protocol": protocol, "cases": cases,
        "references": {key: None for key in OBSERVATION_ORDER},
        "runs": {key: {str(n): {} for n in PARTICLES} for key in OBSERVATION_ORDER},
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
    joint = np.bincount(
        N_CLUTTER * alive + clutter, weights=weights,
        minlength=N_ALIVE * N_CLUTTER,
    ).reshape(N_ALIVE, N_CLUTTER)
    return _from_joint(joint)


def _from_joint(joint: np.ndarray) -> dict[str, Any]:
    mode_a = joint[:LOW_ALIVE_MAX + 1, 1:].sum()
    mode_b = joint[HIGH_ALIVE_MIN:, 0].sum()
    return {
        "mode": [float(mode_a), float(mode_b), float(joint.sum() - mode_a - mode_b)],
        "joint_alive_by_clutter": joint.tolist(),
        "alive": joint.sum(axis=1).tolist(), "clutter": joint.sum(axis=0).tolist(),
    }


def _errors_from_joint(observed: np.ndarray, exact: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray(exact["joint_alive_by_clutter"])
    delta = observed - truth
    p = _from_joint(observed)
    mode_delta = np.asarray(p["mode"]) - np.asarray(exact["mode"])
    index = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    return {
        "mode_A_signed_error": float(mode_delta[0]),
        "mode_B_signed_error": float(mode_delta[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(mode_delta))),
        "full_joint_tv": float(np.abs(delta).sum() / 2),
        "maximum_cell_absolute_error": float(np.max(np.abs(delta))),
        "max_error_cell": list(map(int, index)),
        "max_error_cell_signed_error": float(delta[index]),
        "alive_total_tv": float(np.abs(delta.sum(axis=1)).sum() / 2),
        "clutter_mean_error": float(np.dot(np.arange(N_CLUTTER), delta.sum(axis=0))),
    }


def _reference_at_beta(
    log_base: np.ndarray, log_marginal: np.ndarray, clutter_total: np.ndarray,
    alive: np.ndarray, last_clutter: np.ndarray, rate: float, beta: float,
) -> tuple[dict[str, Any], float]:
    scores = log_base + beta * log_marginal + clutter_total * np.log(rate)
    normalizer = float(logsumexp(scores))
    return _distributions(alive, last_clutter, np.exp(scores - normalizer)), normalizer


def _iid_baseline(exact: dict[str, Any], observation: str) -> dict[str, Any]:
    rng = np.random.default_rng(IID_SEEDS[observation])
    p = np.asarray(exact["joint_alive_by_clutter"]).reshape(-1)
    p /= p.sum()
    baseline = {}
    for n in PARTICLES:
        draws = {key: [] for key in (
            "maximum_mode_absolute_error", "full_joint_tv",
            "maximum_cell_absolute_error", "alive_total_tv", "clutter_mean_error",
        )}
        for counts in rng.multinomial(n, p, size=IID_REPLICATES):
            errors = _errors_from_joint(counts.reshape(N_ALIVE, N_CLUTTER) / n, exact)
            for key, values in draws.items():
                values.append(abs(errors[key]))
        grouped = rng.multinomial(len(SEEDS[observation]) * n, p, size=IID_REPLICATES)
        grouped_max = np.max(
            np.abs(grouped / (len(SEEDS[observation]) * n) - p), axis=1
        )
        baseline[str(n)] = {
            "per_run": {
                key: {"median": float(np.median(values)),
                      "q95": float(np.quantile(values, .95))}
                for key, values in draws.items()
            },
            "group_max_cell_mean": {
                "median": float(np.median(grouped_max)),
                "q95": float(np.quantile(grouped_max, .95)),
            },
        }
    return baseline


def run_reference(output: Path, observation: str) -> dict[str, Any]:
    result = _load(output)
    if result["references"][observation] is not None:
        print(f"{observation} exact reference: reused", flush=True)
        return result
    if observation == "replica" and result["references"]["primary"] is None:
        raise ValueError("run primary exact reference first")
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    case = _case(observation, 1.0)
    n = SUPPORT_SIZE
    log_marginal = np.empty(n)
    log_base = np.empty(n)
    clutter_total = np.empty(n, dtype=np.uint8)
    alive = np.empty(n, dtype=np.uint8)
    last_clutter = np.empty(n, dtype=np.uint8)
    other_models = [_case(observation, rate).model
                    for rate in (CLUTTER_RATES[0], CLUTTER_RATES[-1])]
    lo, next_report = 0, 2_000_000
    for block in _support_blocks(case.model):
        hi = lo + len(block)
        if hi > n:
            raise ValueError("enumeration exceeded frozen support size")
        log_marginal[lo:hi] = case.model.log_marginal_density(block)
        log_count = case.model.log_count_density(block, include_poisson_void=False)
        log_base[lo:hi] = case.model.log_existence_prior(block) + log_count
        clutter_total[lo:hi] = sum(CARDINALITIES) - np.count_nonzero(
            block >= 0, axis=(1, 2)
        )
        alive[lo:hi], last_clutter[lo:hi] = _counts(block)
        if lo == 0:
            for model in other_models:
                direct = model.log_count_density(block, include_poisson_void=False)
                expected = log_count + clutter_total[lo:hi] * np.log(
                    model.expected_clutter_per_frame
                )
                np.testing.assert_allclose(direct, expected, atol=1e-10, rtol=0)
        lo = hi
        if lo >= next_report:
            print(f"{observation} exact support: {lo:,}/{n:,}", flush=True)
            next_report += 2_000_000
    if lo != n:
        raise ValueError("enumeration did not cover frozen support size")
    grid = []
    for rate in CLUTTER_RATES:
        model = _case(observation, rate).model
        base_normalizer = float(logsumexp(log_base + clutter_total * np.log(rate)))
        if not np.isclose(
            base_normalizer - rate * len(CARDINALITIES),
            model.base_log_normalizer, atol=1e-8,
        ):
            raise ValueError("enumerated base normalizer does not match model")
        distribution, log_norm = _reference_at_beta(
            log_base, log_marginal, clutter_total, alive, last_clutter, rate, 1.0
        )
        grid.append({
            "clutter_rate": rate, "mode": distribution["mode"],
            "log_evidence": log_norm - rate * len(CARDINALITIES),
        })
    selected = min(grid, key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]))
    rate = selected["clutter_rate"]
    tempered = []
    for beta in TEMPERED_GRID:
        distribution, _ = _reference_at_beta(
            log_base, log_marginal, clutter_total, alive, last_clutter, rate, beta
        )
        tempered.append({"beta": beta, "mode": distribution["mode"]})
    exact, _ = _reference_at_beta(
        log_base, log_marginal, clutter_total, alive, last_clutter, rate, 1.0
    )
    result["references"][observation] = {
        "support_size": n, "reference_grid": grid, "selected_rate": rate,
        "exact": exact, "log_evidence": selected["log_evidence"],
        "tempered_mode_curve": tempered,
        "iid_baseline": _iid_baseline(exact, observation),
        "mode_competition_at_least_0_10": min(exact["mode"][:2]) >= .10,
        "cpu_seconds": time.process_time() - started,
    }
    _save(output, result)
    print(f"{observation} exact reference: {n:,} states, rate {rate}, "
          f"modes {exact['mode'][:2]}, "
          f"{result['references'][observation]['cpu_seconds']:.1f}s CPU", flush=True)
    return result


class _ModeObserver:
    """RNG-free observer of mode weights and accepted paired crossings."""

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


def _run_order() -> list[tuple[str, int, int]]:
    return [(observation, n, seed)
            for observation in OBSERVATION_ORDER for n in PARTICLES
            for seed in SEEDS[observation]]


def run_one(
    output: Path, *, observation: str, n_particles: int, seed: int
) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    if (observation, n_particles, seed) not in _run_order():
        raise ValueError("run is outside frozen four-slot schedule")
    result = _load(output)
    if any(result["references"][key] is None for key in OBSERVATION_ORDER):
        raise ValueError("run both independent exact references first")
    if str(seed) in result["runs"][observation][str(n_particles)]:
        print(f"{observation} {n_particles} seed {seed}: reused", flush=True)
        return result
    completed = [(obs, n, s) for obs, n, s in _run_order()
                 if str(s) in result["runs"][obs][str(n)]]
    if (observation, n_particles, seed) != _run_order()[len(completed)]:
        raise ValueError("run the frozen observation, particle and seed order")
    _set_cpu_limit(RUN_CPU_CAPS[n_particles])
    ref = result["references"][observation]
    case = _case(observation, ref["selected_rate"])
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
    joint = np.asarray(_distributions(
        alive, clutter, np.full(len(alive), 1 / len(alive))
    )["joint_alive_by_clutter"])
    errors = _errors_from_joint(joint, ref["exact"])
    errors["log_evidence_error"] = (
        sampled.log_evidence_estimate - ref["log_evidence"]
    )
    result["runs"][observation][str(n_particles)][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "joint_alive_by_clutter": joint.tolist(),
        "errors": errors, "stage_mode_trace": observer.records,
    }
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"{observation} {n_particles} seed {seed}: {cpu:.1f}s CPU, "
          f"joint TV {errors['full_joint_tv']:.4f}, "
          f"max cell {errors['maximum_cell_absolute_error']:.4f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    comparison = {}
    for observation in OBSERVATION_ORDER:
        comparison[observation] = {}
        ref = result["references"][observation]
        exact_joint = np.asarray(ref["exact"]["joint_alive_by_clutter"])
        for n in PARTICLES:
            runs = [result["runs"][observation][str(n)][str(seed)]
                    for seed in SEEDS[observation]]
            errors = [run["errors"] for run in runs]
            baseline = ref["iid_baseline"][str(n)]
            means = np.mean([run["joint_alive_by_clutter"] for run in runs], axis=0)
            delta = means - exact_joint
            group_q95 = baseline["group_max_cell_mean"]["q95"]
            flagged = [
                {"alive": int(k), "clutter": int(c), "signed_error": float(delta[k, c])}
                for k, c in np.argwhere(np.abs(delta) > group_q95)
            ]
            scalar = ("maximum_mode_absolute_error", "full_joint_tv",
                      "maximum_cell_absolute_error", "alive_total_tv",
                      "clutter_mean_error", "log_evidence_error")
            comparison[observation][str(n)] = {
                "mean_signed_cell_error": delta.tolist(),
                "max_absolute_mean_cell_error": float(np.max(np.abs(delta))),
                "familywise_flagged_cells": flagged,
                "median_absolute_errors": {
                    key: float(median(abs(e[key]) for e in errors)) for key in scalar
                },
                "error_ranges": {
                    key: [min(e[key] for e in errors), max(e[key] for e in errors)]
                    for key in scalar
                },
                "iid_q95_exceedances": {
                    key: sum(abs(e[key]) > stats["q95"] for e in errors)
                    for key, stats in baseline["per_run"].items()
                },
                "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in runs])),
                "mean_final_mode_error": {
                    key: float(np.mean([e[key] for e in errors]))
                    for key in ("mode_A_signed_error", "mode_B_signed_error")
                },
            }
    a = comparison["primary"]["2048"]
    b = comparison["replica"]["2048"]
    joint_tail = all(
        row["iid_q95_exceedances"]["full_joint_tv"] >= JOINT_TAIL_MIN_EXCEEDANCES
        for row in (a, b)
    )
    signed_cells = lambda row: {
        (x["alive"], x["clutter"], np.sign(x["signed_error"]))
        for x in row["familywise_flagged_cells"]
    }
    shared = sorted(signed_cells(a) & signed_cells(b))
    result["comparison"] = comparison
    result["replication"] = {
        "replicated_joint_tail": joint_tail,
        "replicated_cell_shift": bool(shared),
        "shared_signed_cells": [
            {"alive": k, "clutter": c, "sign": int(sign)} for k, c, sign in shared
        ],
    }
    result["status"] = (
        "four_slot_joint_audit_complete_replicated_signal"
        if joint_tail or shared else "four_slot_joint_audit_complete_no_replicated_signal"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--observation", choices=OBSERVATION_ORDER, required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11o_four_slot_joint_reference.json"))
    args = parser.parse_args()
    if args.case == "reference":
        if args.particles is not None or args.seed is not None:
            parser.error("reference takes no particles or seed")
        run_reference(args.output, args.observation)
    else:
        if args.particles is None or args.seed is None:
            parser.error("sampler requires --particles and --seed")
        run_one(args.output, observation=args.observation,
                n_particles=args.particles, seed=args.seed)


if __name__ == "__main__":
    main()
