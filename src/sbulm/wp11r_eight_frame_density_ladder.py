"""Frozen eight-/nine-detection exact four-slot density ladder."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from math import log
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .data import DetectionSequence
from .existence_smc import sample_slot_existence_posterior
from .slot_existence import ABSENT, SlotExistenceMaternModel
from .wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR, _ScalingCase
from .wp7_convergence_audit import _case_metadata, _large_run_summary, _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11n_dense_three_slot_bridge import _set_cpu_limit
from .wp11q_seven_detection_exact_reference import (
    _Accumulation,
    _canonical_blocks,
    _gp_from_table,
    _gp_table,
    _ModeObserver,
)

COMMON = (
    ((-0.019, 0.001),), ((-0.014, 0.001),), ((-0.007, 0.002),), (),
    ((0.001, 0.000),), ((-0.005, 0.002), (0.006, -0.001)),
    ((0.012, -0.001),),
)
OBSERVATIONS = {
    "eight": COMMON + (((0.018, -0.001),),),
    "nine": COMMON + (((-0.011, 0.002), (0.018, -0.001)),),
}
OBSERVATION_HASHES = {
    "eight": "af1d17a353977b9caa06dcc09b937dab715931028692832379d2270ab0f982f3",
    "nine": "2cb50f0b1c824dbee74e1b80e5220bc722deced9e6d9f2eec6a597e818b12edb",
}
CARDINALITIES = {
    "eight": (1, 1, 1, 0, 1, 2, 1, 1),
    "nine": (1, 1, 1, 0, 1, 2, 1, 2),
}
CANONICAL_SUPPORT = {"eight": 272_486_255, "nine": 555_457_390}
LABELLED_SUPPORT = {"eight": 5_245_666_773, "nine": 10_325_709_693}
OBSERVATION_ORDER = ("eight", "nine")
MASK_MULTISET_COUNT = 91_390
N_SLOTS = 4
N_FRAMES = 8
N_ALIVE = 33
FRAME_INDEX = 5
LOW_ALIVE_MAX = 15
HIGH_ALIVE_MIN = 16
CLUTTER_RATES = (0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
MIN_MODE_MASS = 0.10
MAX_LATE_FRACTION = 0.50
SEEDS = {"eight": (1160, 1161, 1162, 1163, 1164, 1165),
         "nine": (1170, 1171, 1172, 1173, 1174, 1175)}
PARTICLES = (512, 2_048)
IID_SEEDS = {"eight": 34_361, "nine": 34_367}
IID_REPLICATES = 1_000
REFERENCE_CPU_CAP = 4_000
RUN_CPU_CAPS = {512: 600, 2_048: 1_600}
TAIL_MIN_EXCEEDANCES = 2


def _case(observation: str, rate: float) -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(N_FRAMES, dtype=float) * 0.001,
        detections=tuple(np.asarray(frame, dtype=float).reshape(-1, 2)
                         for frame in OBSERVATIONS[observation]),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=N_SLOTS, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.55, expected_clutter_per_frame=rate,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model, truth_mask=np.zeros((N_FRAMES, N_SLOTS), dtype=bool),
        localization_std=0.005, coordinate_std=(0.02, 0.01), length_scale=0.003,
    )


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "paired nested-density four-slot/eight-frame absolute audit",
        "observation_hashes": OBSERVATION_HASHES,
        "observation_order": list(OBSERVATION_ORDER),
        "cardinalities": {k: list(v) for k, v in CARDINALITIES.items()},
        "canonical_support": CANONICAL_SUPPORT,
        "labelled_support": LABELLED_SUPPORT,
        "mask_multisets": MASK_MULTISET_COUNT,
        "reference_method": (
            "exact canonical interval-mask multiset sum with permutation multiplicity; "
            "directly validated one-slot GP likelihood lookup; count, rate and eleven "
            "analytic base-normalizer checks; both references before any sampler"
        ),
        "clutter_rates": list(CLUTTER_RATES),
        "rate_selection": "per observation maximize min(exact P(A), P(B)); tie to lower rate",
        "mode_A": "alive slot-frames <=15 and frame-six clutter >=1",
        "mode_B": "alive slot-frames >=16 and frame-six clutter ==0",
        "mode_other": "remaining states",
        "difficulty_validation": (
            "both observations: selected exact A/B >=0.10 and beta=0.5 B "
            "<= half beta=1 B; if either fails, stop all sampler inference"
        ),
        "joint_cells": (
            "33 alive totals by 3 frame-six clutter counts by (last-frame detections + 1) "
            "last-frame clutter counts: 198 eight-case and 297 nine-case cells"
        ),
        "comparison": (
            "per-run 3D joint TV, maximum mode and cell errors versus 1000 exact IID "
            "multinomial q95 per case and particle count; also 2D collapsed joint, "
            "alive and two clutter means; six-run mean signed 3D cells versus "
            "familywise q95 from 1000 exact groups of six"
        ),
        "persistence_rule": (
            "at 2048 particles: persistent_mode_tail and persistent_joint_tail "
            "require >=2/6 q95 exceedances in both cases; shared_cell_shift "
            "requires the same 3D cell to exceed each familywise six-run mean q95 "
            "with the same sign in both cases; nested cases are not independent"
        ),
        "reference_before_sampler": True,
        "beta_grid": list(BETAS),
        "seeds": {k: list(v) for k, v in SEEDS.items()},
        "particles": list(PARTICLES),
        "iid_replicates": IID_REPLICATES, "iid_seeds": IID_SEEDS,
        "run_order": (
            "eight reference then nine reference; eight 512/2048 seeds 1160..1165; "
            "nine 512/2048 seeds 1170..1175"
        ),
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap_each": REFERENCE_CPU_CAP,
        "run_cpu_caps": {str(n): cap for n, cap in RUN_CPU_CAPS.items()},
        "interpretation": (
            "nested designed eight/nine-detection bridge; descriptive IID references, "
            "not independent population replication or WP11k 19-detection calibration; "
            "no move selection, 8192 or untouched holdout"
        ),
    }


def _metadata() -> dict[str, Any]:
    result = {}
    for observation in OBSERVATION_ORDER:
        info = _case_metadata(_case(observation, CLUTTER_RATES[0]))
        if (info["detections_sha256"] != OBSERVATION_HASHES[observation]
                or tuple(info["detection_cardinalities"]) != CARDINALITIES[observation]):
            raise ValueError(f"frozen {observation} observation changed")
        result[observation] = {"observation": info,
                               "models": [_case_metadata(_case(observation, r))
                                          for r in CLUTTER_RATES]}
    return result


def _load(output: Path) -> dict[str, Any]:
    protocol, cases = _protocol(), _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != cases:
            raise ValueError("checkpoint belongs to a different frozen protocol")
        return result
    result = {
        "experiment": "wp11r_eight_frame_density_ladder",
        "protocol": protocol, "cases": cases,
        "references": {key: None for key in OBSERVATION_ORDER},
        "runs": {key: {str(n): {} for n in PARTICLES} for key in OBSERVATION_ORDER},
        "comparison": None, "status": "in_progress",
    }
    _save(output, result)
    return result


def _counts(states: np.ndarray, observation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    sixth = CARDINALITIES[observation][FRAME_INDEX] - np.count_nonzero(
        states[:, FRAME_INDEX] >= 0, axis=1
    )
    last = CARDINALITIES[observation][-1] - np.count_nonzero(
        states[:, -1] >= 0, axis=1
    )
    return alive, sixth, last


def _mode(alive: np.ndarray, sixth: np.ndarray) -> np.ndarray:
    mode = np.full(len(alive), 2, dtype=np.int8)
    mode[(alive <= LOW_ALIVE_MAX) & (sixth >= 1)] = 0
    mode[(alive >= HIGH_ALIVE_MIN) & (sixth == 0)] = 1
    return mode


def _from_joint(joint: np.ndarray) -> dict[str, Any]:
    a = joint[:LOW_ALIVE_MAX + 1, 1:, :].sum()
    b = joint[HIGH_ALIVE_MIN:, 0, :].sum()
    return {
        "mode": [float(a), float(b), float(joint.sum() - a - b)],
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "joint_alive_by_sixth_clutter": joint.sum(axis=2).tolist(),
        "alive": joint.sum(axis=(1, 2)).tolist(),
        "sixth_clutter": joint.sum(axis=(0, 2)).tolist(),
        "last_clutter": joint.sum(axis=(0, 1)).tolist(),
    }


def _errors(observed: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    truth = np.asarray(exact["joint_alive_by_sixth_by_last_clutter"])
    delta = observed - truth
    mode_delta = np.asarray(_from_joint(observed)["mode"]) - exact["mode"]
    return {
        "mode_A_signed_error": float(mode_delta[0]),
        "mode_B_signed_error": float(mode_delta[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(mode_delta))),
        "full_3d_joint_tv": float(np.abs(delta).sum() / 2),
        "collapsed_2d_joint_tv": float(np.abs(delta.sum(axis=2)).sum() / 2),
        "maximum_3d_cell_absolute_error": float(np.max(np.abs(delta))),
        "alive_total_tv": float(np.abs(delta.sum(axis=(1, 2))).sum() / 2),
        "sixth_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[1]), delta.sum(axis=(0, 2))
        )),
        "last_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[2]), delta.sum(axis=(0, 1))
        )),
    }


def _iid_baseline(exact: dict[str, Any], observation: str) -> dict[str, Any]:
    rng = np.random.default_rng(IID_SEEDS[observation])
    shape = (N_ALIVE, 3, CARDINALITIES[observation][-1] + 1)
    p = np.asarray(exact["joint_alive_by_sixth_by_last_clutter"]).reshape(-1)
    p /= p.sum()
    metrics = ("maximum_mode_absolute_error", "full_3d_joint_tv",
               "collapsed_2d_joint_tv", "maximum_3d_cell_absolute_error",
               "alive_total_tv", "sixth_clutter_mean_error", "last_clutter_mean_error")
    baseline = {}
    for n in PARTICLES:
        draws = {key: [] for key in metrics}
        for counts in rng.multinomial(n, p, size=IID_REPLICATES):
            errors = _errors(counts.reshape(shape) / n, exact)
            for key in metrics:
                draws[key].append(abs(errors[key]))
        grouped = rng.multinomial(len(SEEDS[observation]) * n, p, size=IID_REPLICATES)
        grouped_max = np.max(np.abs(grouped / (len(SEEDS[observation]) * n) - p), axis=1)
        baseline[str(n)] = {
            "per_run": {key: {"median": float(np.median(values)),
                              "q95": float(np.quantile(values, .95))}
                        for key, values in draws.items()},
            "group_max_3d_cell_mean": {
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
    if observation == "nine" and result["references"]["eight"] is None:
        raise ValueError("run eight-detection reference first")
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    model = _case(observation, 1.0).model
    gp_table, gp_strides = _gp_table(model)
    mask_log = model.interval_log_probabilities()
    n_total = sum(CARDINALITIES[observation])
    last_size = CARDINALITIES[observation][-1] + 1
    shape = (N_ALIVE, 3, last_size)
    grid = {rate: _Accumulation(int(np.prod(shape))) for rate in CLUTTER_RATES}
    bases = {rate: _Accumulation(1) for rate in CLUTTER_RATES}
    tempered = {(rate, beta): _Accumulation(3) for rate in CLUTTER_RATES
                for beta in BETAS[:-1]}
    endpoints = [_case(observation, rate).model
                 for rate in (CLUTTER_RATES[0], CLUTTER_RATES[-1])]
    canonical_count = labelled_count = 0
    for block_index, (states, multiplicity, indices) in enumerate(_canonical_blocks(model)):
        n = len(states)
        canonical_count += n
        labelled_count += n * multiplicity
        alive, sixth, last = _counts(states, observation)
        assigned = np.count_nonzero(states >= 0, axis=(1, 2))
        clutter_total = n_total - assigned
        p_d = model.detection_probability
        log_base = (
            mask_log[list(indices)].sum() + log(multiplicity)
            + assigned * log(p_d) + (alive - assigned) * log(1 - p_d)
            + clutter_total * log(model.clutter_intensity)
        )
        log_marginal = _gp_from_table(states, gp_table, gp_strides)
        if block_index in (0, 10_000, 50_000, 90_000):
            direct_base = model.log_existence_prior(states) + model.log_count_density(
                states, include_poisson_void=False
            )
            np.testing.assert_allclose(log_base - log(multiplicity), direct_base,
                                       atol=1e-10, rtol=0)
            np.testing.assert_allclose(log_marginal, model.log_marginal_density(states),
                                       atol=1e-10, rtol=0)
            nominal = model.log_count_density(states, include_poisson_void=False)
            for endpoint in endpoints:
                direct = endpoint.log_count_density(states, include_poisson_void=False)
                np.testing.assert_allclose(
                    direct, nominal + clutter_total * log(endpoint.expected_clutter_per_frame),
                    atol=1e-10, rtol=0,
                )
        cells = (3 * last_size * alive + last_size * sixth + last)
        modes = _mode(alive, sixth)
        for rate in CLUTTER_RATES:
            base_rate = log_base + clutter_total * log(rate)
            bases[rate].update(base_rate, np.zeros(n, dtype=np.int8))
            grid[rate].update(base_rate + log_marginal, cells)
            for beta in BETAS[:-1]:
                tempered[rate, beta].update(base_rate + beta * log_marginal, modes)
        if canonical_count // 50_000_000 > (canonical_count - n) // 50_000_000:
            print(f"{observation} canonical support: {canonical_count:,}/"
                  f"{CANONICAL_SUPPORT[observation]:,}", flush=True)
    if (canonical_count != CANONICAL_SUPPORT[observation]
            or labelled_count != LABELLED_SUPPORT[observation]):
        raise ValueError("frozen canonical or labelled count mismatch")
    rows = []
    for rate in CLUTTER_RATES:
        analytic = _case(observation, rate).model.base_log_normalizer
        enumerated = bases[rate].log_total() - N_FRAMES * rate
        if not np.isclose(enumerated, analytic, atol=1e-8, rtol=0):
            raise ValueError(f"analytic base normalizer mismatch at rate {rate}")
        joint = grid[rate].normalized().reshape(shape)
        rows.append({"clutter_rate": rate, "mode": _from_joint(joint)["mode"],
                     "log_evidence": grid[rate].log_total() - N_FRAMES * rate})
    selected = min(rows, key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]))
    rate = selected["clutter_rate"]
    exact = _from_joint(grid[rate].normalized().reshape(shape))
    curve = [{"beta": beta, "mode": tempered[rate, beta].normalized().tolist()}
             for beta in BETAS[:-1]]
    curve.append({"beta": 1.0, "mode": exact["mode"]})
    valid = (min(exact["mode"][:2]) >= MIN_MODE_MASS
             and curve[2]["mode"][1] <= MAX_LATE_FRACTION * exact["mode"][1])
    result["references"][observation] = {
        "canonical_support_size": canonical_count,
        "labelled_support_size": labelled_count,
        "reference_grid": rows, "selected_rate": rate,
        "exact": exact, "log_evidence": selected["log_evidence"],
        "tempered_mode_curve": curve, "difficulty_valid": bool(valid),
        "iid_baseline": _iid_baseline(exact, observation),
        "cpu_seconds": time.process_time() - started,
    }
    if not valid:
        result["status"] = "density_ladder_invalid_stop"
    _save(output, result)
    print(f"{observation} exact reference: {canonical_count:,} canonical states, "
          f"rate {rate}, modes {exact['mode']}, valid={valid}, "
          f"{result['references'][observation]['cpu_seconds']:.1f}s CPU", flush=True)
    return result


def _run_order() -> list[tuple[str, int, int]]:
    return [(obs, n, seed) for obs in OBSERVATION_ORDER
            for n in PARTICLES for seed in SEEDS[obs]]


def run_one(output: Path, *, observation: str, n_particles: int, seed: int) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP variables to one")
    if (observation, n_particles, seed) not in _run_order():
        raise ValueError("run is outside frozen schedule")
    result = _load(output)
    if any(ref is None or not ref["difficulty_valid"]
           for ref in result["references"].values()):
        raise ValueError("both validated exact references required before sampler")
    if str(seed) in result["runs"][observation][str(n_particles)]:
        print(f"{observation} {n_particles} seed {seed}: reused", flush=True)
        return result
    completed = [(obs, n, s) for obs, n, s in _run_order()
                 if str(s) in result["runs"][obs][str(n)]]
    if (observation, n_particles, seed) != _run_order()[len(completed)]:
        raise ValueError("run frozen observation, particle and seed order")
    _set_cpu_limit(RUN_CPU_CAPS[n_particles])
    ref = result["references"][observation]
    case = _case(observation, ref["selected_rate"])
    observer = _ModeObserver()
    controls = replace(_candidate_controls(seed), n_particles=n_particles)
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
    alive, sixth, last = _counts(sampled.associations, observation)
    last_size = CARDINALITIES[observation][-1] + 1
    joint = np.bincount(3 * last_size * alive + last_size * sixth + last,
                        minlength=N_ALIVE * 3 * last_size).reshape(N_ALIVE, 3, last_size)
    joint = joint / len(sampled.associations)
    errors = _errors(joint, ref["exact"])
    errors["log_evidence_error"] = sampled.log_evidence_estimate - ref["log_evidence"]
    result["runs"][observation][str(n_particles)][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": errors, "stage_mode_trace": observer.records,
    }
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"{observation} {n_particles} seed {seed}: {cpu:.1f}s CPU, "
          f"3D TV {errors['full_3d_joint_tv']:.4f}, "
          f"mode error {errors['maximum_mode_absolute_error']:.4f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    metrics = ("maximum_mode_absolute_error", "full_3d_joint_tv",
               "collapsed_2d_joint_tv", "maximum_3d_cell_absolute_error",
               "alive_total_tv", "sixth_clutter_mean_error", "last_clutter_mean_error")
    comparisons = {}
    for observation in OBSERVATION_ORDER:
        comparisons[observation] = {}
        ref = result["references"][observation]
        truth = np.asarray(ref["exact"]["joint_alive_by_sixth_by_last_clutter"])
        for n in PARTICLES:
            runs = [result["runs"][observation][str(n)][str(seed)]
                    for seed in SEEDS[observation]]
            means = np.asarray([r["joint_alive_by_sixth_by_last_clutter"]
                                for r in runs]).mean(axis=0)
            delta = means - truth
            baseline = ref["iid_baseline"][str(n)]
            q95 = baseline["group_max_3d_cell_mean"]["q95"]
            flagged = [
                {"alive": int(k), "sixth_clutter": int(c), "last_clutter": int(d),
                 "signed_error": float(delta[k, c, d])}
                for k, c, d in np.argwhere(np.abs(delta) > q95)
            ]
            comparisons[observation][str(n)] = {
                "mean_signed_3d_cell_error": delta.tolist(),
                "max_absolute_mean_3d_cell_error": float(np.max(np.abs(delta))),
                "familywise_flagged_cells": flagged,
                "median_absolute_errors": {
                    key: float(median(abs(r["errors"][key]) for r in runs))
                    for key in metrics
                },
                "error_ranges": {
                    key: [min(r["errors"][key] for r in runs),
                          max(r["errors"][key] for r in runs)] for key in metrics
                },
                "iid_q95_exceedances": {
                    key: sum(abs(r["errors"][key]) > baseline["per_run"][key]["q95"]
                             for r in runs) for key in metrics
                },
                "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in runs])),
            }
    high = [comparisons[obs]["2048"] for obs in OBSERVATION_ORDER]
    signed = lambda row: {
        (cell["alive"], cell["sixth_clutter"], cell["last_clutter"],
         int(np.sign(cell["signed_error"]))) for cell in row["familywise_flagged_cells"]
    }
    shared = sorted(signed(high[0]) & signed(high[1]))
    flags = {
        "persistent_mode_tail": all(
            row["iid_q95_exceedances"]["maximum_mode_absolute_error"]
            >= TAIL_MIN_EXCEEDANCES for row in high
        ),
        "persistent_joint_tail": all(
            row["iid_q95_exceedances"]["full_3d_joint_tv"]
            >= TAIL_MIN_EXCEEDANCES for row in high
        ),
        "shared_cell_shift": bool(shared),
    }
    result["comparison"] = comparisons
    result["diagnostic_flags"] = flags
    result["shared_signed_cells"] = [
        {"alive": k, "sixth_clutter": c, "last_clutter": d, "sign": sign}
        for k, c, d, sign in shared
    ]
    result["status"] = ("density_ladder_persistent_absolute_signal"
                        if any(flags.values()) else "density_ladder_no_persistent_flag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--observation", choices=OBSERVATION_ORDER, required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11r_eight_frame_density_ladder.json"))
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
