"""Frozen ten-detection middle-frame absolute audit with exact references."""

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
)
from .wp11r_eight_frame_density_ladder import _ModeObserver

COMMON = (
    ((-0.019, 0.001),), ((-0.014, 0.001),), ((-0.007, 0.002),), (),
    ((0.001, 0.000),), ((-0.005, 0.002), (0.006, -0.001)),
    ((0.012, -0.001),), ((-0.011, 0.002), (0.018, -0.001)),
)
OBSERVATIONS = {
    "early": COMMON[:2] + (((-0.007, 0.002), (0.005, -0.001)),) + COMMON[3:],
    "middle": COMMON[:4] + (((0.001, 0.000), (-0.011, 0.002)),) + COMMON[5:],
}
OBSERVATION_HASHES = {
    "early": "6b460f3d90d695f03e6dd836a8439e263604d6b36700dfcae64e9afdda5b6676",
    "middle": "961a3ee11c17233630ca7e23dab61ccbfcad933cdde1def1ee9a41cf65e13bd9",
}
CARDINALITIES = {
    "early": (1, 1, 2, 0, 1, 2, 1, 2),
    "middle": (1, 1, 1, 0, 2, 2, 1, 2),
}
CANONICAL_SUPPORT = {"early": 1_679_234_508, "middle": 1_878_017_514}
LABELLED_SUPPORT = {"early": 30_414_095_341, "middle": 34_201_578_205}
OBSERVATION_ORDER = ("early", "middle")
MASK_MULTISET_COUNT = 91_390
N_SLOTS = 4
N_FRAMES = 8
N_ALIVE = 33
FRAME_INDEX = 5
MIDDLE_FRAME_INDEX = {"early": 2, "middle": 4}
LOW_ALIVE_MAX = 15
HIGH_ALIVE_MIN = 16
CLUTTER_RATES = (0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
MIN_MODE_MASS = 0.10
MAX_LATE_FRACTION = 0.50
SEEDS = {"early": (1180, 1181, 1182, 1183, 1184, 1185),
         "middle": (1190, 1191, 1192, 1193, 1194, 1195)}
PARTICLES = (512, 2_048)
IID_SEEDS = {"early": 34_501, "middle": 34_507}
IID_REPLICATES = 1_000
REFERENCE_CPU_CAP = 5_000
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
        "purpose": "paired designed middle-frame ten-detection four-slot/eight-frame absolute audit",
        "observation_hashes": OBSERVATION_HASHES,
        "observation_order": list(OBSERVATION_ORDER),
        "cardinalities": {k: list(v) for k, v in CARDINALITIES.items()},
        "middle_frame_index": MIDDLE_FRAME_INDEX,
        "canonical_support": CANONICAL_SUPPORT,
        "labelled_support": LABELLED_SUPPORT,
        "mask_multisets": MASK_MULTISET_COUNT,
        "reference_method": (
            "exact canonical interval-mask multiset sum with permutation multiplicity; "
            "independent combinatorial support derivation; directly checked one-slot GP "
            "likelihood lookup and selected direct collapsed scores; endpoint rate "
            "and eleven analytic base-normalizer checks; both references before sampler"
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
        "joint_cells": "33 alive totals by 3 added-middle clutter by 3 frame-six clutter "
                       "by 3 final-frame clutter: 891 cells in each case",
        "comparison": (
            "per-run 4D joint TV, maximum mode and cell errors versus 1000 exact IID "
            "multinomial q95 per case and particle count; also 3D collapsed joint, "
            "alive and three clutter means; six-run mean signed 4D and collapsed 3D "
            "cells versus familywise q95 from 1000 exact groups of six"
        ),
        "persistence_rule": (
            "at 2048 particles: persistent_mode_tail and persistent_joint_tail "
            "require >=2/6 q95 exceedances in both cases; shared_cell_shift "
            "requires the same collapsed 3D (alive, frame-six clutter, final clutter) "
            "cell to exceed each familywise six-run mean q95 with the same sign; "
            "designed cases share nine observations and are dependent"
        ),
        "reference_before_sampler": True,
        "beta_grid": list(BETAS),
        "seeds": {k: list(v) for k, v in SEEDS.items()},
        "particles": list(PARTICLES),
        "iid_replicates": IID_REPLICATES, "iid_seeds": IID_SEEDS,
        "run_order": (
            "early reference then middle reference; early 512/2048 seeds 1180..1185; "
            "middle 512/2048 seeds 1190..1195"
        ),
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap_each": REFERENCE_CPU_CAP,
        "run_cpu_caps": {str(n): cap for n, cap in RUN_CPU_CAPS.items()},
        "interpretation": (
            "designed ten-detection middle-frame bridge; descriptive IID references, "
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
        "experiment": "wp11s_middle_frame_density_audit",
        "protocol": protocol, "cases": cases,
        "references": {key: None for key in OBSERVATION_ORDER},
        "runs": {key: {str(n): {} for n in PARTICLES} for key in OBSERVATION_ORDER},
        "comparison": None, "status": "in_progress",
    }
    _save(output, result)
    return result


def _counts(
    states: np.ndarray, observation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    middle_index = MIDDLE_FRAME_INDEX[observation]
    middle = CARDINALITIES[observation][middle_index] - np.count_nonzero(
        states[:, middle_index] >= 0, axis=1
    )
    sixth = CARDINALITIES[observation][FRAME_INDEX] - np.count_nonzero(
        states[:, FRAME_INDEX] >= 0, axis=1
    )
    last = CARDINALITIES[observation][-1] - np.count_nonzero(
        states[:, -1] >= 0, axis=1
    )
    return alive, middle, sixth, last


def _mode(alive: np.ndarray, sixth: np.ndarray) -> np.ndarray:
    mode = np.full(len(alive), 2, dtype=np.int8)
    mode[(alive <= LOW_ALIVE_MAX) & (sixth >= 1)] = 0
    mode[(alive >= HIGH_ALIVE_MIN) & (sixth == 0)] = 1
    return mode


def _from_joint(joint: np.ndarray) -> dict[str, Any]:
    a = joint[:LOW_ALIVE_MAX + 1, :, 1:, :].sum()
    b = joint[HIGH_ALIVE_MIN:, :, 0, :].sum()
    return {
        "mode": [float(a), float(b), float(joint.sum() - a - b)],
        "joint_alive_by_middle_by_sixth_by_last_clutter": joint.tolist(),
        "joint_alive_by_sixth_by_last_clutter": joint.sum(axis=1).tolist(),
        "alive": joint.sum(axis=(1, 2, 3)).tolist(),
        "middle_clutter": joint.sum(axis=(0, 2, 3)).tolist(),
        "sixth_clutter": joint.sum(axis=(0, 1, 3)).tolist(),
        "last_clutter": joint.sum(axis=(0, 1, 2)).tolist(),
    }


def _errors(observed: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    truth = np.asarray(exact["joint_alive_by_middle_by_sixth_by_last_clutter"])
    delta = observed - truth
    mode_delta = np.asarray(_from_joint(observed)["mode"]) - exact["mode"]
    return {
        "mode_A_signed_error": float(mode_delta[0]),
        "mode_B_signed_error": float(mode_delta[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(mode_delta))),
        "full_4d_joint_tv": float(np.abs(delta).sum() / 2),
        "collapsed_3d_joint_tv": float(np.abs(delta.sum(axis=1)).sum() / 2),
        "maximum_4d_cell_absolute_error": float(np.max(np.abs(delta))),
        "alive_total_tv": float(np.abs(delta.sum(axis=(1, 2, 3))).sum() / 2),
        "middle_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[1]), delta.sum(axis=(0, 2, 3))
        )),
        "sixth_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[2]), delta.sum(axis=(0, 1, 3))
        )),
        "last_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[3]), delta.sum(axis=(0, 1, 2))
        )),
    }


def _iid_baseline(exact: dict[str, Any], observation: str) -> dict[str, Any]:
    rng = np.random.default_rng(IID_SEEDS[observation])
    shape = (N_ALIVE, 3, 3, 3)
    p = np.asarray(exact["joint_alive_by_middle_by_sixth_by_last_clutter"]).reshape(-1)
    p /= p.sum()
    metrics = ("maximum_mode_absolute_error", "full_4d_joint_tv",
               "collapsed_3d_joint_tv", "maximum_4d_cell_absolute_error",
               "alive_total_tv", "middle_clutter_mean_error",
               "sixth_clutter_mean_error", "last_clutter_mean_error")
    baseline = {}
    for n in PARTICLES:
        draws = {key: [] for key in metrics}
        for counts in rng.multinomial(n, p, size=IID_REPLICATES):
            errors = _errors(counts.reshape(shape) / n, exact)
            for key in metrics:
                draws[key].append(abs(errors[key]))
        grouped = rng.multinomial(len(SEEDS[observation]) * n, p, size=IID_REPLICATES)
        grouped_max = np.max(np.abs(grouped / (len(SEEDS[observation]) * n) - p), axis=1)
        collapsed = grouped.reshape(IID_REPLICATES, *shape).sum(axis=2).reshape(
            IID_REPLICATES, -1
        ) / (len(SEEDS[observation]) * n)
        p3 = p.reshape(shape).sum(axis=1).reshape(-1)
        collapsed_max = np.max(np.abs(collapsed - p3), axis=1)
        baseline[str(n)] = {
            "per_run": {key: {"median": float(np.median(values)),
                              "q95": float(np.quantile(values, .95))}
                        for key, values in draws.items()},
            "group_max_4d_cell_mean": {
                "median": float(np.median(grouped_max)),
                "q95": float(np.quantile(grouped_max, .95)),
            },
            "group_max_collapsed_3d_cell_mean": {
                "median": float(np.median(collapsed_max)),
                "q95": float(np.quantile(collapsed_max, .95)),
            },
        }
    return baseline


def _reweight_clutter(
    accumulation: _Accumulation, rate: float, n_total: int, shape: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    """Recover an exact rate grid from clutter-count coefficients at rate one."""
    coefficients = accumulation.hist.reshape(n_total + 1, *shape)
    weighted = np.tensordot(rate ** np.arange(n_total + 1), coefficients, axes=(0, 0))
    total = float(weighted.sum())
    return weighted / total, float(accumulation.maximum + log(total))


def run_reference(output: Path, observation: str) -> dict[str, Any]:
    result = _load(output)
    if result["references"][observation] is not None:
        print(f"{observation} exact reference: reused", flush=True)
        return result
    if observation == "middle" and result["references"]["early"] is None:
        raise ValueError("run early-middle reference first")
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    model = _case(observation, 1.0).model
    gp_table, gp_strides = _gp_table(model)
    mask_log = model.interval_log_probabilities()
    n_total = sum(CARDINALITIES[observation])
    shape = (N_ALIVE, 3, 3, 3)
    n_cells = int(np.prod(shape))
    grid = _Accumulation((n_total + 1) * n_cells)
    bases = _Accumulation(n_total + 1)
    tempered = {beta: _Accumulation((n_total + 1) * 3) for beta in BETAS[:-1]}
    endpoints = [_case(observation, rate).model
                 for rate in (CLUTTER_RATES[0], CLUTTER_RATES[-1])]
    canonical_count = labelled_count = 0
    for block_index, (states, multiplicity, indices) in enumerate(_canonical_blocks(model)):
        n = len(states)
        canonical_count += n
        labelled_count += n * multiplicity
        alive, middle, sixth, last = _counts(states, observation)
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
        cells = 27 * alive + 9 * middle + 3 * sixth + last
        modes = _mode(alive, sixth)
        bases.update(log_base, clutter_total)
        grid.update(log_base + log_marginal, n_cells * clutter_total + cells)
        for beta in BETAS[:-1]:
            tempered[beta].update(log_base + beta * log_marginal,
                                  3 * clutter_total + modes)
        if canonical_count // 50_000_000 > (canonical_count - n) // 50_000_000:
            print(f"{observation} canonical support: {canonical_count:,}/"
                  f"{CANONICAL_SUPPORT[observation]:,}", flush=True)
    if (canonical_count != CANONICAL_SUPPORT[observation]
            or labelled_count != LABELLED_SUPPORT[observation]):
        raise ValueError("frozen canonical or labelled count mismatch")
    rows = []
    for rate in CLUTTER_RATES:
        analytic = _case(observation, rate).model.base_log_normalizer
        _, base_log_total = _reweight_clutter(bases, rate, n_total, (1,))
        enumerated = base_log_total - N_FRAMES * rate
        if not np.isclose(enumerated, analytic, atol=1e-8, rtol=0):
            raise ValueError(f"analytic base normalizer mismatch at rate {rate}")
        joint, grid_log_total = _reweight_clutter(grid, rate, n_total, shape)
        rows.append({"clutter_rate": rate, "mode": _from_joint(joint)["mode"],
                     "log_evidence": grid_log_total - N_FRAMES * rate})
    selected = min(rows, key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]))
    rate = selected["clutter_rate"]
    selected_joint, _ = _reweight_clutter(grid, rate, n_total, shape)
    exact = _from_joint(selected_joint)
    curve = [{"beta": beta, "mode": _reweight_clutter(tempered[beta], rate, n_total,
                                                       (3,))[0].tolist()}
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
        result["status"] = "middle_density_invalid_stop"
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
    alive, middle, sixth, last = _counts(sampled.associations, observation)
    joint = np.bincount(27 * alive + 9 * middle + 3 * sixth + last,
                        minlength=N_ALIVE * 27).reshape(N_ALIVE, 3, 3, 3)
    joint = joint / len(sampled.associations)
    errors = _errors(joint, ref["exact"])
    errors["log_evidence_error"] = sampled.log_evidence_estimate - ref["log_evidence"]
    result["runs"][observation][str(n_particles)][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "joint_alive_by_middle_by_sixth_by_last_clutter": joint.tolist(),
        "errors": errors, "stage_mode_trace": observer.records,
    }
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"{observation} {n_particles} seed {seed}: {cpu:.1f}s CPU, "
          f"4D TV {errors['full_4d_joint_tv']:.4f}, "
          f"mode error {errors['maximum_mode_absolute_error']:.4f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    metrics = ("maximum_mode_absolute_error", "full_4d_joint_tv",
               "collapsed_3d_joint_tv", "maximum_4d_cell_absolute_error",
               "alive_total_tv", "middle_clutter_mean_error",
               "sixth_clutter_mean_error", "last_clutter_mean_error")
    comparisons = {}
    for observation in OBSERVATION_ORDER:
        comparisons[observation] = {}
        ref = result["references"][observation]
        truth = np.asarray(ref["exact"]["joint_alive_by_middle_by_sixth_by_last_clutter"])
        for n in PARTICLES:
            runs = [result["runs"][observation][str(n)][str(seed)]
                    for seed in SEEDS[observation]]
            means = np.asarray([r["joint_alive_by_middle_by_sixth_by_last_clutter"]
                                for r in runs]).mean(axis=0)
            delta = means - truth
            baseline = ref["iid_baseline"][str(n)]
            q95 = baseline["group_max_4d_cell_mean"]["q95"]
            flagged = [
                {"alive": int(k), "middle_clutter": int(m),
                 "sixth_clutter": int(c), "last_clutter": int(d),
                 "signed_error": float(delta[k, m, c, d])}
                for k, m, c, d in np.argwhere(np.abs(delta) > q95)
            ]
            collapsed_delta = delta.sum(axis=1)
            collapsed_q95 = baseline["group_max_collapsed_3d_cell_mean"]["q95"]
            collapsed_flagged = [
                {"alive": int(k), "sixth_clutter": int(c), "last_clutter": int(d),
                 "signed_error": float(collapsed_delta[k, c, d])}
                for k, c, d in np.argwhere(np.abs(collapsed_delta) > collapsed_q95)
            ]
            comparisons[observation][str(n)] = {
                "mean_signed_4d_cell_error": delta.tolist(),
                "max_absolute_mean_4d_cell_error": float(np.max(np.abs(delta))),
                "familywise_flagged_4d_cells": flagged,
                "mean_signed_collapsed_3d_cell_error": collapsed_delta.tolist(),
                "max_absolute_mean_collapsed_3d_cell_error": float(
                    np.max(np.abs(collapsed_delta))
                ),
                "familywise_flagged_collapsed_3d_cells": collapsed_flagged,
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
         int(np.sign(cell["signed_error"])))
        for cell in row["familywise_flagged_collapsed_3d_cells"]
    }
    shared = sorted(signed(high[0]) & signed(high[1]))
    flags = {
        "persistent_mode_tail": all(
            row["iid_q95_exceedances"]["maximum_mode_absolute_error"]
            >= TAIL_MIN_EXCEEDANCES for row in high
        ),
        "persistent_joint_tail": all(
            row["iid_q95_exceedances"]["full_4d_joint_tv"]
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
    result["status"] = ("middle_density_persistent_absolute_signal"
                        if any(flags.values()) else "middle_density_no_persistent_flag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--observation", choices=OBSERVATION_ORDER, required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11s_middle_frame_density_audit.json"))
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
