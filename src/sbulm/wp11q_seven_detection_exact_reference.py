"""Frozen seven-detection four-slot/eight-frame exact reference."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from dataclasses import replace
from itertools import combinations_with_replacement
from itertools import product as cartesian_product
from math import factorial, log, prod
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

OBSERVATION = (
    ((-0.019, 0.001),), ((-0.014, 0.001),), ((-0.007, 0.002),), (), ((0.001, 0.000),),
    ((-0.005, 0.002), (0.006, -0.001)), ((0.012, -0.001),), (),
)
OBSERVATION_HASH = "11a2f6f7c529f080ed0fdbb3eaf3da39c2b9170ce660e95f0b119de5d241734b"
CARDINALITIES = (1, 1, 1, 0, 1, 2, 1, 0)
CANONICAL_SUPPORT_SIZE = 126_031_206
LABELLED_SUPPORT_SIZE = 2_460_244_269
MASK_MULTISET_COUNT = 91_390
N_SLOTS = 4
N_FRAMES = 8
FRAME_INDEX = 5
N_ALIVE = 33
N_CLUTTER = 3
LOW_ALIVE_MAX = 15
HIGH_ALIVE_MIN = 16
CLUTTER_RATES = (0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
MIN_MODE_MASS = 0.10
MAX_LATE_FRACTION = 0.50
SEEDS = (1120, 1121, 1122, 1123, 1124, 1125)
PARTICLES = (512, 2_048)
IID_REPLICATES = 1_000
IID_SEED = 34_351
REFERENCE_CPU_CAP = 12_000
RUN_CPU_CAPS = {512: 600, 2_048: 1_600}
BATCH_SIZE = 20_000
TAIL_MIN_EXCEEDANCES = 3


def _case(rate: float) -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(N_FRAMES, dtype=float) * 0.001,
        detections=tuple(np.asarray(frame, dtype=float).reshape(-1, 2)
                         for frame in OBSERVATION),
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
        "purpose": "independent exact seven-detection/four-slot/eight-frame late-mode reference",
        "observation_hash": OBSERVATION_HASH,
        "cardinalities": list(CARDINALITIES),
        "canonical_support_size": CANONICAL_SUPPORT_SIZE,
        "labelled_support_size": LABELLED_SUPPORT_SIZE,
        "mask_multisets": MASK_MULTISET_COUNT,
        "exact_method": (
            "enumerate nondecreasing slot-interval masks, retain all frame assignments, "
            "weight by 4! divided by factorials of repeated interval-mask counts"
        ),
        "clutter_rates": list(CLUTTER_RATES),
        "rate_selection": "maximize min(exact P(A), exact P(B)); tie to smaller rate",
        "mode_A": "alive slot-frames <= 15 and frame-six clutter >= 1",
        "mode_B": "alive slot-frames >= 16 and frame-six clutter == 0",
        "mode_other": "all remaining states",
        "difficulty_validation": (
            "selected exact P(A) and P(B) each >= 0.10 and beta=0.5 exact P(B) "
            "<= half beta=1 P(B); fail and stop sampler without modifying case"
        ),
        "joint_cells": "all 33 alive totals (0..32) by frame-six clutter (0..2)",
        "reference_checks": (
            "canonical weighted versus full labelled small-case sum; canonical and "
            "labelled support counts; eleven analytic base normalizers; direct "
            "endpoint Poisson-rate reweighting and GP lookup checks on deterministic blocks"
        ),
        "comparison": (
            "six independent unchanged WP11g 10% four-path PG seeds at 512 then 2048; "
            "per-run full joint TV, maximum cell and maximum mode error versus "
            "1000 exact multinomial IID q95; six-run mean maximum signed cell "
            "versus familywise q95 from 1000 six-draw exact groups"
        ),
        "diagnostic_flags": (
            "at 2048, mode_error_tail or joint_tail if >=3/6 runs exceed their "
            "corresponding IID q95; cell_shift if maximum absolute six-run mean "
            "cell residual exceeds the familywise six-draw q95 and its sign "
            "agrees in both ordered three-seed halves; no move selection"
        ),
        "reference_before_sampler": True,
        "beta_grid": list(BETAS),
        "seeds": list(SEEDS), "particles": list(PARTICLES),
        "run_order": "exact reference checkpoint, 512 seeds 1120..1125, 2048 seeds 1120..1125",
        "iid_seed": IID_SEED, "iid_replicates": IID_REPLICATES,
        "thread_env": {key: "1" for key in THREAD_ENV},
        "reference_cpu_cap": REFERENCE_CPU_CAP,
        "run_cpu_caps": {str(n): cap for n, cap in RUN_CPU_CAPS.items()},
        "interpretation": (
            "eight-frame seven-detection exact bridge, not WP11k's 19-detection "
            "primary; descriptive IID comparisons, no 8192 or holdout"
        ),
    }


def _metadata() -> dict[str, Any]:
    case = _case(CLUTTER_RATES[0])
    info = _case_metadata(case)
    if (info["detections_sha256"] != OBSERVATION_HASH
            or tuple(info["detection_cardinalities"]) != CARDINALITIES):
        raise ValueError("frozen observation changed")
    return {"observation": info,
            "models": [_case_metadata(_case(rate)) for rate in CLUTTER_RATES]}


def _load(output: Path) -> dict[str, Any]:
    protocol, cases = _protocol(), _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != cases:
            raise ValueError("checkpoint belongs to a different frozen protocol")
        return result
    result = {
        "experiment": "wp11q_seven_detection_exact_reference", "protocol": protocol,
        "cases": cases, "reference": None,
        "runs": {str(n): {} for n in PARTICLES},
        "comparison": None, "status": "in_progress",
    }
    _save(output, result)
    return result


def _mask_multiplicity(indices: tuple[int, ...]) -> int:
    counts = np.unique(indices, return_counts=True)[1]
    return factorial(len(indices)) // prod(factorial(int(count)) for count in counts)


def _canonical_blocks(
    model: SlotExistenceMaternModel,
) -> Iterator[tuple[np.ndarray, int, tuple[int, ...]]]:
    """Each block represents every labelled permutation of its mask multiset."""
    masks = model.interval_masks()
    for indices in combinations_with_replacement(range(len(masks)), model.n_slots):
        selected = masks[list(indices)].T
        states = tuple(model.frame_states(selected[t], n)
                       for t, n in enumerate(model.sequence.cardinalities))
        sizes = tuple(len(frame) for frame in states)
        total = prod(sizes)
        multiplicity = _mask_multiplicity(indices)
        for start in range(0, total, BATCH_SIZE):
            frame_indices = np.unravel_index(
                np.arange(start, min(start + BATCH_SIZE, total)), sizes
            )
            yield (np.stack([frame[frame_indices[t]] for t, frame in enumerate(states)],
                            axis=1), multiplicity, indices)


def _gp_table(model: SlotExistenceMaternModel) -> tuple[np.ndarray, np.ndarray]:
    """Score every possible single-slot detection sequence exactly once."""
    radices = np.asarray(model.sequence.cardinalities, dtype=int) + 1
    strides = np.cumprod(np.r_[1, radices[:-1]])
    patterns = np.asarray(list(cartesian_product(*(range(n) for n in radices))),
                          dtype=np.int64)
    states = np.full((len(patterns), model.sequence.n_frames, model.n_slots), ABSENT)
    states[:, :, 0] = np.where(patterns > 0, patterns - 1, -1)
    codes = patterns @ strides
    table = np.empty(int(np.prod(radices)), dtype=float)
    table[codes] = model.log_marginal_density(states)
    return table, strides


def _gp_from_table(states: np.ndarray, table: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """The collapsed GP factorizes across slots; absence and misses score alike."""
    digits = np.where(states >= 0, states + 1, 0)
    codes = np.einsum("ntk,t->nk", digits, strides)
    return table[codes].sum(axis=1)


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


def _from_joint(joint: np.ndarray) -> dict[str, Any]:
    a = joint[:LOW_ALIVE_MAX + 1, 1:].sum()
    b = joint[HIGH_ALIVE_MIN:, 0].sum()
    return {
        "mode": [float(a), float(b), float(joint.sum() - a - b)],
        "joint_alive_by_clutter": joint.tolist(),
        "alive": joint.sum(axis=1).tolist(),
        "clutter": joint.sum(axis=0).tolist(),
    }


def _errors(observed: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    truth = np.asarray(exact["joint_alive_by_clutter"])
    delta = observed - truth
    mode_delta = np.asarray(_from_joint(observed)["mode"]) - exact["mode"]
    return {
        "mode_A_signed_error": float(mode_delta[0]),
        "mode_B_signed_error": float(mode_delta[1]),
        "maximum_mode_absolute_error": float(np.max(np.abs(mode_delta))),
        "full_joint_tv": float(np.abs(delta).sum() / 2),
        "maximum_cell_absolute_error": float(np.max(np.abs(delta))),
        "alive_total_tv": float(np.abs(delta.sum(axis=1)).sum() / 2),
        "clutter_mean_error": float(np.dot(np.arange(N_CLUTTER), delta.sum(axis=0))),
    }


class _Accumulation:
    """Stable weighted histogram sum across batches with changing maxima."""

    def __init__(self, size: int) -> None:
        self.maximum = -np.inf
        self.hist = np.zeros(size, dtype=float)

    def update(self, scores: np.ndarray, cells: np.ndarray) -> None:
        block_max = float(np.max(scores))
        new_max = max(block_max, self.maximum)
        self.hist *= np.exp(self.maximum - new_max)
        self.hist += np.bincount(
            cells, weights=np.exp(scores - new_max), minlength=len(self.hist)
        )
        self.maximum = new_max

    def log_total(self) -> float:
        return float(self.maximum + np.log(self.hist.sum()))

    def normalized(self) -> np.ndarray:
        return self.hist / self.hist.sum()


def _iid_baseline(exact: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(IID_SEED)
    p = np.asarray(exact["joint_alive_by_clutter"]).reshape(-1)
    p /= p.sum()
    baseline = {}
    for n in PARTICLES:
        draws = {key: [] for key in (
            "maximum_mode_absolute_error", "full_joint_tv",
            "maximum_cell_absolute_error", "alive_total_tv", "clutter_mean_error",
        )}
        for counts in rng.multinomial(n, p, size=IID_REPLICATES):
            errors = _errors(counts.reshape(N_ALIVE, N_CLUTTER) / n, exact)
            for key, values in draws.items():
                values.append(abs(errors[key]))
        grouped = rng.multinomial(len(SEEDS) * n, p, size=IID_REPLICATES)
        grouped_max = np.max(np.abs(grouped / (len(SEEDS) * n) - p), axis=1)
        baseline[str(n)] = {
            "per_run": {key: {"median": float(np.median(values)),
                              "q95": float(np.quantile(values, .95))}
                        for key, values in draws.items()},
            "group_max_cell_mean": {
                "median": float(np.median(grouped_max)),
                "q95": float(np.quantile(grouped_max, .95)),
            },
        }
    return baseline


def run_reference(output: Path) -> dict[str, Any]:
    result = _load(output)
    if result["reference"] is not None:
        print("exact reference: reused", flush=True)
        return result
    _set_cpu_limit(REFERENCE_CPU_CAP)
    started = time.process_time()
    model = _case(1.0).model
    gp_table, gp_strides = _gp_table(model)
    mask_log = model.interval_log_probabilities()
    p_d = model.detection_probability
    total_detections = sum(CARDINALITIES)
    grid = {rate: _Accumulation(N_ALIVE * N_CLUTTER) for rate in CLUTTER_RATES}
    bases = {rate: _Accumulation(1) for rate in CLUTTER_RATES}
    tempered = {(rate, beta): _Accumulation(3) for rate in CLUTTER_RATES
                for beta in BETAS[:-1]}
    reference_models = [_case(rate).model for rate in (CLUTTER_RATES[0],
                                                       CLUTTER_RATES[-1])]
    canonical_count = labelled_count = 0
    for block_index, (states, multiplicity, indices) in enumerate(_canonical_blocks(model)):
        n = len(states)
        canonical_count += n
        labelled_count += n * multiplicity
        alive, clutter = _counts(states)
        assigned = np.count_nonzero(states >= 0, axis=(1, 2))
        total_clutter = total_detections - assigned
        log_base = (
            mask_log[list(indices)].sum() + log(multiplicity)
            + assigned * log(p_d) + (alive - assigned) * log(1 - p_d)
            + total_clutter * log(model.clutter_intensity)
        )
        if block_index in (0, 10_000, 50_000, 90_000):
            raw = model.log_existence_prior(states) + model.log_count_density(
                states, include_poisson_void=False
            )
            np.testing.assert_allclose(log_base - log(multiplicity), raw, atol=1e-10)
            np.testing.assert_allclose(
                _gp_from_table(states, gp_table, gp_strides),
                model.log_marginal_density(states), atol=1e-10, rtol=0,
            )
            for endpoint in reference_models:
                direct = endpoint.log_count_density(states, include_poisson_void=False)
                nominal = model.log_count_density(states, include_poisson_void=False)
                np.testing.assert_allclose(
                    direct, nominal + total_clutter * log(endpoint.expected_clutter_per_frame),
                    atol=1e-10, rtol=0,
                )
        log_marginal = _gp_from_table(states, gp_table, gp_strides)
        cells = N_CLUTTER * alive + clutter
        modes = _mode(alive, clutter)
        for rate in CLUTTER_RATES:
            base_rate = log_base + total_clutter * log(rate)
            bases[rate].update(base_rate, np.zeros(n, dtype=np.int8))
            grid[rate].update(base_rate + log_marginal, cells)
            for beta in BETAS[:-1]:
                tempered[rate, beta].update(base_rate + beta * log_marginal, modes)
        if canonical_count // 2_000_000 > (canonical_count - n) // 2_000_000:
            print(f"canonical support: {canonical_count:,}/{CANONICAL_SUPPORT_SIZE:,}",
                  flush=True)
    if (canonical_count != CANONICAL_SUPPORT_SIZE
            or labelled_count != LABELLED_SUPPORT_SIZE):
        raise ValueError("canonical or labelled support count differs from frozen value")
    rows = []
    for rate in CLUTTER_RATES:
        analytic = _case(rate).model.base_log_normalizer
        enumerated = bases[rate].log_total() - N_FRAMES * rate
        if not np.isclose(enumerated, analytic, atol=1e-8, rtol=0):
            raise ValueError(f"analytic base normalizer mismatch at rate {rate}")
        joint = grid[rate].normalized().reshape(N_ALIVE, N_CLUTTER)
        rows.append({"clutter_rate": rate, "mode": _from_joint(joint)["mode"],
                     "log_evidence": grid[rate].log_total() - N_FRAMES * rate})
    selected = min(rows, key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]))
    rate = selected["clutter_rate"]
    exact = _from_joint(grid[rate].normalized().reshape(N_ALIVE, N_CLUTTER))
    curve = [{"beta": beta, "mode": tempered[rate, beta].normalized().tolist()}
             for beta in BETAS[:-1]]
    curve.append({"beta": 1.0, "mode": exact["mode"]})
    valid = (min(exact["mode"][:2]) >= MIN_MODE_MASS
             and curve[2]["mode"][1] <= MAX_LATE_FRACTION * exact["mode"][1])
    result["reference"] = {
        "canonical_support_size": canonical_count,
        "labelled_support_size": labelled_count,
        "mask_multiset_count": MASK_MULTISET_COUNT,
        "reference_grid": rows, "selected_rate": rate,
        "exact": exact, "log_evidence": selected["log_evidence"],
        "tempered_mode_curve": curve, "difficulty_valid": bool(valid),
        "iid_baseline": _iid_baseline(exact),
        "cpu_seconds": time.process_time() - started,
    }
    if not valid:
        result["status"] = "frozen_eight_frame_bridge_invalid_stop"
    _save(output, result)
    print(f"exact reference: {canonical_count:,} canonical states, "
          f"rate {rate}, modes {exact['mode']}, valid={valid}, "
          f"{result['reference']['cpu_seconds']:.1f}s CPU", flush=True)
    return result


class _ModeObserver:
    """Passive mode trace; it does not consume random numbers."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def begin_stage(self, step: int, beta: float) -> None:
        self.records.append({"step": step, "beta": float(beta),
                             "accepted_A_to_B": {"pg": 0, "other": 0},
                             "accepted_B_to_A": {"pg": 0, "other": 0}})

    def record_resampling(
        self, associations: np.ndarray, weights: np.ndarray, ancestors: np.ndarray
    ) -> None:
        modes = _mode(*_counts(associations))
        before = np.bincount(modes, weights=weights, minlength=3)
        after = np.bincount(modes[ancestors], minlength=3) / len(ancestors)
        self.records[-1].update({"weighted_mode": before.tolist(),
                                 "resampled_mode": after.tolist(),
                                 "resampling_mode_tv": float(np.abs(before - after).sum() / 2)})

    def record_pair(
        self, before: np.ndarray, proposed: np.ndarray,
        accepted: np.ndarray, pg_selected: np.ndarray,
    ) -> None:
        source, destination = _mode(*_counts(before)), _mode(*_counts(proposed))
        for label, selected in (("pg", pg_selected), ("other", ~pg_selected)):
            self.records[-1]["accepted_A_to_B"][label] += int(np.count_nonzero(
                accepted & selected & (source == 0) & (destination == 1)))
            self.records[-1]["accepted_B_to_A"][label] += int(np.count_nonzero(
                accepted & selected & (source == 1) & (destination == 0)))

    def end_stage(self, associations: np.ndarray) -> None:
        self.records[-1]["post_mode"] = (
            np.bincount(_mode(*_counts(associations)), minlength=3) / len(associations)
        ).tolist()


def run_one(output: Path, *, n_particles: int, seed: int) -> dict[str, Any]:
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    if n_particles not in PARTICLES or seed not in SEEDS:
        raise ValueError("run is outside frozen schedule")
    result = _load(output)
    ref = result["reference"]
    if ref is None or not ref["difficulty_valid"]:
        raise ValueError("validated exact reference is required before sampler")
    if str(seed) in result["runs"][str(n_particles)]:
        print(f"{n_particles} seed {seed}: reused", flush=True)
        return result
    order = [(n, s) for n in PARTICLES for s in SEEDS]
    completed = [(n, s) for n, s in order if str(s) in result["runs"][str(n)]]
    if (n_particles, seed) != order[len(completed)]:
        raise ValueError("run frozen particle and seed order")
    _set_cpu_limit(RUN_CPU_CAPS[n_particles])
    case = _case(ref["selected_rate"])
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
    alive, clutter = _counts(sampled.associations)
    joint = np.bincount(N_CLUTTER * alive + clutter,
                        minlength=N_ALIVE * N_CLUTTER).reshape(N_ALIVE, N_CLUTTER)
    joint = joint / len(sampled.associations)
    errors = _errors(joint, ref["exact"])
    errors["log_evidence_error"] = sampled.log_evidence_estimate - ref["log_evidence"]
    result["runs"][str(n_particles)][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "joint_alive_by_clutter": joint.tolist(), "errors": errors,
        "stage_mode_trace": observer.records,
    }
    if len(completed) + 1 == len(order):
        _finish(result)
    _save(output, result)
    print(f"{n_particles} seed {seed}: {cpu:.1f}s CPU, "
          f"joint TV {errors['full_joint_tv']:.4f}, "
          f"mode error {errors['maximum_mode_absolute_error']:.4f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    ref = result["reference"]
    truth = np.asarray(ref["exact"]["joint_alive_by_clutter"])
    comparison = {}
    for n in PARTICLES:
        runs = [result["runs"][str(n)][str(seed)] for seed in SEEDS]
        means = np.asarray([r["joint_alive_by_clutter"] for r in runs]).mean(axis=0)
        delta = means - truth
        q95 = ref["iid_baseline"][str(n)]["group_max_cell_mean"]["q95"]
        flagged = []
        for k, c in np.argwhere(np.abs(delta) > q95):
            first = np.asarray([r["joint_alive_by_clutter"] for r in runs[:3]])
            last = np.asarray([r["joint_alive_by_clutter"] for r in runs[3:]])
            first_error = float(first[:, k, c].mean() - truth[k, c])
            last_error = float(last[:, k, c].mean() - truth[k, c])
            flagged.append({"alive": int(k), "clutter": int(c),
                            "signed_error": float(delta[k, c]),
                            "same_sign_halves": bool(first_error * last_error > 0)})
        keys = ("maximum_mode_absolute_error", "full_joint_tv",
                "maximum_cell_absolute_error", "alive_total_tv", "clutter_mean_error")
        comparison[str(n)] = {
            "mean_signed_cell_error": delta.tolist(),
            "max_absolute_mean_cell_error": float(np.max(np.abs(delta))),
            "familywise_flagged_cells": flagged,
            "median_absolute_errors": {
                key: float(median(abs(r["errors"][key]) for r in runs)) for key in keys
            },
            "error_ranges": {
                key: [min(r["errors"][key] for r in runs),
                      max(r["errors"][key] for r in runs)] for key in keys
            },
            "iid_q95_exceedances": {
                key: sum(abs(r["errors"][key]) >
                         ref["iid_baseline"][str(n)]["per_run"][key]["q95"] for r in runs)
                for key in keys
            },
            "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in runs])),
        }
    high = comparison["2048"]
    flags = {
        "mode_error_tail": high["iid_q95_exceedances"]["maximum_mode_absolute_error"]
        >= TAIL_MIN_EXCEEDANCES,
        "joint_tail": high["iid_q95_exceedances"]["full_joint_tv"]
        >= TAIL_MIN_EXCEEDANCES,
        "cell_shift": any(cell["same_sign_halves"]
                          for cell in high["familywise_flagged_cells"]),
    }
    result["comparison"] = comparison
    result["diagnostic_flags"] = flags
    result["status"] = ("eight_frame_absolute_diagnostic_signal"
                        if any(flags.values()) else "eight_frame_absolute_audit_no_flag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11q_seven_detection_exact_reference.json"))
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
