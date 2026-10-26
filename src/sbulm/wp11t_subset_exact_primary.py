"""Independent subset-convolution absolute reference for the exposed WP11k primary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .existence_smc import sample_slot_existence_posterior
from .slot_existence import ABSENT, SlotExistenceMaternModel
from .wp7_convergence_audit import _case_metadata, _large_run_summary, _save
from .wp8_generalization import _metadata as _regime_metadata
from .wp8_generalization import make_case
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls
from .wp11k_variance_calibration import (
    BANK_CARDINALITIES,
    BANK_HASHES,
    REGIMES,
    _JointObserver,
    _set_cpu_limit,
)
from .wp11q_seven_detection_exact_reference import _gp_from_table, _gp_table
from .wp11s_middle_frame_density_audit import _case as _ten_case

PRIMARY = next(regime for regime in REGIMES if regime.name == "wp11k_primary")
PRIMARY_HASH = "e7fc37d377e5b956377af6a64e41837f15e42d43e61a932af48400da9bffd922"
PRIMARY_CARDINALITIES = (2, 2, 4, 4, 3, 2, 1, 1)
TEN_FILE_SHA256 = "b041c2ab5466a33edb570a925979f516fba50ab7692a5bcb34a5d5f7b4e83dbc"
ARCHIVE_FILE_SHA256 = "fda3728fc5a57c7cd31ae84e13380344e81a796bde2c054926ef1fa1ad9ccecc"
TEN_CASES = ("early", "middle")
TEN_RATE = 1.5
SEEDS = (1200, 1201, 1202, 1203, 1204, 1205)
N_PARTICLES = 2_048
IID_SEED = 34_531
IID_REPLICATES = 1_000
REFERENCE_CPU_CAP = 1_200
RUN_CPU_CAP = 750
VALIDATION_LOGZ_TOL = 1e-9
VALIDATION_TV_TOL = 1e-9
VALIDATION_CELL_TOL = 1e-10
PRIMARY_LOGZ_TOL = 1e-8
PRIMARY_TV_TOL = 1e-7
PRIMARY_CELL_TOL = 1e-8
PRIMARY_NEGATIVE_TOL = 1e-10
PRIMARY_IMAG_TOL = 1e-8
PRIMARY_MASS_TOL = 1e-8
TAIL_MIN_EXCEEDANCES = 2


def _case():
    case = make_case(PRIMARY)
    info = _case_metadata(case)
    if (info["detections_sha256"] != PRIMARY_HASH
            or tuple(info["detection_cardinalities"]) != PRIMARY_CARDINALITIES
            or PRIMARY_HASH != BANK_HASHES[PRIMARY.name]
            or PRIMARY_CARDINALITIES != BANK_CARDINALITIES[PRIMARY.name]):
        raise ValueError("exposed WP11k primary observation changed")
    return case


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "absolute four-slot/eight-frame WP11k 19-detection primary reference",
        "primary_hash": PRIMARY_HASH,
        "cardinalities": list(PRIMARY_CARDINALITIES),
        "ten_result_sha256": TEN_FILE_SHA256,
        "archive_sha256": ARCHIVE_FILE_SHA256,
        "reference_method": (
            "independent slot-mask/detection polynomial; squarefree subset fourth power "
            "by ranked zeta/Mobius transforms; 33-point Fourier alive coefficient extraction; "
            "clutter counts read from assigned-detection union"
        ),
        "numerical_validation": {
            "ten_cases": list(TEN_CASES), "ten_rate": TEN_RATE,
            "ten_logz_abs": VALIDATION_LOGZ_TOL,
            "ten_joint_tv": VALIDATION_TV_TOL,
            "ten_max_cell_abs": VALIDATION_CELL_TOL,
            "primary_direct_vs_paired_logz_abs": PRIMARY_LOGZ_TOL,
            "primary_direct_vs_paired_joint_tv": PRIMARY_TV_TOL,
            "primary_direct_vs_paired_max_cell_abs": PRIMARY_CELL_TOL,
            "primary_min_raw_probability": -PRIMARY_NEGATIVE_TOL,
            "primary_max_relative_imag": PRIMARY_IMAG_TOL,
            "primary_max_mass_error": PRIMARY_MASS_TOL,
            "longdouble_and_complex_longdouble": True,
            "stop_sampler_if_any_fail": True,
        },
        "joint": "33 alive totals by 3 frame-six clutter counts by 2 final-frame clutter counts",
        "archived_comparison": (
            "retrospective four WP11k seeds 800..803 at 512/2048: alive histogram, "
            "low<=20/mid=21/high>=22 by frame-six clutter, evidence"
        ),
        "fresh_comparison": (
            "six unchanged WP11g seeds at 2048: full 3D joint TV, band/clutter TV, "
            "max band error, alive TV, both clutter-mean errors, max 3D cell and evidence"
        ),
        "iid": (
            "1000 exact multinomial draws at 512 and 2048 per-run descriptive q95; "
            "1000 groups of six at 2048 for max mean signed 3D cell familywise q95"
        ),
        "diagnostic_flags": (
            "at 2048, mode and full-joint tails require >=2/6 respective IID q95 "
            "exceedances; cell shift requires familywise mean signed 3D cell q95 "
            "and same sign in both ordered three-seed halves; no move gate"
        ),
        "reference_before_new_sampler": True,
        "seeds": list(SEEDS), "particles": N_PARTICLES,
        "iid_seed": IID_SEED, "iid_replicates": IID_REPLICATES,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "reference_cpu_cap": REFERENCE_CPU_CAP, "run_cpu_cap": RUN_CPU_CAP,
        "interpretation": (
            "exposed synthetic primary, not the untouched holdout; exact algebra in real "
            "arithmetic with empirical numerical cross-checks, not a formal interval proof; "
            "IID q95 descriptive for correlated SMC; no move or 8192"
        ),
    }


def _metadata() -> dict[str, Any]:
    return json.loads(json.dumps(_regime_metadata(PRIMARY, _case())))


def _checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"frozen source checkpoint changed: {path}")
    return json.loads(raw)


def _load(output: Path) -> dict[str, Any]:
    protocol, case = _protocol(), _metadata()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["case"] != case:
            raise ValueError("checkpoint belongs to a different frozen reference")
        return result
    result = {
        "experiment": "wp11t_subset_exact_primary", "protocol": protocol,
        "case": case, "validations": None, "reference": None,
        "archive_comparison": None, "runs": {}, "comparison": None,
        "status": "in_progress",
    }
    _save(output, result)
    return result


def _single_slot_coefficients(
    model: SlotExistenceMaternModel,
) -> tuple[np.ndarray, float, int]:
    """Coefficient for each assigned-detection subset and alive interval length."""
    cards = model.sequence.cardinalities
    n_detections = sum(cards)
    digits = np.asarray(list(product(*(range(n + 1) for n in cards))), dtype=np.int8)
    assigned = digits > 0
    gp_table, gp_strides = _gp_table(model)
    gp = gp_table[digits @ gp_strides]
    offsets = np.cumsum((0,) + cards[:-1])
    bits = np.zeros(len(digits), dtype=np.int32)
    for t, offset in enumerate(offsets):
        bits |= np.where(assigned[:, t],
                         1 << (offset + np.maximum(digits[:, t] - 1, 0)), 0)
    scores, indices, lengths = [], [], []
    for mask, prior in zip(model.interval_masks(),
                           model.interval_log_probabilities(), strict=True):
        valid = np.all(~assigned | mask, axis=1)
        n_assigned = assigned[valid].sum(axis=1)
        scores.append(
            prior + n_assigned * np.log(model.detection_probability / model.clutter_intensity)
            + (mask.sum() - n_assigned) * np.log(1 - model.detection_probability)
            + gp[valid]
        )
        indices.append(bits[valid])
        lengths.extend([int(mask.sum())] * int(valid.sum()))
    log_scores = np.concatenate(scores)
    subset = np.concatenate(indices)
    log_z1 = float(logsumexp(log_scores))
    coefficients = np.zeros((model.sequence.n_frames + 1, 1 << n_detections),
                            dtype=np.longdouble)
    np.add.at(coefficients, (lengths, subset),
              np.exp(log_scores.astype(np.longdouble) - np.longdouble(log_z1)))
    return coefficients, log_z1, len(log_scores)


def _ranked_zeta(f: np.ndarray, n: int, counts: np.ndarray) -> np.ndarray:
    out = np.zeros((n + 1, 1 << n), dtype=f.dtype)
    out[counts, np.arange(1 << n)] = f
    for bit in range(n):
        view = out.reshape(n + 1, -1, 2, 1 << bit)
        view[:, :, 1, :] += view[:, :, 0, :]
    return out


def _mobius(f: np.ndarray, n: int) -> None:
    for bit in range(n):
        view = f.reshape(n + 1, -1, 2, 1 << bit)
        view[:, :, 1, :] -= view[:, :, 0, :]


def _square(f: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(f)
    for rank in range(n + 1):
        for first in range(rank + 1):
            out[rank] += f[first] * f[rank - first]
    return out


def _fourth_power(f: np.ndarray, n: int, counts: np.ndarray,
                  method: str) -> np.ndarray:
    if method == "direct":
        transformed = _ranked_zeta(f, n, counts)
        fourth = _square(_square(transformed, n), n)
        _mobius(fourth, n)
        return fourth[counts, np.arange(1 << n)]
    if method != "paired":
        raise ValueError("unknown independent polynomial parenthesization")
    transformed = _ranked_zeta(f, n, counts)
    pair = _square(transformed, n)
    _mobius(pair, n)
    pair = pair[counts, np.arange(1 << n)]
    transformed_pair = _ranked_zeta(pair, n, counts)
    fourth = _square(transformed_pair, n)
    _mobius(fourth, n)
    return fourth[counts, np.arange(1 << n)]


def _grouping(cards: tuple[int, ...], middle_frame: int | None):
    n = sum(cards)
    subsets = np.arange(1 << n, dtype=np.int32)
    offsets = np.cumsum((0,) + cards[:-1])
    frames = (5, 7) if middle_frame is None else (middle_frame, 5, 7)
    counts = []
    for frame in frames:
        digits = (subsets >> offsets[frame]) & ((1 << cards[frame]) - 1)
        assigned = np.fromiter((int(v).bit_count() for v in digits),
                               dtype=np.int8, count=len(digits))
        counts.append(cards[frame] - assigned)
    shape = tuple(cards[frame] + 1 for frame in frames)
    group = np.ravel_multi_index(counts, shape)
    order = np.argsort(group, kind="stable")
    groups, starts = np.unique(group[order], return_index=True)
    if not np.array_equal(groups, np.arange(np.prod(shape))):
        raise ValueError("clutter group support is incomplete")
    return order, starts, shape


def _exact_joint(model: SlotExistenceMaternModel, middle_frame: int | None,
                 method: str) -> dict[str, Any]:
    """Four-slot exact joint using squarefree subset convolution and Fourier inversion."""
    started = time.process_time()
    coefficients, log_z1, n_slot_states = _single_slot_coefficients(model)
    n = sum(model.sequence.cardinalities)
    n_alive = model.n_slots * model.sequence.n_frames + 1
    counts = np.fromiter((i.bit_count() for i in range(1 << n)),
                         dtype=np.int8, count=1 << n)
    nonzero = np.flatnonzero(np.any(coefficients, axis=0))
    sparse = coefficients[:, nonzero].astype(np.clongdouble)
    order, starts, shape = _grouping(model.sequence.cardinalities, middle_frame)
    moments = np.zeros((n_alive, int(np.prod(shape))), dtype=np.clongdouble)
    two_pi = 2 * np.arccos(np.longdouble(-1))
    lengths = np.arange(model.sequence.n_frames + 1, dtype=np.longdouble)
    for frequency in range(n_alive // 2 + 1):
        phase = np.exp(1j * two_pi * frequency * lengths / n_alive).astype(np.clongdouble)
        f = np.zeros(1 << n, dtype=np.clongdouble)
        f[nonzero] = phase @ sparse
        union = _fourth_power(f, n, counts, method)
        moments[frequency] = np.add.reduceat(union[order], starts)
        if frequency:
            moments[n_alive - frequency] = np.conj(moments[frequency])
    angles = np.arange(n_alive, dtype=np.longdouble)
    inverse = np.exp(-1j * two_pi * angles[:, None] * angles[None, :] / n_alive)
    raw = (inverse.astype(np.clongdouble) @ moments / n_alive).reshape((n_alive,) + shape)
    mass = moments[0].sum().real
    unnormalized = raw / mass
    real = unnormalized.real
    raw_mass = float(real.sum())
    minimum = float(real.min())
    imaginary = float(np.max(np.abs(unnormalized.imag)))
    if mass <= 0:
        raise ValueError("non-positive exact partition")
    clipped = np.maximum(real, 0)
    normalized = clipped / clipped.sum()
    log_evidence = (n * np.log(model.clutter_intensity)
                    - model.sequence.n_frames * model.expected_clutter_per_frame
                    + 4 * log_z1 + np.log(mass))
    return {
        "joint": np.asarray(normalized, dtype=float).tolist(),
        "log_evidence": float(log_evidence),
        "raw_min_cell": minimum, "raw_mass_error": abs(raw_mass - 1),
        "relative_imaginary_residual": imaginary,
        "clipping_tv": float(np.abs(normalized - real).sum() / 2),
        "n_slot_states": n_slot_states, "n_valid_single_subsets": len(nonzero),
        "acceptance_normalizer": float(mass),
        "cpu_seconds": time.process_time() - started,
    }


def _bands(joint: np.ndarray) -> np.ndarray:
    collapsed = joint.sum(axis=2)
    return np.stack((collapsed[:21].sum(axis=0), collapsed[21],
                     collapsed[22:].sum(axis=0)))


def _summary(joint: np.ndarray) -> dict[str, Any]:
    return {
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "band_by_sixth_clutter": _bands(joint).tolist(),
        "band_mass": _bands(joint).sum(axis=1).tolist(),
        "alive": joint.sum(axis=(1, 2)).tolist(),
        "sixth_clutter": joint.sum(axis=(0, 2)).tolist(),
        "last_clutter": joint.sum(axis=(0, 1)).tolist(),
    }


def _errors(joint: np.ndarray, exact: dict[str, Any]) -> dict[str, float]:
    truth = np.asarray(exact["joint_alive_by_sixth_by_last_clutter"])
    delta = joint - truth
    bands = _bands(joint) - np.asarray(exact["band_by_sixth_clutter"])
    return {
        "full_3d_joint_tv": float(np.abs(delta).sum() / 2),
        "band_by_sixth_joint_tv": float(np.abs(bands).sum() / 2),
        "maximum_band_mass_error": float(np.max(np.abs(bands.sum(axis=1)))),
        "alive_total_tv": float(np.abs(delta.sum(axis=(1, 2))).sum() / 2),
        "maximum_3d_cell_error": float(np.max(np.abs(delta))),
        "sixth_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[1]), delta.sum(axis=(0, 2))
        )),
        "last_clutter_mean_error": float(np.dot(
            np.arange(delta.shape[2]), delta.sum(axis=(0, 1))
        )),
    }


def _iid_baseline(exact: dict[str, Any]) -> dict[str, Any]:
    p = np.asarray(exact["joint_alive_by_sixth_by_last_clutter"]).reshape(-1)
    rng = np.random.default_rng(IID_SEED)
    keys = ("full_3d_joint_tv", "band_by_sixth_joint_tv", "maximum_band_mass_error",
            "alive_total_tv", "maximum_3d_cell_error", "sixth_clutter_mean_error",
            "last_clutter_mean_error")
    baseline = {}
    for n in (512, N_PARTICLES):
        draws = {key: [] for key in keys}
        for counts in rng.multinomial(n, p, size=IID_REPLICATES):
            errors = _errors(counts.reshape(33, 3, 2) / n, exact)
            for key in keys:
                draws[key].append(abs(errors[key]))
        baseline[str(n)] = {
            "per_run": {key: {"median": float(np.median(vals)),
                              "q95": float(np.quantile(vals, .95))}
                        for key, vals in draws.items()},
        }
    grouped = rng.multinomial(len(SEEDS) * N_PARTICLES, p, size=IID_REPLICATES)
    maximum = np.max(np.abs(grouped / (len(SEEDS) * N_PARTICLES) - p), axis=1)
    baseline[str(N_PARTICLES)]["group_max_mean_signed_3d_cell"] = {
        "median": float(np.median(maximum)),
        "q95": float(np.quantile(maximum, .95)),
    }
    return baseline


def _archive_comparison(exact: dict[str, Any], baseline: dict[str, Any],
                        archive: dict[str, Any]) -> dict[str, Any]:
    if archive["status"] != "audit_completed_no_candidate_gate":
        raise ValueError("expected completed frozen WP11k archive")
    out = {}
    for n in (512, N_PARTICLES):
        runs = archive["runs"]["wp11k_primary"][str(n)]
        if set(runs) != {str(seed) for seed in (800, 801, 802, 803)}:
            raise ValueError("archived WP11k seeds changed")
        rows = {}
        for seed, run in runs.items():
            observed = np.asarray(run["joint_trace"][-1]["post_joint"])
            band_delta = observed - exact["band_by_sixth_clutter"]
            alive_delta = np.asarray(run["summary"]["alive_count_histogram"]) - exact["alive"]
            rows[seed] = {
                "band_joint_tv": float(np.abs(band_delta).sum() / 2),
                "maximum_band_mass_error": float(np.max(np.abs(band_delta.sum(axis=1)))),
                "alive_total_tv": float(np.abs(alive_delta).sum() / 2),
                "sixth_clutter_mean_error": float(np.dot(
                    np.arange(observed.shape[1]), band_delta.sum(axis=0)
                )),
                "log_evidence_error": run["summary"]["log_evidence_estimate"]
                                      - exact["log_evidence"],
            }
        out[str(n)] = {
            "per_seed": rows,
            "iid_q95_exceedances": {
                "maximum_band_mass_error": sum(
                    row["maximum_band_mass_error"] > baseline[str(n)]["per_run"]
                    ["maximum_band_mass_error"]["q95"] for row in rows.values()
                ),
                "alive_total_tv": sum(
                    row["alive_total_tv"] > baseline[str(n)]["per_run"]
                    ["alive_total_tv"]["q95"] for row in rows.values()
                ),
            },
        }
    return out


def run_reference(output: Path, *, ten_path: Path, archive_path: Path) -> dict[str, Any]:
    result = _load(output)
    if result["reference"] is not None:
        print("WP11t absolute reference: reused", flush=True)
        return result
    _set_cpu_limit(REFERENCE_CPU_CAP)
    ten = _checked_json(ten_path, TEN_FILE_SHA256)
    archive = _checked_json(archive_path, ARCHIVE_FILE_SHA256)
    validations = {}
    for name in TEN_CASES:
        prior = ten["references"][name]
        if not prior["difficulty_valid"] or prior["selected_rate"] != TEN_RATE:
            raise ValueError("ten-detection validation reference changed")
        model = _ten_case(name, TEN_RATE).model
        replay = _exact_joint(model, 2 if name == "early" else 4, "direct")
        actual = np.asarray(replay["joint"])
        expected = np.asarray(prior["exact"]["joint_alive_by_middle_by_sixth_by_last_clutter"])
        delta = actual - expected
        errors = {
            "log_evidence_absolute_error": abs(replay["log_evidence"] - prior["log_evidence"]),
            "joint_tv": float(np.abs(delta).sum() / 2),
            "maximum_cell_absolute_error": float(np.max(np.abs(delta))),
            "cpu_seconds": replay["cpu_seconds"],
        }
        validations[name] = errors
        if (errors["log_evidence_absolute_error"] > VALIDATION_LOGZ_TOL
                or errors["joint_tv"] > VALIDATION_TV_TOL
                or errors["maximum_cell_absolute_error"] > VALIDATION_CELL_TOL):
            result["validations"] = validations
            result["status"] = "ten_case_validation_failed_stop"
            _save(output, result)
            return result
    model = _case().model
    table, strides = _gp_table(model)
    check = model.sample_base_associations(n_samples=32, rng=np.random.default_rng(34111))
    np.testing.assert_allclose(_gp_from_table(check, table, strides),
                               model.log_marginal_density(check), atol=1e-10, rtol=0)
    direct = _exact_joint(model, None, "direct")
    paired = _exact_joint(model, None, "paired")
    a, b = np.asarray(direct["joint"]), np.asarray(paired["joint"])
    numerical = {
        "direct_vs_paired_log_evidence_absolute_error": abs(
            direct["log_evidence"] - paired["log_evidence"]
        ),
        "direct_vs_paired_joint_tv": float(np.abs(a - b).sum() / 2),
        "direct_vs_paired_max_cell_absolute_error": float(np.max(np.abs(a - b))),
        "direct_raw_min_cell": direct["raw_min_cell"],
        "paired_raw_min_cell": paired["raw_min_cell"],
        "maximum_relative_imaginary_residual": max(
            direct["relative_imaginary_residual"], paired["relative_imaginary_residual"]
        ),
        "maximum_raw_mass_error": max(direct["raw_mass_error"], paired["raw_mass_error"]),
        "maximum_clipping_tv": max(direct["clipping_tv"], paired["clipping_tv"]),
        "direct_cpu_seconds": direct["cpu_seconds"],
        "paired_cpu_seconds": paired["cpu_seconds"],
    }
    valid = (
        numerical["direct_vs_paired_log_evidence_absolute_error"] <= PRIMARY_LOGZ_TOL
        and numerical["direct_vs_paired_joint_tv"] <= PRIMARY_TV_TOL
        and numerical["direct_vs_paired_max_cell_absolute_error"] <= PRIMARY_CELL_TOL
        and min(direct["raw_min_cell"], paired["raw_min_cell"]) >= -PRIMARY_NEGATIVE_TOL
        and numerical["maximum_relative_imaginary_residual"] <= PRIMARY_IMAG_TOL
        and numerical["maximum_raw_mass_error"] <= PRIMARY_MASS_TOL
    )
    result["validations"] = validations
    result["reference"] = {
        **_summary(a), "log_evidence": direct["log_evidence"],
        "n_slot_states": direct["n_slot_states"],
        "n_valid_single_subsets": direct["n_valid_single_subsets"],
        "numerical": numerical, "valid": bool(valid),
    }
    if valid:
        result["reference"]["iid_baseline"] = _iid_baseline(result["reference"])
        result["archive_comparison"] = _archive_comparison(
            result["reference"], result["reference"]["iid_baseline"], archive
        )
        result["status"] = "validated_reference_before_sampler"
    else:
        result["status"] = "primary_numerical_validation_failed_stop"
    _save(output, result)
    print(f"WP11t primary exact reference: valid={valid}, logZ={direct['log_evidence']:.8f}, "
          f"numerical joint TV={numerical['direct_vs_paired_joint_tv']:.3g}", flush=True)
    return result


def run_one(output: Path, *, seed: int) -> dict[str, Any]:
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to one")
    if seed not in SEEDS:
        raise ValueError("seed outside frozen schedule")
    result = _load(output)
    if result["reference"] is None or not result["reference"]["valid"]:
        raise ValueError("validated primary absolute reference required before sampling")
    if str(seed) in result["runs"]:
        print(f"primary {N_PARTICLES} seed {seed}: reused", flush=True)
        return result
    if seed != SEEDS[len(result["runs"])]:
        raise ValueError("run frozen seed order")
    _set_cpu_limit(RUN_CPU_CAP)
    case = _case()
    observer = _JointObserver(case)
    controls = replace(_candidate_controls(seed), n_particles=N_PARTICLES)
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
    states = sampled.associations
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    sixth = PRIMARY_CARDINALITIES[5] - np.count_nonzero(states[:, 5] >= 0, axis=1)
    last = PRIMARY_CARDINALITIES[7] - np.count_nonzero(states[:, 7] >= 0, axis=1)
    joint = np.bincount(6 * alive + 2 * sixth + last,
                        minlength=33 * 3 * 2).reshape(33, 3, 2) / len(states)
    errors = _errors(joint, result["reference"])
    errors["log_evidence_error"] = sampled.log_evidence_estimate - result["reference"][
        "log_evidence"
    ]
    result["runs"][str(seed)] = {
        "cpu_seconds": cpu, "wall_seconds": wall,
        "summary": _large_run_summary(case, sampled, wall, seed),
        "joint_alive_by_sixth_by_last_clutter": joint.tolist(),
        "errors": errors, "joint_trace": observer.records,
    }
    if len(result["runs"]) == len(SEEDS):
        _finish(result)
    _save(output, result)
    print(f"primary {N_PARTICLES} seed {seed}: {cpu:.1f}s CPU, "
          f"joint TV {errors['full_3d_joint_tv']:.4f}, "
          f"band error {errors['maximum_band_mass_error']:.4f}; "
          f"status={result['status']}", flush=True)
    return result


def _finish(result: dict[str, Any]) -> None:
    runs = [result["runs"][str(seed)] for seed in SEEDS]
    ref = result["reference"]
    truth = np.asarray(ref["joint_alive_by_sixth_by_last_clutter"])
    observed = np.asarray([run["joint_alive_by_sixth_by_last_clutter"] for run in runs])
    delta = observed.mean(axis=0) - truth
    q95 = ref["iid_baseline"][str(N_PARTICLES)]["group_max_mean_signed_3d_cell"]["q95"]
    flagged = []
    for alive, clutter, last in np.argwhere(np.abs(delta) > q95):
        first = observed[:3, alive, clutter, last].mean() - truth[alive, clutter, last]
        second = observed[3:, alive, clutter, last].mean() - truth[alive, clutter, last]
        flagged.append({
            "alive": int(alive), "sixth_clutter": int(clutter), "last_clutter": int(last),
            "signed_error": float(delta[alive, clutter, last]),
            "same_sign_halves": bool(first * second > 0),
        })
    metrics = ("full_3d_joint_tv", "band_by_sixth_joint_tv", "maximum_band_mass_error",
               "alive_total_tv", "maximum_3d_cell_error", "sixth_clutter_mean_error",
               "last_clutter_mean_error")
    comparisons = {
        "mean_signed_3d_cell_error": delta.tolist(),
        "maximum_absolute_mean_3d_cell_error": float(np.max(np.abs(delta))),
        "familywise_flagged_cells": flagged,
        "median_absolute_errors": {
            key: float(median(abs(run["errors"][key]) for run in runs)) for key in metrics
        },
        "error_ranges": {
            key: [min(run["errors"][key] for run in runs),
                  max(run["errors"][key] for run in runs)] for key in metrics
        },
        "iid_q95_exceedances": {
            key: sum(abs(run["errors"][key]) >
                     ref["iid_baseline"][str(N_PARTICLES)]["per_run"][key]["q95"]
                     for run in runs) for key in metrics
        },
        "mean_cpu_seconds": float(np.mean([run["cpu_seconds"] for run in runs])),
        "log_evidence_error_range": [min(run["errors"]["log_evidence_error"] for run in runs),
                                     max(run["errors"]["log_evidence_error"] for run in runs)],
    }
    flags = {
        "mode_tail": comparisons["iid_q95_exceedances"]["maximum_band_mass_error"]
                     >= TAIL_MIN_EXCEEDANCES,
        "joint_tail": comparisons["iid_q95_exceedances"]["full_3d_joint_tv"]
                      >= TAIL_MIN_EXCEEDANCES,
        "cell_shift": any(cell["same_sign_halves"] for cell in flagged),
    }
    result["comparison"] = comparisons
    result["diagnostic_flags"] = flags
    result["status"] = ("absolute_primary_diagnostic_signal" if any(flags.values())
                        else "absolute_primary_no_persistent_flag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "sampler"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11t_subset_exact_primary.json"))
    parser.add_argument("--ten-path", type=Path,
                        default=Path("results/wp11s_middle_frame_density_audit.json"))
    parser.add_argument("--archive-path", type=Path,
                        default=Path("results/wp11k_variance_calibration.json"))
    args = parser.parse_args()
    if args.case == "reference":
        if args.seed is not None:
            parser.error("reference takes no sampler seed")
        run_reference(args.output, ten_path=args.ten_path, archive_path=args.archive_path)
    else:
        if args.seed is None:
            parser.error("sampler requires a frozen seed")
        run_one(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
