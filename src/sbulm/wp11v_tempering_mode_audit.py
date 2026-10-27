"""Frozen independent four-slot tempering-stage count/clutter audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import replace
from itertools import product
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
)
from .wp7_convergence_audit import _save
from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls

SEED_PAIRS = ((101, 103), (107, 109), (113, 127), (131, 137),
              (139, 149), (151, 157), (163, 167), (173, 179))
SEEDS = (1400, 1401, 1402, 1403)
N_PARTICLES = 2_048
RUN_CPU_CAP = 750
IID_SEED = 34_621
IID_REPLICATES = 1_000
CASE_HASH = "400975d533de5fa9ba5833f0ade92e20045dd7397701637cd8db4534c531e1c4"
CASE_CARDINALITIES = (2, 2, 2, 3, 1, 1, 0, 2)
REFERENCE_SHA256 = "4577d3c6214c4493b5b59b0b8302936f5af04bc37e0ed1c16c0cb8f1b4ce06d6"
NUMERIC_LOGZ_TOL = 1e-8
NUMERIC_TV_TOL = 1e-7
NUMERIC_CELL_TOL = 1e-8
MIN_RAW_CELL = -1e-10
MAX_IMAG_RESIDUAL = 1e-8
MAX_RAW_MASS_ERROR = 1e-8
EXACT_TOY_LOGZ_TOL = 1e-9
EXACT_TOY_TV_TOL = 1e-9
EXACT_TOY_CELL_TOL = 1e-10


def _select_case() -> tuple[int, GeneralizationRegime, Any]:
    """First prespecified geometry passing cardinality-only constraints."""

    for index, (trajectory_seed, corruption_seed) in enumerate(SEED_PAIRS):
        regime = GeneralizationRegime(
            name="wp11v_independent", detection_probability=0.50,
            expected_clutter_per_frame=0.35, central_separation=0.006,
            trajectory_seed=trajectory_seed, corruption_seed=corruption_seed,
            max_clutter_per_frame=2, truth_mask=STAGGERED_TRUTH_MASK,
        )
        case = make_case(regime)
        cards = tuple(case.model.sequence.cardinalities)
        if (11 <= sum(cards) <= 14 and cards[5] >= 1 and cards[7] >= 1
                and sum(cards[2:5]) >= 4):
            return index, regime, case
    raise RuntimeError("prespecified independent density ladder exhausted")


def _case() -> tuple[Any, dict[str, Any]]:
    index, regime, case = _select_case()
    metadata = json.loads(json.dumps(_metadata(regime, case)))
    info = metadata["model_and_detections"]
    if info["detections_sha256"] != CASE_HASH:
        raise ValueError("selected independent observation hash changed")
    if tuple(info["detection_cardinalities"]) != CASE_CARDINALITIES:
        raise ValueError("selected independent cardinalities changed")
    return case, {"ladder_index": index, **metadata}


def _protocol() -> dict[str, Any]:
    return {
        "seed_pairs": [list(pair) for pair in SEED_PAIRS],
        "selection": "first 11..14 points, frame-six and last >=1, middle frames 3..5 >=4",
        "regime": {"detection_probability": 0.50, "expected_clutter_per_frame": 0.35,
                   "central_separation": 0.006, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "selected_hash": CASE_HASH,
        "selected_cardinalities": list(CASE_CARDINALITIES),
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "per_run_cpu_cap": RUN_CPU_CAP,
        "iid_seed": IID_SEED, "iid_replicates": IID_REPLICATES,
        "onset": "two consecutive rejuvenated band-by-sixth TVs > max(.05,2*iid_q95)",
        "reference": "exact beta-scaled four-slot squarefree subset convolution",
        "reference_sha256": REFERENCE_SHA256,
        "numerical_envelope": {
            "toy_logz": EXACT_TOY_LOGZ_TOL, "toy_tv": EXACT_TOY_TV_TOL,
            "toy_max_cell": EXACT_TOY_CELL_TOL,
            "direct_pair_logz": NUMERIC_LOGZ_TOL, "direct_pair_tv": NUMERIC_TV_TOL,
            "direct_pair_max_cell": NUMERIC_CELL_TOL, "minimum_raw_cell": MIN_RAW_CELL,
            "maximum_imaginary_residual": MAX_IMAG_RESIDUAL,
            "maximum_raw_mass_error": MAX_RAW_MASS_ERROR,
        },
        "no_move_or_holdout": True,
    }


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


def _fourth_power(f: np.ndarray, n: int, counts: np.ndarray, method: str) -> np.ndarray:
    transformed = _ranked_zeta(f, n, counts)
    pair = _square(transformed, n)
    if method == "direct":
        fourth = _square(pair, n)
        _mobius(fourth, n)
        return fourth[counts, np.arange(1 << n)]
    if method != "paired":
        raise ValueError("unknown subset calculation")
    _mobius(pair, n)
    pair = pair[counts, np.arange(1 << n)]
    transformed_pair = _ranked_zeta(pair, n, counts)
    fourth = _square(transformed_pair, n)
    _mobius(fourth, n)
    return fourth[counts, np.arange(1 << n)]


class _ExactComputer:
    """Cache one-slot patterns, then evaluate the exact tempered joint at beta."""

    def __init__(self, model: SlotExistenceMaternModel, frames: tuple[int, int]):
        if model.n_slots != 4:
            raise ValueError("the squarefree calculation requires four slots")
        self.model = model
        self.cards = tuple(model.sequence.cardinalities)
        self.n = sum(self.cards)
        self.frames = frames
        self.n_alive = model.n_slots * model.sequence.n_frames + 1
        digits = np.asarray(list(product(*(range(card + 1) for card in self.cards))),
                            dtype=np.int8)
        assigned = digits > 0
        # Each GP factor is evaluated once for each possible single-slot
        # detection sequence, with inactive slots absent throughout.
        gp_states = np.full((len(digits), model.sequence.n_frames, 4), ABSENT)
        gp_states[:, :, 0] = np.where(assigned, digits - 1, -1)
        gp = model.log_marginal_density(gp_states)
        offsets = np.cumsum((0,) + self.cards[:-1])
        bits = np.zeros(len(digits), dtype=np.int32)
        for t, offset in enumerate(offsets):
            bits |= np.where(assigned[:, t],
                             1 << (offset + np.maximum(digits[:, t] - 1, 0)), 0)
        base, gps, subsets, lengths = [], [], [], []
        for mask, prior in zip(model.interval_masks(),
                               model.interval_log_probabilities(), strict=True):
            valid = np.all(~assigned | mask, axis=1)
            n_assigned = assigned[valid].sum(axis=1)
            base.append(
                prior + n_assigned * np.log(
                    model.detection_probability / model.clutter_intensity
                ) + (mask.sum() - n_assigned) * np.log1p(-model.detection_probability)
            )
            gps.append(gp[valid])
            subsets.append(bits[valid])
            lengths.extend([int(mask.sum())] * int(valid.sum()))
        self.base = np.concatenate(base)
        self.gp = np.concatenate(gps)
        self.subsets = np.concatenate(subsets)
        self.lengths = np.asarray(lengths, dtype=np.int8)
        subsets_all = np.arange(1 << self.n, dtype=np.int32)
        self.counts = np.fromiter((int(v).bit_count() for v in subsets_all),
                                  dtype=np.int8, count=1 << self.n)
        clutter = []
        for frame in frames:
            digit = (subsets_all >> offsets[frame]) & ((1 << self.cards[frame]) - 1)
            n_assigned = np.fromiter((int(v).bit_count() for v in digit),
                                      dtype=np.int8, count=1 << self.n)
            clutter.append(self.cards[frame] - n_assigned)
        self.shape = tuple(self.cards[frame] + 1 for frame in frames)
        group = np.ravel_multi_index(clutter, self.shape)
        self.order = np.argsort(group, kind="stable")
        labels, self.starts = np.unique(group[self.order], return_index=True)
        if not np.array_equal(labels, np.arange(np.prod(self.shape))):
            raise ValueError("incomplete clutter grouping")

    def compute(self, beta: float, method: str = "direct") -> dict[str, Any]:
        if not 0 <= beta <= 1:
            raise ValueError("beta outside the unit interval")
        started = time.process_time()
        scores = self.base + beta * self.gp
        log_z1 = float(logsumexp(scores))
        coefficients = np.zeros((self.model.sequence.n_frames + 1, 1 << self.n),
                                dtype=np.longdouble)
        np.add.at(coefficients, (self.lengths, self.subsets),
                  np.exp(scores.astype(np.longdouble) - np.longdouble(log_z1)))
        nonzero = np.flatnonzero(np.any(coefficients, axis=0))
        sparse = coefficients[:, nonzero].astype(np.clongdouble)
        moments = np.zeros((self.n_alive, int(np.prod(self.shape))), dtype=np.clongdouble)
        two_pi = 2 * np.arccos(np.longdouble(-1))
        lengths = np.arange(self.model.sequence.n_frames + 1, dtype=np.longdouble)
        for frequency in range(self.n_alive // 2 + 1):
            phase = np.exp(1j * two_pi * frequency * lengths / self.n_alive).astype(
                np.clongdouble
            )
            f = np.zeros(1 << self.n, dtype=np.clongdouble)
            f[nonzero] = phase @ sparse
            union = _fourth_power(f, self.n, self.counts, method)
            moments[frequency] = np.add.reduceat(union[self.order], self.starts)
            if frequency:
                moments[self.n_alive - frequency] = np.conj(moments[frequency])
        angles = np.arange(self.n_alive, dtype=np.longdouble)
        inverse = np.exp(-1j * two_pi * angles[:, None] * angles[None, :]
                         / self.n_alive)
        raw = (inverse.astype(np.clongdouble) @ moments / self.n_alive)
        mass = moments[0].sum().real
        if mass <= 0:
            raise ValueError("non-positive partition function")
        normalized = raw / mass
        real = normalized.real
        clipped = np.maximum(real, 0)
        joint = (clipped / clipped.sum()).reshape((self.n_alive,) + self.shape)
        log_z = (self.n * np.log(self.model.clutter_intensity)
                 - self.model.sequence.n_frames * self.model.expected_clutter_per_frame
                 + 4 * log_z1 + np.log(mass))
        return {
            "beta": float(beta), "joint": np.asarray(joint, dtype=float).tolist(),
            "log_partition": float(log_z),
            "minimum_raw_cell": float(real.min()),
            "maximum_relative_imaginary_residual": float(np.max(np.abs(normalized.imag))),
            "raw_mass_error": abs(float(real.sum()) - 1),
            "clipping_tv": float(np.abs(clipped / clipped.sum() - real).sum() / 2),
            "cpu_seconds": time.process_time() - started,
            "n_one_slot_states": len(scores),
            "n_valid_single_subsets": len(nonzero),
        }


def _numeric_ok(result: dict[str, Any]) -> bool:
    return (result["minimum_raw_cell"] >= MIN_RAW_CELL
            and result["maximum_relative_imaginary_residual"] <= MAX_IMAG_RESIDUAL
            and result["raw_mass_error"] <= MAX_RAW_MASS_ERROR)


def _bands(joint: np.ndarray) -> np.ndarray:
    collapsed = joint.sum(axis=2)
    return np.stack((collapsed[:21].sum(axis=0), collapsed[21],
                     collapsed[22:].sum(axis=0)))


def _errors(observed: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    difference = observed - exact
    band_difference = _bands(observed) - _bands(exact)
    return {
        "full_joint_tv": float(np.abs(difference).sum() / 2),
        "band_by_sixth_tv": float(np.abs(band_difference).sum() / 2),
        "maximum_band_mass_error": float(np.max(np.abs(band_difference.sum(axis=1)))),
        "sixth_clutter_mean_error": float(np.dot(
            np.arange(difference.shape[1]), difference.sum(axis=(0, 2))
        )),
        "last_clutter_mean_error": float(np.dot(
            np.arange(difference.shape[2]), difference.sum(axis=(0, 1))
        )),
    }


def _toy_model() -> SlotExistenceMaternModel:
    sequence = DetectionSequence(
        times=np.arange(3, dtype=float) * 0.001,
        detections=(np.asarray([[-0.008, 0.002]]), np.empty((0, 2)),
                    np.asarray([[0.008, -0.002]])),
    )
    return SlotExistenceMaternModel(
        sequence, n_slots=4, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.5, expected_clutter_per_frame=0.35,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
    )


def _toy_validations() -> dict[str, Any]:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    prior = model.log_existence_prior(support)
    count = model.log_count_density(support, include_poisson_void=True)
    gp = model.log_marginal_density(support)
    alive = np.count_nonzero(support != ABSENT, axis=(1, 2))
    first_clutter = 1 - np.count_nonzero(support[:, 0] >= 0, axis=1)
    last_clutter = 1 - np.count_nonzero(support[:, 2] >= 0, axis=1)
    cells = 4 * alive + 2 * first_clutter + last_clutter
    computer = _ExactComputer(model, (0, 2))
    results = {}
    for beta in (0.0, 0.5, 1.0):
        scores = prior + count + beta * gp
        expected_log_z = float(logsumexp(scores))
        expected = (np.bincount(cells, weights=np.exp(scores - expected_log_z),
                                minlength=13 * 2 * 2).reshape(13, 2, 2))
        actual = computer.compute(beta)
        joint = np.asarray(actual["joint"])
        delta = joint - expected
        errors = {
            "log_partition_absolute_error": abs(actual["log_partition"] - expected_log_z),
            "joint_tv": float(np.abs(delta).sum() / 2),
            "maximum_cell_absolute_error": float(np.max(np.abs(delta))),
            "numerical": {key: actual[key] for key in (
                "minimum_raw_cell", "maximum_relative_imaginary_residual",
                "raw_mass_error", "clipping_tv"
            )},
        }
        results[str(beta)] = errors
        if (errors["log_partition_absolute_error"] > EXACT_TOY_LOGZ_TOL
                or errors["joint_tv"] > EXACT_TOY_TV_TOL
                or errors["maximum_cell_absolute_error"] > EXACT_TOY_CELL_TOL
                or not _numeric_ok(actual)):
            raise ValueError(f"toy exact validation failed at beta={beta}: {errors}")
    return results


def _reference_protocol() -> dict[str, Any]:
    protocol = _protocol()
    protocol["reference_sha256"] = None
    return protocol


def run_reference(output: Path) -> dict[str, Any]:
    case, metadata = _case()
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["protocol"] != _reference_protocol() or saved["case"] != metadata:
            raise ValueError("reference checkpoint differs from frozen case")
        return saved
    result: dict[str, Any] = {
        "experiment": "wp11v_tempering_reference", "protocol": _reference_protocol(),
        "case": metadata, "toy_validations": None, "selected_checks": {},
        "status": "reference_in_progress",
    }
    result["toy_validations"] = _toy_validations()
    computer = _ExactComputer(case.model, (5, 7))
    for beta in (0.0, 0.5, 1.0):
        direct = computer.compute(beta, "direct")
        paired = computer.compute(beta, "paired")
        a, b = np.asarray(direct["joint"]), np.asarray(paired["joint"])
        numerical = {
            "log_partition_absolute_difference": abs(
                direct["log_partition"] - paired["log_partition"]
            ),
            "joint_tv": float(np.abs(a - b).sum() / 2),
            "maximum_cell_absolute_difference": float(np.max(np.abs(a - b))),
        }
        result["selected_checks"][str(beta)] = {
            "direct": direct, "paired_numerical": {
                **numerical,
                "minimum_raw_cell": paired["minimum_raw_cell"],
                "maximum_relative_imaginary_residual": paired[
                    "maximum_relative_imaginary_residual"
                ],
                "raw_mass_error": paired["raw_mass_error"],
                "clipping_tv": paired["clipping_tv"],
                "cpu_seconds": paired["cpu_seconds"],
            },
        }
        if (not _numeric_ok(direct) or not _numeric_ok(paired)
                or numerical["log_partition_absolute_difference"] > NUMERIC_LOGZ_TOL
                or numerical["joint_tv"] > NUMERIC_TV_TOL
                or numerical["maximum_cell_absolute_difference"] > NUMERIC_CELL_TOL
                or beta == 0 and abs(direct["log_partition"]
                                     - case.model.base_log_normalizer) > NUMERIC_LOGZ_TOL):
            result["status"] = "numerical_reference_failed_stop"
            _save(output, result)
            return result
        _save(output, result)
    result["base_log_normalizer"] = case.model.base_log_normalizer
    result["status"] = "validated_reference_before_sampler"
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry",), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, metadata = _case()
    result = {"protocol": _protocol(), "case": metadata}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
