"""Passive joint-mode diagnosis of the completed WP11g exposed-case runs.

Reproduce the two frozen seeds with the same sampler and CPU cap. The observer
uses no random numbers and must reproduce every saved non-timing summary.
This is a post-hoc diagnosis, not another stability or holdout test.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .existence_smc import sample_slot_existence_posterior
from .slot_existence import ABSENT
from .wp7_convergence_audit import _large_run_summary, _save
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import (
    CPU_LIMIT_SECONDS,
    SEEDS,
    _candidate_controls,
    _frozen_inputs,
)

REFERENCE_SHA256 = "c435ecb10bfe32bd108666ecd5839d35a21d578810cce65b050fc52b77c1f2a4"
FRAME_INDEX = 5  # The sixth frame, identified after WP11g was completed.


def _counts(states: NDArray[np.int64], cardinality: int) -> tuple[NDArray, NDArray]:
    alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
    clutter = cardinality - np.count_nonzero(states[:, FRAME_INDEX] >= 0, axis=1)
    return alive, clutter


def _joint(states: NDArray[np.int64], cardinality: int) -> NDArray[np.int64]:
    alive, clutter = _counts(states, cardinality)
    if np.any(clutter < 0) or np.any(clutter > cardinality):
        raise ValueError("invalid frame-six clutter count")
    return np.bincount(
        alive * (cardinality + 1) + clutter,
        minlength=(states.shape[1] * states.shape[2] + 1) * (cardinality + 1),
    ).reshape(states.shape[1] * states.shape[2] + 1, cardinality + 1)


def _empty_move(n_frames: int) -> dict[str, Any]:
    return {
        "selected": 0,
        "accepted": 0,
        "state_changed": 0,
        "count_signature_changed": 0,
        "alive_total_changed": 0,
        "frame_six_clutter_changed": 0,
        "both_changed": 0,
        "per_frame_alive_changes": [0] * n_frames,
        "per_frame_assigned_changes": [0] * n_frames,
        "alive_band_transitions": [[0] * 3 for _ in range(3)],
    }


class JointModeObserver:
    """Compact label-invariant stage and pair-update statistics."""

    def __init__(self, cardinality: int, n_frames: int) -> None:
        self.cardinality = cardinality
        self.n_frames = n_frames
        self.stages: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None

    def begin_stage(self, step: int, beta: float) -> None:
        self._active = {
            "step": step,
            "beta": float(beta),
            "moves": {"local": _empty_move(self.n_frames), "pg": _empty_move(self.n_frames)},
        }

    def record_resampling(
        self, associations: NDArray[np.int64], weights: NDArray[np.float64],
        ancestors: NDArray[np.int64],
    ) -> None:
        assert self._active is not None
        alive, clutter = _counts(associations, self.cardinality)
        indexes = alive * (self.cardinality + 1) + clutter
        n_bins = (associations.shape[1] * associations.shape[2] + 1) * (
            self.cardinality + 1
        )
        weighted = np.bincount(indexes, weights=weights, minlength=n_bins)
        resampled = np.bincount(indexes[ancestors], minlength=n_bins) / len(ancestors)
        self._active["resampling_joint_tv"] = float(np.abs(weighted - resampled).sum() / 2)

    def record_pair(
        self, before: NDArray[np.int64], proposed: NDArray[np.int64],
        accepted: NDArray[np.bool_], pg_selected: NDArray[np.bool_],
    ) -> None:
        assert self._active is not None
        before_alive = np.count_nonzero(before != ABSENT, axis=2)
        after_alive = np.count_nonzero(proposed != ABSENT, axis=2)
        before_assigned = np.count_nonzero(before >= 0, axis=2)
        after_assigned = np.count_nonzero(proposed >= 0, axis=2)
        before_total = before_alive.sum(axis=1)
        after_total = after_alive.sum(axis=1)
        alive_changed = before_total != after_total
        clutter_changed = before_assigned[:, FRAME_INDEX] != after_assigned[:, FRAME_INDEX]
        count_changed = np.any(
            (before_alive != after_alive) | (before_assigned != after_assigned), axis=1
        )
        state_changed = np.any(before != proposed, axis=(1, 2))
        before_band = np.digitize(before_total, (20, 21), right=True)
        after_band = np.digitize(after_total, (20, 21), right=True)
        for name, mask in (("local", ~pg_selected), ("pg", pg_selected)):
            stats = self._active["moves"][name]
            changed = mask & accepted
            stats["selected"] += int(mask.sum())
            stats["accepted"] += int(changed.sum())
            stats["state_changed"] += int(np.count_nonzero(changed & state_changed))
            stats["count_signature_changed"] += int(np.count_nonzero(changed & count_changed))
            stats["alive_total_changed"] += int(np.count_nonzero(changed & alive_changed))
            stats["frame_six_clutter_changed"] += int(np.count_nonzero(changed & clutter_changed))
            stats["both_changed"] += int(
                np.count_nonzero(changed & alive_changed & clutter_changed)
            )
            for time_index in range(self.n_frames):
                stats["per_frame_alive_changes"][time_index] += int(np.count_nonzero(
                    changed & (before_alive[:, time_index] != after_alive[:, time_index])
                ))
                stats["per_frame_assigned_changes"][time_index] += int(np.count_nonzero(
                    changed & (before_assigned[:, time_index] != after_assigned[:, time_index])
                ))
            transitions = np.bincount(
                3 * before_band[changed] + after_band[changed], minlength=9
            ).reshape(3, 3)
            for i in range(3):
                for j in range(3):
                    stats["alive_band_transitions"][i][j] += int(transitions[i, j])

    def end_stage(self, associations: NDArray[np.int64]) -> None:
        assert self._active is not None
        joint = _joint(associations, self.cardinality)
        self._active["joint_alive_total_by_frame_six_clutter"] = joint.tolist()
        self._active["mean_alive_total"] = float(
            np.dot(np.arange(joint.shape[0]), joint.sum(axis=1)) / joint.sum()
        )
        self._active["mean_frame_six_clutter"] = float(
            np.dot(np.arange(joint.shape[1]), joint.sum(axis=0)) / joint.sum()
        )
        self.stages.append(self._active)
        self._active = None


def _protocol() -> dict[str, Any]:
    return {
        "purpose": "passive diagnosis, not an added candidate or stability gate",
        "reference_file_sha256": REFERENCE_SHA256,
        "case": "cluttered_staggered_crossing",
        "seeds": list(SEEDS),
        "candidate_controls": "unchanged WP11g four-path 10% mixture",
        "maximum_process_cpu_seconds_per_seed": CPU_LIMIT_SECONDS,
        "maximum_seed_reproductions": 2,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "frame_index_zero_based": FRAME_INDEX,
        "alive_bands": ["0..20", "21", "22..32"],
        "final_count_signatures": "all 8192 per-particle alive/assigned vectors, zlib+base64 uint8",
        "reproduction_check": "all saved WP11g summary fields except measured wall runtime equal exactly",
        "holdout": "closed",
    }


def _reference(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("WP11g reference checkpoint hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "failed_early_before_new_holdout":
        raise ValueError("WP11g checkpoint is not final")
    if set(result["candidate_runs"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("WP11g candidate seeds differ")
    return result


def _encode_signatures(associations: NDArray[np.int64]) -> dict[str, Any]:
    alive = np.count_nonzero(associations != ABSENT, axis=2)
    assigned = np.count_nonzero(associations >= 0, axis=2)
    signatures = np.concatenate((alive, assigned), axis=1).astype(np.uint8)
    raw = signatures.tobytes()
    return {
        "shape": list(signatures.shape),
        "encoding": "zlib+base64:uint8:C-order",
        "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def _decode_signatures(record: dict[str, Any]) -> NDArray[np.uint8]:
    if record["encoding"] != "zlib+base64:uint8:C-order":
        raise ValueError("unknown count-signature encoding")
    raw = zlib.decompress(base64.b64decode(record["data"], validate=True))
    if hashlib.sha256(raw).hexdigest() != record["sha256_uncompressed"]:
        raise ValueError("count-signature checksum differs")
    if len(raw) != int(np.prod(record["shape"])):
        raise ValueError("count-signature shape differs")
    return np.frombuffer(raw, dtype=np.uint8).reshape(record["shape"])


def _diagnosis(runs: dict[str, Any]) -> dict[str, Any]:
    first, second = (runs[str(seed)]["stages"] for seed in SEEDS)
    if len(first) != len(second):
        raise ValueError("different number of tempering stages")
    stage_differences = []
    for a, b in zip(first, second, strict=True):
        x = np.asarray(a["joint_alive_total_by_frame_six_clutter"])
        y = np.asarray(b["joint_alive_total_by_frame_six_clutter"])
        px, py = x / x.sum(), y / y.sum()
        stage_differences.append({
            "step": a["step"],
            "beta_100": a["beta"],
            "beta_200": b["beta"],
            "alive_total_tv": float(np.abs(px.sum(axis=1) - py.sum(axis=1)).sum() / 2),
            "frame_six_clutter_mean_difference": float(abs(
                a["mean_frame_six_clutter"] - b["mean_frame_six_clutter"]
            )),
            "joint_tv": float(np.abs(px - py).sum() / 2),
        })
    final_a = np.asarray(first[-1]["joint_alive_total_by_frame_six_clutter"])
    final_b = np.asarray(second[-1]["joint_alive_total_by_frame_six_clutter"])
    groups = {}
    for name, slc in (("0..20", slice(None, 21)), ("21", slice(21, 22)),
                      ("22..32", slice(22, None))):
        groups[name] = []
        for joint in (final_a, final_b):
            sub = joint[slc]
            mass = float(sub.sum() / joint.sum())
            conditional_clutter = (
                float(np.dot(np.arange(joint.shape[1]), sub.sum(axis=0)) / sub.sum())
                if sub.sum() else None
            )
            groups[name].append({"mass": mass, "conditional_frame_six_clutter": conditional_clutter})
    signature_a, signature_b = (
        _decode_signatures(runs[str(seed)]["final_count_signatures"]) for seed in SEEDS
    )
    n_frames = signature_a.shape[1] // 2
    comparisons = {}
    for name, view in (("alive_per_frame", slice(None, n_frames)),
                       ("assigned_per_frame", slice(n_frames, None)),
                       ("complete_count_signature", slice(None))):
        x, y = signature_a[:, view], signature_b[:, view]
        patterns, inverse = np.unique(np.concatenate((x, y)), axis=0, return_inverse=True)
        p = np.bincount(inverse[:len(x)], minlength=len(patterns)) / len(x)
        q = np.bincount(inverse[len(x):], minlength=len(patterns)) / len(y)
        order = np.argsort(np.abs(p - q))[::-1][:10]
        comparisons[name] = {
            "descriptive_tv": float(np.abs(p - q).sum() / 2),
            "distinct_patterns": len(patterns),
            "largest_mass_differences": [
                {"pattern": patterns[i].tolist(), "seed_100_mass": float(p[i]),
                 "seed_200_mass": float(q[i])} for i in order
            ],
        }
    return {
        "stage_differences": stage_differences,
        "final_alive_groups": groups,
        "signature_comparisons": comparisons,
    }


def run_seed(*, output: Path, control_path: Path, reference_path: Path, seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("seed must be 100 or 200")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread environment variables to 1")
    case, _, metadata = _frozen_inputs(control_path)
    reference = _reference(reference_path)
    if metadata != reference["case"]:
        raise ValueError("WP11g case metadata differs")
    protocol = _protocol()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("existing checkpoint belongs to a different diagnostic protocol")
    else:
        result = {
            "experiment": "wp11h_joint_mode_diagnostic", "protocol": protocol,
            "runs": {}, "diagnosis": None, "status": "in_progress",
        }
        _save(output, result)
    if str(seed) in result["runs"]:
        print(f"seed {seed}: reused", flush=True)
        return result
    if result["status"] != "in_progress":
        raise ValueError("diagnostic has already completed")
    if seed == SEEDS[1] and str(SEEDS[0]) not in result["runs"]:
        raise ValueError("run seed 100 first")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < CPU_LIMIT_SECONDS + 5:
        raise RuntimeError("system CPU hard limit is below the frozen budget")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS + 5))
    observer = JointModeObserver(
        case.model.sequence.cardinalities[FRAME_INDEX], case.model.sequence.n_frames
    )
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence, n_slots=case.model.n_slots,
        localization_std=case.localization_std, coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower, roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior,
        config=_candidate_controls(seed), observer=observer,
    )
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    summary = _large_run_summary(case, sampled, wall_seconds, seed)
    expected = reference["candidate_runs"][str(seed)]["summary"].copy()
    expected.pop("runtime_seconds")
    summary.pop("runtime_seconds")
    if summary != expected:
        raise ValueError("passive diagnostic did not reproduce WP11g non-timing summary")
    if len(observer.stages) != sampled.n_tempering_steps:
        raise ValueError("stage diagnostic count differs")
    result["runs"][str(seed)] = {
        "reproduced_wp11g_summary": True,
        "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
        "stages": observer.stages,
        "final_count_signatures": _encode_signatures(sampled.associations),
    }
    if len(result["runs"]) == len(SEEDS):
        result["diagnosis"] = _diagnosis(result["runs"])
        result["status"] = "diagnostic_complete_no_holdout"
    _save(output, result)
    print(f"seed {seed}: {cpu_seconds:.1f}s CPU, {wall_seconds:.1f}s wall; "
          f"status={result['status']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--control", type=Path, default=Path("results/wp8_generalization.json"))
    parser.add_argument("--reference", type=Path, default=Path("results/wp11g_exposed_wp8_high_particle.json"))
    parser.add_argument("--output", type=Path, default=Path("results/wp11h_joint_mode_diagnostic.json"))
    args = parser.parse_args()
    run_seed(output=args.output, control_path=args.control, reference_path=args.reference,
             seed=args.seed)


if __name__ == "__main__":
    main()
