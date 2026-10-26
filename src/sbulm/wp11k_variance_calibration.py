"""Frozen exact-reference and multi-seed variance audit of unchanged WP11g PG."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict, replace
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
    slot_event_marginals,
    slot_existence_support_size,
)
from .wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR, _ScalingCase
from .wp7_convergence_audit import _case_metadata, _compare_large, _large_run_summary, _save
from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)
from .wp11f_interleaved_cost import THREAD_ENV
from .wp11g_exposed_wp8_high_particle import _candidate_controls

EXACT_HASH = "879487937d467118e113a552e88412b828967dbfc2e3a42868d2658f861f9de9"
EXACT_SUPPORT_SIZE = 36_590
REGIMES = (
    GeneralizationRegime(
        name="wp11k_primary", detection_probability=0.68,
        expected_clutter_per_frame=0.28, central_separation=0.007,
        trajectory_seed=79, corruption_seed=83, max_clutter_per_frame=2,
        truth_mask=STAGGERED_TRUTH_MASK,
    ),
    GeneralizationRegime(
        name="wp11k_stress", detection_probability=0.62,
        expected_clutter_per_frame=0.50, central_separation=0.004,
        trajectory_seed=89, corruption_seed=97, max_clutter_per_frame=3,
        truth_mask=STAGGERED_TRUTH_MASK,
    ),
)
BANK_HASHES = {
    "wp11k_primary": "e7fc37d377e5b956377af6a64e41837f15e42d43e61a932af48400da9bffd922",
    "wp11k_stress": "7f6d4ebab4984dd5243684587b1a9da7e6babe71f4e2f79494395bf2b113b495",
}
BANK_CARDINALITIES = {
    "wp11k_primary": (2, 2, 4, 4, 3, 2, 1, 1),
    "wp11k_stress": (1, 1, 2, 2, 3, 1, 2, 0),
}
EXACT_SEEDS = (820, 821, 822, 823, 824, 825)
PRIMARY_SEEDS = (800, 801, 802, 803)
STRESS_SEEDS = (810, 811, 812, 813)
EXACT_PARTICLES = (512, 2_048)
PRIMARY_PARTICLES = (512, 2_048)
STRESS_PARTICLES = (512,)
IID_REPLICATES = 200
REFERENCE_CPU_CAP = 600
CPU_CAPS = {"exact": {512: 250, 2048: 450},
            "wp11k_primary": {512: 300, 2048: 750},
            "wp11k_stress": {512: 300}}


def _exact_case() -> _ScalingCase:
    sequence = DetectionSequence(
        times=np.arange(4, dtype=float) * 0.001,
        detections=(
            np.asarray([[-0.009, 0.001]]),
            np.empty((0, 2)),
            np.asarray([[0.008, 0.0], [-0.007, 0.002]]),
            np.asarray([[-0.010, 0.001]]),
        ),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=3, localization_std=0.005,
        coordinate_std=(0.02, 0.01), length_scale=0.003,
        detection_probability=0.65, expected_clutter_per_frame=0.25,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model, truth_mask=np.zeros((4, 3), dtype=bool),
        localization_std=0.005, coordinate_std=(0.02, 0.01), length_scale=0.003,
    )


def _protocol() -> dict[str, Any]:
    return {
        "sampler": "unchanged WP11g 10% four-path PG mixture",
        "exact_case": {
            "hash": EXACT_HASH, "support_size": EXACT_SUPPORT_SIZE,
            "frames": 4, "slots": 3, "cardinalities": [1, 0, 2, 1],
            "seeds": list(EXACT_SEEDS), "particles": list(EXACT_PARTICLES),
        },
        "bank": {
            r.name: {"regime": json.loads(json.dumps(asdict(r))),
                     "hash": BANK_HASHES[r.name],
                     "cardinalities": list(BANK_CARDINALITIES[r.name]),
                     "seeds": list(PRIMARY_SEEDS if r.name == "wp11k_primary" else STRESS_SEEDS),
                     "particles": list(PRIMARY_PARTICLES if r.name == "wp11k_primary"
                                       else STRESS_PARTICLES)}
            for r in REGIMES
        },
        "iid_exact_replicates": IID_REPLICATES,
        "iid_baseline_seed": 18_113,
        "thread_env": {name: "1" for name in THREAD_ENV},
        "reference_cpu_cap": REFERENCE_CPU_CAP,
        "per_run_cpu_caps": {case: {str(n): cap for n, cap in sizes.items()}
                             for case, sizes in CPU_CAPS.items()},
        "run_order": (
            "reference, exact 512 then 2048 seeds 820..825; primary 512 then 2048 "
            "seeds 800..803; stress 512 seeds 810..813"
        ),
        "audit": (
            "report absolute exact-case count-TV and clutter error against 200 iid draws; "
            "compare four-seed primary pairwise seven-metric spread at 512 and 2048; "
            "report joint mode trajectories and four-seed stress spread at 512"
        ),
        "interpretation": (
            "diagnostic, not a candidate-selection gate; no WP8 or untouched holdout run"
        ),
    }


def _metadata_for_cases() -> dict[str, Any]:
    exact = _exact_case()
    exact_info = _case_metadata(exact)
    if (exact_info["detections_sha256"] != EXACT_HASH
            or slot_existence_support_size(exact.model) != EXACT_SUPPORT_SIZE):
        raise ValueError("exact-reference case changed")
    bank = {}
    for regime in REGIMES:
        info = _metadata(regime, make_case(regime))
        observed = info["model_and_detections"]
        if (observed["detections_sha256"] != BANK_HASHES[regime.name]
                or tuple(observed["detection_cardinalities"])
                != BANK_CARDINALITIES[regime.name]):
            raise ValueError("eight-frame observation changed")
        bank[regime.name] = info
    return {"exact": exact_info, "bank": bank}


def _load(output: Path) -> dict[str, Any]:
    protocol, metadata = _protocol(), _metadata_for_cases()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["cases"] != metadata:
            raise ValueError("checkpoint belongs to a different frozen audit")
        return result
    result = {
        "experiment": "wp11k_variance_calibration", "protocol": protocol,
        "cases": metadata, "reference": None,
        "runs": {
            "exact": {str(n): {} for n in EXACT_PARTICLES},
            "wp11k_primary": {str(n): {} for n in PRIMARY_PARTICLES},
            "wp11k_stress": {"512": {}},
        },
        "comparisons": None, "status": "in_progress",
    }
    _save(output, result)
    return result


def _histogram(values: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(values, weights=weights, minlength=size)[:size]


def _count_signatures(states: np.ndarray) -> np.ndarray:
    return np.concatenate((np.count_nonzero(states != ABSENT, axis=2),
                           np.count_nonzero(states >= 0, axis=2)), axis=1)


def _reference_errors(states: np.ndarray, reference: dict[str, Any]) -> dict[str, float]:
    n = len(states)
    weights = np.full(n, 1 / n)
    alive_total = np.count_nonzero(states != ABSENT, axis=(1, 2))
    alive_hist = _histogram(alive_total, weights, len(reference["alive_total_histogram"]))
    alive_tv = float(np.abs(alive_hist - reference["alive_total_histogram"]).sum() / 2)
    assigned_frame = np.count_nonzero(states >= 0, axis=2)
    clutter = np.asarray(reference["cardinalities"]) - np.mean(assigned_frame, axis=0)
    clutter_error = float(np.max(np.abs(clutter - reference["expected_clutter"])))
    lookup = {tuple(row): i for i, row in enumerate(reference["count_signatures"])}
    indices = np.asarray([lookup[tuple(row)] for row in _count_signatures(states)])
    signature_hist = _histogram(indices, weights, len(lookup))
    signature_tv = float(np.abs(signature_hist - reference["count_signature_probabilities"]).sum() / 2)
    return {"alive_total_tv": alive_tv, "maximum_clutter_error": clutter_error,
            "count_signature_tv": signature_tv}


def _baseline(
    support: np.ndarray, probabilities: np.ndarray, reference: dict[str, Any]
) -> dict[str, Any]:
    rng = np.random.default_rng(_protocol()["iid_baseline_seed"])
    values = {}
    for n in EXACT_PARTICLES:
        draws = {name: [] for name in (
            "alive_total_tv", "maximum_clutter_error", "count_signature_tv"
        )}
        for _ in range(IID_REPLICATES):
            indices = rng.choice(len(support), size=n, p=probabilities)
            measured = _reference_errors(support[indices], reference)
            for key, value in measured.items():
                draws[key].append(value)
        values[str(n)] = {
            key: {"median": float(np.median(seq)), "q95": float(np.quantile(seq, .95))}
            for key, seq in draws.items()
        }
    return values


def _set_cpu_limit(seconds: int) -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY and hard < seconds + 5:
        raise RuntimeError("system hard CPU limit is below frozen budget")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))


def run_reference(output: Path) -> dict[str, Any]:
    result = _load(output)
    if result["reference"] is not None:
        print("exact reference: reused", flush=True)
        return result
    _set_cpu_limit(REFERENCE_CPU_CAP)
    case = _exact_case()
    started = time.process_time()
    support, _ = enumerate_slot_existence_support(case.model)
    log_joint = case.model.log_joint_density(support)
    log_evidence = float(logsumexp(log_joint))
    probabilities = np.exp(log_joint - log_evidence)
    events = slot_event_marginals(
        support, case.model.sequence, probabilities=probabilities
    )
    signatures, inverse = np.unique(_count_signatures(support), axis=0, return_inverse=True)
    result["reference"] = {
        "support_size": len(support), "log_evidence": log_evidence,
        "cpu_seconds": time.process_time() - started,
        "cardinalities": list(case.model.sequence.cardinalities),
        "alive_total_histogram": _histogram(
            np.count_nonzero(support != ABSENT, axis=(1, 2)), probabilities, 13
        ).tolist(),
        "expected_clutter": events.expected_clutter.tolist(),
        "event_marginals": {
            "existence": events.existence_probabilities.tolist(),
            "missed": events.missed_probabilities.tolist(),
            "clutter": events.expected_clutter.tolist(),
            "birth": events.birth_time_probabilities.tolist(),
            "death": events.death_time_probabilities.tolist(),
        },
        "count_signatures": signatures.tolist(),
        "count_signature_probabilities": _histogram(
            inverse, probabilities, len(signatures)
        ).tolist(),
    }
    result["reference"]["iid_baseline"] = _baseline(
        support, probabilities, result["reference"]
    )
    _save(output, result)
    print(f"exact reference: {len(support):,} states; "
          f"{result['reference']['cpu_seconds']:.1f}s CPU", flush=True)
    return result


class _JointObserver:
    """Passive label-invariant joint mode trace, without RNG consumption."""

    def __init__(self, case: _ScalingCase) -> None:
        self.n_slots = case.model.n_slots
        self.n_frames = case.model.sequence.n_frames
        self.frame = min(5, self.n_frames - 1)
        self.cardinality = case.model.sequence.cardinalities[self.frame]
        self.records: list[dict[str, Any]] = []

    def _joint(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        alive = np.count_nonzero(states != ABSENT, axis=(1, 2))
        boundary = int(self.n_slots * self.n_frames * .625)
        band = np.where(alive <= boundary, 0, np.where(alive == boundary + 1, 1, 2))
        clutter = self.cardinality - np.count_nonzero(states[:, self.frame] >= 0, axis=1)
        joint = np.zeros((3, self.cardinality + 1), dtype=float)
        np.add.at(joint, (band, clutter), weights)
        return joint

    def begin_stage(self, step: int, beta: float) -> None:
        self.records.append({"step": step, "beta": float(beta)})

    def record_resampling(
        self, associations: np.ndarray, weights: np.ndarray, ancestors: np.ndarray
    ) -> None:
        weighted = self._joint(associations, weights)
        resampled = self._joint(
            associations[ancestors], np.full(len(ancestors), 1 / len(ancestors))
        )
        self.records[-1]["weighted_to_resampled_joint_tv"] = float(
            np.abs(weighted - resampled).sum() / 2
        )

    def record_pair(
        self, before: np.ndarray, proposed: np.ndarray,
        accepted: np.ndarray, pg_selected: np.ndarray,
    ) -> None:
        return

    def end_stage(self, associations: np.ndarray) -> None:
        joint = self._joint(
            associations, np.full(len(associations), 1 / len(associations))
        )
        self.records[-1]["post_joint"] = joint.tolist()
        self.records[-1]["low_alive_band_mass"] = float(np.sum(joint[0]))
        self.records[-1]["high_alive_band_mass"] = float(np.sum(joint[2]))
        self.records[-1]["expected_clutter_at_frame"] = float(
            np.sum(joint * np.arange(self.cardinality + 1)[None, :])
        )


def _run_order() -> list[tuple[str, int, int]]:
    return (
        [("exact", n, seed) for n in EXACT_PARTICLES for seed in EXACT_SEEDS]
        + [("wp11k_primary", n, seed)
           for n in PRIMARY_PARTICLES for seed in PRIMARY_SEEDS]
        + [("wp11k_stress", 512, seed) for seed in STRESS_SEEDS]
    )


def run_one(output: Path, *, case_name: str, n_particles: int, seed: int) -> dict[str, Any]:
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("set all four BLAS/OMP thread variables to 1")
    if (case_name, n_particles, seed) not in _run_order():
        raise ValueError("run is outside the frozen audit")
    result = _load(output)
    if result["reference"] is None:
        raise ValueError("run the exact reference first")
    key = str(n_particles)
    if str(seed) in result["runs"][case_name][key]:
        print(f"{case_name} {n_particles} seed {seed}: reused", flush=True)
        return result
    completed = [triplet for triplet in _run_order()
                 if str(triplet[2]) in result["runs"][triplet[0]][str(triplet[1])]]
    if (case_name, n_particles, seed) != _run_order()[len(completed)]:
        raise ValueError("run the frozen case, particle-count and seed order")
    _set_cpu_limit(CPU_CAPS[case_name][n_particles])
    case = (_exact_case() if case_name == "exact" else make_case(
        next(regime for regime in REGIMES if regime.name == case_name)
    ))
    controls = replace(_candidate_controls(seed), n_particles=n_particles)
    observer = _JointObserver(case)
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
    cpu_seconds, wall_seconds = time.process_time() - started_cpu, time.perf_counter() - started_wall
    summary = _large_run_summary(case, sampled, wall_seconds, seed)
    record: dict[str, Any] = {
        "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
        "summary": summary, "joint_trace": observer.records,
    }
    if case_name == "exact":
        reference = result["reference"]
        record["errors"] = _reference_errors(sampled.associations, reference)
        record["errors"]["maximum_event_error"] = max(
            float(np.max(np.abs(np.asarray(summary["event_marginals"][name])
                                - np.asarray(reference["event_marginals"][name]))))
            for name in ("existence", "missed", "birth", "death")
        )
        record["errors"]["log_evidence_error"] = (
            summary["log_evidence_estimate"] - reference["log_evidence"]
        )
    result["runs"][case_name][key][str(seed)] = record
    if len(completed) + 1 == len(_run_order()):
        _finish(result)
    _save(output, result)
    print(f"{case_name} {n_particles} seed {seed}: {cpu_seconds:.1f}s CPU, "
          f"{sampled.n_tempering_steps} stages; status={result['status']}", flush=True)
    return result


def _pairwise_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_runs = [r["summary"] for r in records]
    pairs = [_compare_large([all_runs[i], all_runs[j]])["metrics"]
             for i in range(len(records)) for j in range(i + 1, len(records))]
    return {
        "four_seed_maxima": _compare_large(all_runs)["metrics"],
        "pairwise_median": {key: float(median(p[key] for p in pairs)) for key in pairs[0]},
        "pairwise_range": {key: [min(p[key] for p in pairs), max(p[key] for p in pairs)]
                           for key in pairs[0]},
        "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in records])),
        "final_low_band_mass": [r["joint_trace"][-1]["low_alive_band_mass"]
                                for r in records],
        "final_high_band_mass": [r["joint_trace"][-1]["high_alive_band_mass"]
                                 for r in records],
        "final_joint_mode_tv_max": max(
            float(np.abs(np.asarray(records[i]["joint_trace"][-1]["post_joint"])
                         - np.asarray(records[j]["joint_trace"][-1]["post_joint"])).sum() / 2)
            for i in range(len(records)) for j in range(i + 1, len(records))
        ),
    }


def _finish(result: dict[str, Any]) -> None:
    reference = result["reference"]
    exact = {}
    for n in EXACT_PARTICLES:
        runs = [result["runs"]["exact"][str(n)][str(seed)] for seed in EXACT_SEEDS]
        errors = [r["errors"] for r in runs]
        exact[str(n)] = {
            "median_errors": {key: float(median(e[key] for e in errors))
                              for key in errors[0]},
            "ranges": {key: [min(e[key] for e in errors), max(e[key] for e in errors)]
                       for key in errors[0]},
            "iid_q95_exceedances": {
                key: sum(e[key] > reference["iid_baseline"][str(n)][key]["q95"]
                         for e in errors)
                for key in ("alive_total_tv", "maximum_clutter_error", "count_signature_tv")
            },
            "mean_cpu_seconds": float(np.mean([r["cpu_seconds"] for r in runs])),
        }
    primary = {str(n): _pairwise_summary([
        result["runs"]["wp11k_primary"][str(n)][str(seed)] for seed in PRIMARY_SEEDS
    ]) for n in PRIMARY_PARTICLES}
    stress = _pairwise_summary([
        result["runs"]["wp11k_stress"]["512"][str(seed)] for seed in STRESS_SEEDS
    ])
    result["comparisons"] = {
        "exact": exact, "primary": primary, "stress_512": stress,
        "primary_2048_512_median_pairwise_ratios": {
            key: (primary["2048"]["pairwise_median"][key]
                  / primary["512"]["pairwise_median"][key])
            for key in primary["512"]["pairwise_median"]
            if primary["512"]["pairwise_median"][key] > 0
        },
    }
    result["status"] = "audit_completed_no_candidate_gate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "exact", *BANK_HASHES), required=True)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11k_variance_calibration.json"))
    args = parser.parse_args()
    if args.case == "reference":
        if args.particles is not None or args.seed is not None:
            parser.error("the reference requires no particle count or seed")
        run_reference(args.output)
    else:
        if args.particles is None or args.seed is None:
            parser.error("a sampler run requires --particles and --seed")
        run_one(args.output, case_name=args.case, n_particles=args.particles, seed=args.seed)


if __name__ == "__main__":
    main()
