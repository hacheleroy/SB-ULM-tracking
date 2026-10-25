"""Frozen exact-oracle and synthetic transfer checks for guided tracklet moves.

Run the four-slot exact oracle first. Only if every oracle run passes do the
unchanged WP8 failure and two predeclared, previously untested synthetic
realizations proceed. A two-seed eight-frame comparison is a diagnostic,
not an exact posterior calibration or the earlier six-run stability test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .slot_existence import ABSENT, slot_event_marginals
from .tsbm_reference import compare_tsbm_association_samples
from .wp5_scaling_calibration import _event_errors, _exact_posterior, _sample_case
from .wp7_convergence_audit import (
    STABILITY_THRESHOLD,
    _case_metadata,
    _compare_large,
    _large_run_summary,
    _make_case,
    _save,
)
from .wp8_generalization import REGIMES, make_case

N_PARTICLES_ORACLE = 2_048
N_PARTICLES_LARGE = 8_192
ORACLE_SEEDS = (300, 301, 302)
LARGE_SEEDS = (100, 200)
BRIDGE_HASH = "62d3aeeafff58eb47404f87291829004f5bd498d07ab1494afa86ba77115b8ef"
CASE_REGIMES = (
    REGIMES[1],
    replace(
        REGIMES[1],
        name="cluttered_staggered_new_realization_a",
        trajectory_seed=29,
        corruption_seed=18,
    ),
    replace(
        REGIMES[1],
        name="cluttered_staggered_new_realization_b",
        trajectory_seed=31,
        corruption_seed=19,
    ),
)
EXPECTED_HASHES = (
    "afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae",
    "c5c1ddd70435c6fdc4af36c0d2b4d490bb9cb59465e9d8a41d218e1462768d2c",
    "c0301404d31cacaf6db0be996b8997324319ebce1225160092ba9f6117758d2a",
)
MOVE_PROBABILITIES = {
    "association": 0.15,
    "interval": 0.30,
    "slot_swap": 0.05,
    "blocked_slot": 0.15,
    "paired_slot": 0.20,
    "guided_tracklet": 0.15,
}


def _edge_and_clutter_marginals(
    states: np.ndarray, case: Any, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    """Label-invariant observed-detection marginals, not full posterior modes."""

    n_samples, n_frames, _ = states.shape
    weights = (
        np.full(n_samples, 1.0 / n_samples)
        if probabilities is None
        else probabilities
    )
    cardinalities = case.model.sequence.cardinalities
    clutter = []
    edges = []
    for t, cardinality in enumerate(cardinalities):
        clutter.append(
            [
                float(np.sum(weights * ~np.any(states[:, t] == detection, axis=1)))
                for detection in range(cardinality)
            ]
        )
        if t == n_frames - 1:
            continue
        edges.append(
            [
                [
                    float(np.sum(weights * np.any(
                        (states[:, t] == first) & (states[:, t + 1] == second), axis=1
                    )))
                    for second in range(cardinalities[t + 1])
                ]
                for first in range(cardinality)
            ]
        )
    return {"detection_clutter": clutter, "adjacent_association_edges": edges}


def _maximum_nested_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("incompatible marginal names")
        return max((_maximum_nested_difference(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("incompatible nested marginal shapes")
        return max(
            (_maximum_nested_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    return abs(float(left) - float(right))


def _count_signature_histogram(states: np.ndarray) -> dict[str, float]:
    alive = np.count_nonzero(states != ABSENT, axis=2)
    assigned = np.count_nonzero(states >= 0, axis=2)
    signatures, counts = np.unique(
        np.concatenate((alive, assigned), axis=1), axis=0, return_counts=True
    )
    return {
        ",".join(str(value) for value in signature): float(count / states.shape[0])
        for signature, count in zip(signatures, counts, strict=True)
    }


def _signature_comparison(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    keys = set(first) | set(second)
    return {
        "histogram_tv": 0.5 * sum(abs(first.get(k, 0.0) - second.get(k, 0.0)) for k in keys),
        "mass_on_other_run_support_first": sum(v for k, v in first.items() if k in second),
        "mass_on_other_run_support_second": sum(v for k, v in second.items() if k in first),
    }


def _load_or_initialize(output: Path, wp7_output: Path, wp8_output: Path) -> dict[str, Any]:
    bridge = _make_case(bridge=True)
    bridge_meta = _case_metadata(bridge)
    cases = {regime.name: make_case(regime) for regime in CASE_REGIMES}
    metadata = {name: _case_metadata(case) for name, case in cases.items()}
    if bridge_meta["detections_sha256"] != BRIDGE_HASH or any(
        metadata[regime.name]["detections_sha256"] != digest
        for regime, digest in zip(CASE_REGIMES, EXPECTED_HASHES, strict=True)
    ):
        raise RuntimeError("predeclared oracle or large-case detections changed")
    wp7 = json.loads(wp7_output.read_text(encoding="utf-8"))
    wp8 = json.loads(wp8_output.read_text(encoding="utf-8"))
    threshold_record = wp7["bridge"][str(N_PARTICLES_ORACLE)]
    failed = wp8["results"][CASE_REGIMES[0].name]
    if (
        wp7["cases"]["bridge"] != bridge_meta
        or wp8["cases"][CASE_REGIMES[0].name]["model_and_detections"] != metadata[CASE_REGIMES[0].name]
        or failed["status"] != "failed_early"
        or failed["cross_run"]["seeds"] != list(LARGE_SEEDS)
    ):
        raise ValueError("baseline artifacts do not match the frozen oracle and WP8 failure")
    protocol = {
        "method": "WP11 three-frame guided conditional two-slot tracklet MH move",
        "move_probabilities": MOVE_PROBABILITIES,
        "proposal_window": 3,
        "ess_target": 0.8,
        "sweeps_per_stage": 8,
        "extra_final_sweeps": 8,
        "oracle_particles": N_PARTICLES_ORACLE,
        "oracle_seeds": list(ORACLE_SEEDS),
        "oracle_thresholds": threshold_record["thresholds"],
        "large_particles": N_PARTICLES_LARGE,
        "large_seeds": list(LARGE_SEEDS),
        "inherited_seven_metric_gate": STABILITY_THRESHOLD,
        "large_comparison": "two-seed diagnostic; edge, per-detection clutter, and count signatures are ungated",
        "oracle_first": True,
    }
    baseline = failed["cross_run"]
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if (
            result["protocol"] != protocol
            or result["bridge"]["metadata"] != bridge_meta
            or result["cases"] != metadata
            or result["wp8_failed_baseline"] != baseline
        ):
            raise ValueError("existing WP11 checkpoint has a different frozen protocol")
        return result
    return {
        "experiment": "wp11_guided_tracklet_oracle_then_transfer",
        "protocol": protocol,
        "bridge": {"metadata": bridge_meta, "runs": {}, "status": "in_progress"},
        "cases": metadata,
        "wp8_failed_baseline": baseline,
        "results": {
            regime.name: {"runs": {}, "cross_run": None, "status": "in_progress"}
            for regime in CASE_REGIMES
        },
    }


def run_oracle(result: dict[str, Any], output: Path) -> None:
    case = _make_case(bridge=True)
    exact, _ = _exact_posterior(case.model)
    exact_events = slot_event_marginals(
        exact.associations, case.model.sequence, probabilities=exact.probabilities
    )
    exact_edges = _edge_and_clutter_marginals(exact.associations, case, exact.probabilities)
    thresholds = result["protocol"]["oracle_thresholds"]
    for seed in ORACLE_SEEDS:
        if str(seed) in result["bridge"]["runs"]:
            continue
        sampled, runtime = _sample_case(
            case, method="tracklet", n_particles=N_PARTICLES_ORACLE, seed=seed
        )
        comparison = compare_tsbm_association_samples(sampled.associations, exact)
        errors = _event_errors(
            exact_events, slot_event_marginals(sampled.associations, case.model.sequence)
        )
        sampled_edges = _edge_and_clutter_marginals(sampled.associations, case)
        checks = {
            "valid_support": comparison.invalid_sample_fraction == 0.0,
            "total_variation": comparison.total_variation_distance <= thresholds["total_variation"],
            "jensen_shannon": comparison.jensen_shannon_divergence <= thresholds["jensen_shannon"],
            "exact_mass_covered": (
                comparison.exact_mass_covered_by_empirical_support
                >= thresholds["exact_mass_covered"]
            ),
            "credible_set_recall_95": comparison.credible_set_recall_95 >= thresholds["credible_set_recall_95"],
            "event_marginals": max(errors.values()) <= thresholds["maximum_event_error"],
        }
        result["bridge"]["runs"][str(seed)] = {
            "runtime_seconds": runtime,
            "comparison": comparison.to_dict(),
            "maximum_event_error": max(errors.values()),
            "maximum_edge_error": _maximum_nested_difference(
                exact_edges["adjacent_association_edges"], sampled_edges["adjacent_association_edges"]
            ),
            "maximum_detection_clutter_error": _maximum_nested_difference(
                exact_edges["detection_clutter"], sampled_edges["detection_clutter"]
            ),
            "mean_tracklet_acceptance_rate": float(np.mean(
                [stage.tracklet_acceptance_rate for stage in sampled.stages]
            )),
            "mean_tracklet_state_change_rate": float(np.mean(
                [stage.tracklet_state_change_rate for stage in sampled.stages]
            )),
            "checks": checks,
            "passed": all(checks.values()),
        }
        _save(output, result)
        print(f"oracle seed {seed}: {'PASS' if all(checks.values()) else 'FAIL'}, {runtime:.1f}s", flush=True)
    result["bridge"]["status"] = (
        "passed" if all(run["passed"] for run in result["bridge"]["runs"].values())
        and len(result["bridge"]["runs"]) == len(ORACLE_SEEDS) else "failed"
    )
    _save(output, result)


def run_large(result: dict[str, Any], output: Path) -> None:
    if result["bridge"]["status"] != "passed":
        raise RuntimeError("WP11 exact-oracle gate did not pass; large runs are blocked")
    for regime in CASE_REGIMES:
        case = make_case(regime)
        record = result["results"][regime.name]
        for seed in LARGE_SEEDS:
            if str(seed) in record["runs"]:
                continue
            sampled, runtime = _sample_case(
                case, method="tracklet", n_particles=N_PARTICLES_LARGE, seed=seed
            )
            summary = _large_run_summary(case, sampled, runtime, seed)
            summary["detection_marginals"] = _edge_and_clutter_marginals(sampled.associations, case)
            summary["count_signature_histogram"] = _count_signature_histogram(sampled.associations)
            summary["mean_tracklet_acceptance_rate"] = float(np.mean(
                [stage.tracklet_acceptance_rate for stage in sampled.stages]
            ))
            summary["mean_tracklet_state_change_rate"] = float(np.mean(
                [stage.tracklet_state_change_rate for stage in sampled.stages]
            ))
            record["runs"][str(seed)] = summary
            _save(output, result)
            print(f"{regime.name} seed {seed}: {runtime:.1f}s, {sampled.n_tempering_steps} stages", flush=True)
        first, second = (record["runs"][str(seed)] for seed in LARGE_SEEDS)
        comparison = _compare_large([first, second])
        comparison["additional_diagnostics"] = {
            "maximum_detection_clutter_difference": _maximum_nested_difference(
                first["detection_marginals"]["detection_clutter"],
                second["detection_marginals"]["detection_clutter"],
            ),
            "maximum_association_edge_difference": _maximum_nested_difference(
                first["detection_marginals"]["adjacent_association_edges"],
                second["detection_marginals"]["adjacent_association_edges"],
            ),
            **_signature_comparison(
                first["count_signature_histogram"], second["count_signature_histogram"]
            ),
        }
        record["cross_run"] = comparison
        record["status"] = (
            "diagnostic_pair_below_gate"
            if comparison["stability_passed"] else "diagnostic_pair_failed"
        )
        _save(output, result)
        print(
            f"{regime.name}: {record['status']}; worst={max(comparison['metrics'].values()):.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("oracle", "large", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/wp11_tracklet_validation.json"))
    parser.add_argument("--wp7-output", type=Path, default=Path("results/wp7_convergence_audit.json"))
    parser.add_argument("--wp8-output", type=Path, default=Path("results/wp8_generalization.json"))
    args = parser.parse_args()
    result = _load_or_initialize(args.output, args.wp7_output, args.wp8_output)
    _save(args.output, result)
    if args.stage in ("oracle", "all"):
        run_oracle(result, args.output)
    if args.stage in ("large", "all"):
        run_large(result, args.output)


if __name__ == "__main__":
    main()
