"""Frozen exact-oracle then reserved-case conditional-GP tracklet audit.

The four-slot oracle is checked before either sampler sees the new eight-frame
realization. The new and WP6 paired mixtures use identical model, data,
particles, sweeps and seeds. Two-seed agreement is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .slot_existence import slot_event_marginals
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
from .wp11_tracklet_validation import (
    _count_signature_histogram,
    _edge_and_clutter_marginals,
    _maximum_nested_difference,
    _signature_comparison,
)

BRIDGE_HASH = "62d3aeeafff58eb47404f87291829004f5bd498d07ab1494afa86ba77115b8ef"
RESERVED = replace(
    REGIMES[1],
    name="cluttered_staggered_reserved_c",
    trajectory_seed=53,
    corruption_seed=29,
)
RESERVED_HASH = "bd8c6aed400ffd38c24621610e34b7baf9c74a8a9eca620b65a33b8e4006815c"
ORACLE_SEEDS = (400, 401, 402)
RESERVED_SEEDS = (310, 410)
ORACLE_PARTICLES = 2_048
RESERVED_PARTICLES = 8_192
ARMS = ("conditional_gp_nonself", "paired")
PROPOSAL_MIXTURE = {
    "association": 0.15,
    "interval": 0.30,
    "slot_swap": 0.05,
    "blocked_slot": 0.15,
    "paired_slot": 0.20,
    "conditional_gp_nonself_tracklet": 0.15,
}


def _load_or_initialize(output: Path, wp7_output: Path) -> dict[str, Any]:
    bridge_meta = _case_metadata(_make_case(bridge=True))
    reserved_meta = _case_metadata(make_case(RESERVED))
    if (
        bridge_meta["detections_sha256"] != BRIDGE_HASH
        or reserved_meta["detections_sha256"] != RESERVED_HASH
    ):
        raise RuntimeError("frozen oracle or reserved observations changed")
    wp7 = json.loads(wp7_output.read_text(encoding="utf-8"))
    if wp7["cases"]["bridge"] != bridge_meta:
        raise ValueError("WP7 oracle baseline metadata changed")
    thresholds = wp7["bridge"][str(ORACLE_PARTICLES)]["thresholds"]
    protocol = {
        "proposal": "GP prediction from fixed outside-window detections; non-self three-frame pair HMM",
        "proposal_mixture": PROPOSAL_MIXTURE,
        "oracle_seeds": list(ORACLE_SEEDS),
        "oracle_particles": ORACLE_PARTICLES,
        "oracle_thresholds": thresholds,
        "reserved_regime": json.loads(json.dumps(asdict(RESERVED))),
        "reserved_observation_hash": RESERVED_HASH,
        "reserved_seeds": list(RESERVED_SEEDS),
        "reserved_particles": RESERVED_PARTICLES,
        "arms_in_order": list(ARMS),
        "ess_target": 0.80,
        "sweeps_per_stage": 8,
        "extra_final_sweeps": 8,
        "seven_metric_gate": STABILITY_THRESHOLD,
        "interpretation": "two-seed diagnostic only; no exact eight-frame posterior",
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing["protocol"] != protocol
            or existing["bridge"]["metadata"] != bridge_meta
            or existing["reserved"]["metadata"] != reserved_meta
        ):
            raise ValueError("existing WP11b checkpoint has a different frozen protocol")
        return existing
    return {
        "experiment": "wp11b_gp_predictive_nonself_tracklets",
        "protocol": protocol,
        "bridge": {"metadata": bridge_meta, "status": "in_progress", "runs": {}},
        "reserved": {
            "metadata": reserved_meta,
            "arms": {
                arm: {"status": "in_progress", "runs": {}, "cross_run": None}
                for arm in ARMS
            },
        },
    }


def run_oracle(result: dict[str, Any], output: Path) -> None:
    case = _make_case(bridge=True)
    exact, _ = _exact_posterior(case.model)
    expected_events = slot_event_marginals(
        exact.associations, case.model.sequence, probabilities=exact.probabilities
    )
    expected_edges = _edge_and_clutter_marginals(
        exact.associations, case, exact.probabilities
    )
    thresholds = result["protocol"]["oracle_thresholds"]
    for seed in ORACLE_SEEDS:
        if str(seed) in result["bridge"]["runs"]:
            continue
        sampled, runtime = _sample_case(
            case, method=ARMS[0], n_particles=ORACLE_PARTICLES, seed=seed
        )
        comparison = compare_tsbm_association_samples(sampled.associations, exact)
        errors = _event_errors(
            expected_events,
            slot_event_marginals(sampled.associations, case.model.sequence),
        )
        edges = _edge_and_clutter_marginals(sampled.associations, case)
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
            "comparison": comparison.to_dict(),
            "maximum_event_error": max(errors.values()),
            "maximum_edge_error": _maximum_nested_difference(
                expected_edges["adjacent_association_edges"],
                edges["adjacent_association_edges"],
            ),
            "maximum_detection_clutter_error": _maximum_nested_difference(
                expected_edges["detection_clutter"], edges["detection_clutter"]
            ),
            "mean_tracklet_acceptance_rate": float(np.mean(
                [stage.tracklet_acceptance_rate for stage in sampled.stages]
            )),
            "mean_tracklet_state_change_rate": float(np.mean(
                [stage.tracklet_state_change_rate for stage in sampled.stages]
            )),
            "runtime_seconds": runtime,
            "checks": checks,
            "passed": all(checks.values()),
        }
        _save(output, result)
        print(f"oracle seed {seed}: {'PASS' if all(checks.values()) else 'FAIL'}, {runtime:.1f}s", flush=True)
    result["bridge"]["status"] = (
        "passed" if len(result["bridge"]["runs"]) == len(ORACLE_SEEDS)
        and all(run["passed"] for run in result["bridge"]["runs"].values())
        else "failed"
    )
    _save(output, result)


def run_reserved(result: dict[str, Any], output: Path) -> None:
    if result["bridge"]["status"] != "passed":
        raise RuntimeError("large-case inference is blocked by the oracle gate")
    case = make_case(RESERVED)
    for arm in ARMS:
        record = result["reserved"]["arms"][arm]
        for seed in RESERVED_SEEDS:
            if str(seed) in record["runs"]:
                continue
            sampled, runtime = _sample_case(
                case, method=arm, n_particles=RESERVED_PARTICLES, seed=seed
            )
            summary = _large_run_summary(case, sampled, runtime, seed)
            summary["detection_marginals"] = _edge_and_clutter_marginals(
                sampled.associations, case
            )
            summary["count_signature_histogram"] = _count_signature_histogram(
                sampled.associations
            )
            summary["mean_tracklet_acceptance_rate"] = float(np.mean(
                [stage.tracklet_acceptance_rate for stage in sampled.stages]
            ))
            summary["mean_tracklet_state_change_rate"] = float(np.mean(
                [stage.tracklet_state_change_rate for stage in sampled.stages]
            ))
            record["runs"][str(seed)] = summary
            _save(output, result)
            print(f"{arm} seed {seed}: {runtime:.1f}s, {sampled.n_tempering_steps} stages", flush=True)
        first, second = (record["runs"][str(seed)] for seed in RESERVED_SEEDS)
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
            "diagnostic_pair_below_gate" if comparison["stability_passed"]
            else "diagnostic_pair_failed"
        )
        _save(output, result)
        print(f"{arm}: {record['status']}, worst={max(comparison['metrics'].values()):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("oracle", "reserved", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/wp11b_conditional_validation.json"))
    parser.add_argument("--wp7-output", type=Path, default=Path("results/wp7_convergence_audit.json"))
    args = parser.parse_args()
    result = _load_or_initialize(args.output, args.wp7_output)
    _save(args.output, result)
    if args.stage in ("oracle", "all"):
        run_oracle(result, args.output)
    if args.stage in ("reserved", "all"):
        run_reserved(result, args.output)


if __name__ == "__main__":
    main()
