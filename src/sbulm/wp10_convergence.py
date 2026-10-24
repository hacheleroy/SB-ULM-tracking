"""Frozen particle-count and rejuvenation-budget check on the failed WP8 data.

The observations, prior, paired move mixture, seeds, and seven summary metrics
are unchanged. This two-seed screen is not an eight-frame exact oracle or a
six-run stability claim.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .existence_smc import BirthDeathSMCResult, sample_slot_existence_posterior
from .wp5_scaling_calibration import _controls, _sample_case, _ScalingCase
from .wp7_convergence_audit import (
    STABILITY_THRESHOLD,
    _case_metadata,
    _compare_large,
    _large_run_summary,
    _save,
)
from .wp8_generalization import REGIMES, make_case

BASE = REGIMES[1]
BASE_HASH = "afb3abbe5c56fc10162e69c35a2ead20ce6fe55c4cfab5f129363e0dd6a35bae"
SEEDS = (100, 200)
BASELINE_CLUTTER_DISCREPANCY = 0.2012939453125
BASELINE_ALIVE_COUNT_TV = 0.151123046875


@dataclass(frozen=True)
class ConvergenceArm:
    name: str
    n_particles: int
    mh_sweeps: int
    final_mh_sweeps: int


ARMS = (
    ConvergenceArm("double_particles", 16_384, 8, 8),
    ConvergenceArm("double_rejuvenation", 8_192, 16, 16),
)


def _sample_controlled(
    case: _ScalingCase, arm: ConvergenceArm, seed: int
) -> tuple[BirthDeathSMCResult, float]:
    if arm.mh_sweeps == 8 and arm.final_mh_sweeps == 8:
        return _sample_case(case, method="paired", n_particles=arm.n_particles, seed=seed)
    controls = _controls(
        "paired",
        n_particles=arm.n_particles,
        seed=seed,
        final_mh_sweeps=arm.final_mh_sweeps,
    )
    controls = replace(controls, mh_sweeps=arm.mh_sweeps)
    start = time.perf_counter()
    sampled = sample_slot_existence_posterior(
        case.model.sequence,
        n_slots=case.model.n_slots,
        localization_std=case.localization_std,
        coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=case.model.roi_lower,
        roi_upper=case.model.roi_upper,
        birth_death_prior=case.model.birth_death_prior,
        config=controls,
    )
    return sampled, time.perf_counter() - start


def _load_or_initialize(output: Path, wp8_output: Path) -> dict[str, Any]:
    case = make_case(BASE)
    metadata = _case_metadata(case)
    if metadata["detections_sha256"] != BASE_HASH:
        raise RuntimeError("the WP8 hard-case observations changed")
    wp8 = json.loads(wp8_output.read_text(encoding="utf-8"))
    original = wp8["results"][BASE.name]
    if (
        wp8["cases"][BASE.name]["model_and_detections"] != metadata
        or wp8["protocol"]["n_particles"] != 8_192
        or original["status"] != "failed_early"
        or original["cross_run"]["seeds"] != list(SEEDS)
        or original["cross_run"]["metrics"]["maximum_clutter_marginal_difference"]
        != BASELINE_CLUTTER_DISCREPANCY
        or original["cross_run"]["metrics"]["maximum_alive_count_tv"] != BASELINE_ALIVE_COUNT_TV
    ):
        raise ValueError("WP8 baseline is not the frozen failed two-seed case")
    baseline = {
        "n_particles": 8_192,
        "seeds": list(SEEDS),
        "cross_run": original["cross_run"],
        "run_diagnostics": {
            str(seed): {
                key: original["runs"][str(seed)][key]
                for key in (
                    "n_tempering_steps",
                    "runtime_seconds",
                    "minimum_stage_ess_fraction",
                    "mean_unique_parent_fraction",
                    "final_unique_state_fraction",
                    "mean_paired_acceptance_rate",
                    "mean_paired_state_change_rate",
                )
            }
            for seed in SEEDS
        },
    }
    protocol = {
        "sampler": "WP6 paired move mixture, unchanged target and ESS target 0.80",
        "arms": [asdict(arm) for arm in ARMS],
        "seeds_in_order": list(SEEDS),
        "stability_threshold": STABILITY_THRESHOLD,
        "decision": "two-seed diagnostic only; below-gate is not a six-run pass",
        "comparison": "seven inherited WP8 pairwise event and count metrics",
    }
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if (
            result["protocol"] != protocol
            or result["case"] != metadata
            or result["baseline"] != baseline
        ):
            raise ValueError("existing checkpoint belongs to a different frozen protocol")
        return result
    return {
        "experiment": "wp10_particle_and_rejuvenation_convergence",
        "protocol": protocol,
        "case": metadata,
        "baseline": baseline,
        "results": {
            arm.name: {"status": "in_progress", "runs": {}, "cross_run": None} for arm in ARMS
        },
    }


def run_convergence(*, output: Path, wp8_output: Path) -> dict[str, Any]:
    result = _load_or_initialize(output, wp8_output)
    _save(output, result)
    case = make_case(BASE)
    for arm in ARMS:
        record = result["results"][arm.name]
        if record["status"] != "in_progress":
            print(f"{arm.name}: reused {record['status']}", flush=True)
            continue
        for seed in SEEDS:
            key = str(seed)
            if key not in record["runs"]:
                sampled, runtime = _sample_controlled(case, arm, seed)
                record["runs"][key] = _large_run_summary(case, sampled, runtime, seed)
                _save(output, result)
                print(
                    f"{arm.name} seed {seed}: {runtime:.1f}s, {sampled.n_tempering_steps} stages",
                    flush=True,
                )
        comparison = _compare_large([record["runs"][str(seed)] for seed in SEEDS])
        record["cross_run"] = comparison
        record["status"] = (
            "diagnostic_pair_below_gate"
            if comparison["stability_passed"]
            else "diagnostic_pair_failed"
        )
        _save(output, result)
        print(
            f"{arm.name}: {record['status']}, worst={max(comparison['metrics'].values()):.4f}",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wp10_convergence.json"))
    parser.add_argument("--wp8-output", type=Path, default=Path("results/wp8_generalization.json"))
    args = parser.parse_args()
    result = run_convergence(output=args.output, wp8_output=args.wp8_output)
    print({name: record["status"] for name, record in result["results"].items()})


if __name__ == "__main__":
    main()
