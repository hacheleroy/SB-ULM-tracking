"""Compare likelihood-informed paired moves with the frozen WP5 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .wp5_scaling_calibration import (
    EXACT_REGIMES,
    _run_exact_regime,
    _run_large_stability_case,
)

METHODS = ("blocked", "paired")
STABILITY_METRICS = (
    "maximum_existence_marginal_difference",
    "maximum_missed_marginal_difference",
    "maximum_clutter_marginal_difference",
    "maximum_birth_marginal_difference",
    "maximum_death_marginal_difference",
    "maximum_alive_count_tv",
    "maximum_assigned_count_tv",
)


def _plot(result: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(STABILITY_METRICS))
    width = 0.38
    labels = (
        "existence",
        "missed",
        "clutter",
        "birth",
        "death",
        "alive count",
        "assigned count",
    )
    for index, method in enumerate(METHODS):
        case = result["large_cases"][method]
        values = [case["cross_run_metrics"][name] for name in STABILITY_METRICS]
        axes[0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=method,
            color="#bdbdbd" if method == "blocked" else "#2166ac",
        )
    axes[0].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_title("Four slots, eight frames: independent-run disagreement")
    axes[0].set_ylabel("Maximum pairwise difference or TV")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    exact = result["exact_cases"]
    case_labels = [case["regime"]["name"].replace("three_slots_", "") for case in exact]
    for index, method in enumerate(METHODS):
        values = [
            mean(
                run["maximum_event_error"]
                for run in case["runs"]
                if run["method"] == method
            )
            for case in exact
        ]
        axes[1].bar(
            np.arange(len(exact)) + (index - 0.5) * width,
            values,
            width,
            label=method,
            color="#bdbdbd" if method == "blocked" else "#2166ac",
        )
    axes[1].axhline(0.06, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(len(exact)), case_labels)
    axes[1].set_title("Three-slot exact-oracle event errors")
    axes[1].set_ylabel("Maximum absolute marginal error")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_wp6_paired_scaling(
    *,
    output_dir: Path,
    n_particles: int = 2_048,
    n_repeats: int = 3,
    baseline_replicates: int = 100,
) -> dict[str, Any]:
    """Run paired/control comparisons without overwriting the WP5 record."""

    if n_particles < 2 or n_repeats < 2 or baseline_replicates < 1:
        raise ValueError("invalid particle, repeat, or baseline count")
    output_dir.mkdir(parents=True, exist_ok=True)
    exact = [
        _run_exact_regime(
            regime,
            n_particles=n_particles,
            n_repeats=n_repeats,
            baseline_replicates=baseline_replicates,
            methods=METHODS,
        )
        for regime in EXACT_REGIMES
    ]
    large = {
        method: _run_large_stability_case(
            n_particles=n_particles,
            n_repeats=n_repeats,
            method=method,
        )
        for method in METHODS
    }
    exact_gates = {
        method: all(case["method_gates"][method] for case in exact)
        for method in METHODS
    }
    result: dict[str, Any] = {
        "experiment": "wp6_local_likelihood_informed_pair_moves",
        "method": {
            "proposal": (
                "uniform local conditional support for a pair of slots in one "
                "or two frames; finite candidate sets are symmetric on reversal"
            ),
            "selection": (
                "multiple-try weighting by the exact existence/count base "
                "and tempered collapsed Matern localization likelihood"
            ),
            "correction": (
                "independent multiple-try Metropolis ratio of forward and "
                "reverse weight sums; reverse set includes the current state"
            ),
            "mixture": {
                "whole_frame_association": 0.20,
                "interval_replacement": 0.35,
                "global_slot_swap": 0.05,
                "conditional_base_blocked_slot": 0.15,
                "local_paired_multiple_try": 0.25,
                "paired_candidates": 2,
            },
            "oracle_used_during_inference": False,
        },
        "configuration": {
            "n_particles": n_particles,
            "n_repeats": n_repeats,
            "baseline_replicates": baseline_replicates,
            "stability_threshold": 0.10,
            "seeds": list(range(100, 100 + n_repeats)),
        },
        "exact_cases": exact,
        "exact_gates": exact_gates,
        "large_cases": large,
        "large_stability": {method: large[method]["stability_passed"] for method in METHODS},
        "paired_gate_passed": exact_gates["paired"] and large["paired"]["stability_passed"],
        "interpretation_limit": (
            "four-slot cross-run agreement is necessary but not an exact "
            "posterior-calibration proof; retain the three-slot exact ceiling"
        ),
    }
    with (output_dir / "wp6_paired_scaling.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    _plot(result, output_dir / "wp6_paired_scaling.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-particles", type=int, default=2_048)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--baseline-replicates", type=int, default=100)
    parser.add_argument("--heldout-only", action="store_true")
    parser.add_argument("--heldout-seed-start", type=int, default=200)
    args = parser.parse_args()
    if args.heldout_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        heldout = _run_large_stability_case(
            n_particles=args.n_particles,
            n_repeats=args.n_repeats,
            method="paired",
            sampling_seed_start=args.heldout_seed_start,
        )
        with (args.output_dir / "wp6_paired_heldout.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(heldout, stream, indent=2)
        print(json.dumps({"heldout_stability_passed": heldout["stability_passed"]}))
        return
    result = run_wp6_paired_scaling(
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        n_repeats=args.n_repeats,
        baseline_replicates=args.baseline_replicates,
    )
    print(
        json.dumps(
            {
                "exact_gates": result["exact_gates"],
                "large_stability": result["large_stability"],
                "paired_gate_passed": result["paired_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
