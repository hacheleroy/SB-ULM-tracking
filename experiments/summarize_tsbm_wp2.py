#!/usr/bin/env python3
"""Consolidate TSBM calibration runs and apply the fixed WP2 acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sbulm.calibration import physical_crossing_config
from sbulm.data import DetectionSequence
from sbulm.simulation import simulate_crossing
from sbulm.tsbm_reference import (
    collapse_global_label_probabilities,
    collapse_global_label_symmetry,
    compare_association_probabilities,
    enumerate_tsbm_brownian_localization_posterior,
    rao_blackwellized_association_probabilities,
)


def parse_run(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("runs must use LABEL=SUMMARY_JSON") from error
    summary_path = Path(path)
    if not label or not summary_path.is_file():
        raise argparse.ArgumentTypeError(f"invalid run specification: {value}")
    return label, summary_path


def sampled_endpoint_joint(samples_path: Path) -> list[list[float]]:
    with np.load(samples_path) as archive:
        associations = archive["sampled_associations"]
        permutations = np.unique(archive["exact_associations"][:, 0], axis=0)
    joint = np.asarray(
        [
            [
                np.mean(
                    np.all(associations[:, 0] == initial, axis=1)
                    & np.all(associations[:, -1] == final, axis=1)
                )
                for final in permutations
            ]
            for initial in permutations
        ]
    )
    return joint.tolist()


def rao_blackwellized_comparisons(
    summary: dict[str, Any], samples_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute soft association weights for legacy and current summaries."""

    regime = summary["regime"]
    config = physical_crossing_config(
        central_separation_um=regime["separation_um"],
        localization_std_um=regime["localization_std_um"],
    )
    _, physical_sequence = simulate_crossing(config, seed=regime["simulation_seed"])
    sequence = DetectionSequence(
        times=np.linspace(0.0, 1.0, physical_sequence.n_frames),
        detections=physical_sequence.detections,
        reference_ids=physical_sequence.reference_ids,
    )
    exact = enumerate_tsbm_brownian_localization_posterior(
        sequence,
        localization_std=config.localization_std,
        diffusion_std=summary["physical_design"][
            "brownian_diffusion_std_mm_per_sqrt_normalized_time"
        ],
    )
    with np.load(samples_path) as archive:
        sampled_paths = archive["sampled_paths"]
        np.testing.assert_allclose(
            archive["exact_probabilities"], exact.probabilities, rtol=1e-10, atol=1e-12
        )
    estimated = rao_blackwellized_association_probabilities(
        sampled_paths,
        sequence,
        exact.associations,
        localization_std=config.localization_std,
    )
    labelled = compare_association_probabilities(
        estimated, exact, n_path_samples=sampled_paths.shape[0]
    )
    canonical_exact = collapse_global_label_symmetry(exact)
    canonical_support, canonical_estimated = collapse_global_label_probabilities(
        exact.associations, estimated
    )
    if not np.array_equal(canonical_support, canonical_exact.associations):
        raise RuntimeError("canonical association supports are not aligned")
    canonical = compare_association_probabilities(
        canonical_estimated,
        canonical_exact,
        n_path_samples=sampled_paths.shape[0],
    )
    return labelled.to_dict(), canonical.to_dict()


def portable_training_config(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain the run configuration without leaking machine-local paths."""

    training = dict(summary["training"])
    for key in (
        "resume_score_checkpoint",
        "resume_control_variate_checkpoint",
        "resume_optimizer_checkpoint",
    ):
        value = training.get(key)
        if not value:
            continue
        checkpoint = Path(value)
        source_run = checkpoint.parents[2].name if len(checkpoint.parents) >= 3 else None
        training[key] = {
            "source_run": source_run,
            "file": checkpoint.name,
        }
    return training


def consolidate(label: str, summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hard_canonical = summary["canonical_sampled_target"]
    rao_blackwellized, canonical = rao_blackwellized_comparisons(
        summary, summary_path.with_name("samples.npz")
    )
    baseline = summary["finite_sample_baseline"]["canonical"]
    continuous = summary["continuous_target"]
    exact_joint = np.asarray(summary["exact_target"]["endpoint_permutation_joint"])
    sample_joint = np.asarray(sampled_endpoint_joint(summary_path.with_name("samples.npz")))

    tv_threshold = max(0.05, 3.0 * baseline["total_variation_q95"])
    js_threshold = max(0.01, 3.0 * baseline["jensen_shannon_q95"])
    wasserstein_threshold = max(
        1.0, 3.0 * continuous["exact_finite_sample_baseline_um"]
    )
    endpoint_joint_tv = 0.5 * float(np.sum(np.abs(exact_joint - sample_joint)))
    checks = {
        "canonical_tv": canonical["total_variation_distance"] <= tv_threshold,
        "canonical_js": canonical["jensen_shannon_divergence"] <= js_threshold,
        "endpoint_joint_tv": endpoint_joint_tv <= 0.05,
        "continuous_sliced_wasserstein": (
            continuous["canonical_sliced_wasserstein_um"]
            <= wasserstein_threshold
        ),
        "credible_mode_recall": (
            hard_canonical.get("credible_set_recall_95", 0.0) == 1.0
        ),
        "support_mass": (
            hard_canonical.get("exact_mass_covered_by_empirical_support", 0.0)
            >= 0.95
        ),
    }
    return {
        "label": label,
        "regime": summary["regime"]["name"],
        "regime_parameters": summary["regime"],
        "upstream": summary["upstream"],
        "physical_design": summary["physical_design"],
        "training": portable_training_config(summary),
        "sampling": {
            **summary["sampling"],
            "direction": summary["sampling"].get("direction", "f"),
        },
        "sample_direction": summary["sampling"].get("direction", "f"),
        "training_seconds": summary["training"]["training_seconds"],
        "canonical_total_variation": canonical["total_variation_distance"],
        "canonical_jensen_shannon": canonical["jensen_shannon_divergence"],
        "rao_blackwellized_target": rao_blackwellized,
        "canonical_rao_blackwellized_target": canonical,
        "hard_projection_canonical_total_variation": hard_canonical[
            "total_variation_distance"
        ],
        "hard_projection_canonical_jensen_shannon": hard_canonical[
            "jensen_shannon_divergence"
        ],
        "hard_projection_credible_set_recall_95": hard_canonical.get(
            "credible_set_recall_95", 0.0
        ),
        "hard_projection_exact_mass_covered": hard_canonical.get(
            "exact_mass_covered_by_empirical_support", 0.0
        ),
        "canonical_tv_finite_sample_q95": baseline["total_variation_q95"],
        "canonical_js_finite_sample_q95": baseline["jensen_shannon_q95"],
        "canonical_sliced_wasserstein_um": continuous[
            "canonical_sliced_wasserstein_um"
        ],
        "sliced_wasserstein_finite_sample_um": continuous[
            "exact_finite_sample_baseline_um"
        ],
        "exact_endpoint_joint": exact_joint.tolist(),
        "sampled_endpoint_joint": sample_joint.tolist(),
        "endpoint_joint_total_variation": endpoint_joint_tv,
        "sampled_interior_residual_median_um": continuous[
            "sampled_observation_residuals"
        ]["interior_median_um"],
        "exact_interior_residual_median_um": continuous[
            "exact_observation_residuals"
        ]["interior_median_um"],
        "thresholds": {
            "canonical_total_variation": tv_threshold,
            "canonical_jensen_shannon": js_threshold,
            "endpoint_joint_total_variation": 0.05,
            "canonical_sliced_wasserstein_um": wasserstein_threshold,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def plot_runs(runs: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    colors = {"ambiguous": "#315b7d", "easy": "#b14f3b"}
    max_checkpoints = max(
        len([run for run in runs if run["regime"] == regime]) for regime in colors
    )
    for regime in ("ambiguous", "easy"):
        selected = [run for run in runs if run["regime"] == regime]
        if not selected:
            continue
        x = np.arange(len(selected))
        labels = [run["label"] for run in selected]
        axes[0].plot(
            x,
            [run["canonical_total_variation"] for run in selected],
            marker="o",
            color=colors[regime],
            label=regime.replace("_", " "),
        )
        axes[1].plot(
            x,
            [run["canonical_sliced_wasserstein_um"] for run in selected],
            marker="o",
            color=colors[regime],
            label=regime.replace("_", " "),
        )
        for axis in axes:
            for index, label in zip(x, labels, strict=True):
                axis.annotate(
                    label,
                    (index, axis.lines[-1].get_ydata()[index]),
                    xytext=(0, -17 if regime == "easy" else 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    rotation=22,
                )
    axes[0].set(
        title="Soft association calibration",
        ylabel="Rao–Blackwellized canonical TV",
    )
    axes[1].set(
        title="Continuous path calibration",
        ylabel="Sliced Wasserstein distance (µm)",
    )
    axes[0].axhline(
        max(run["thresholds"]["canonical_total_variation"] for run in runs),
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="acceptance limit",
    )
    axes[1].axhline(
        max(run["thresholds"]["canonical_sliced_wasserstein_um"] for run in runs),
        color="#333333",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].set_ylim(0.0, 1.08)
    axes[1].set_ylim(
        0.0,
        1.1 * max(run["canonical_sliced_wasserstein_um"] for run in runs),
    )
    for axis in axes:
        axis.set_xlabel("Training/evaluation checkpoint")
        axis.set_xticks([])
        axis.set_xlim(-0.3, max_checkpoints - 0.4)
        axis.margins(y=0.18)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [consolidate(label, path) for label, path in args.run]
    required_regimes = ("easy", "ambiguous")
    by_regime = {
        regime: min(
            (run for run in runs if run["regime"] == regime),
            key=lambda run: run["canonical_total_variation"],
        )["label"]
        for regime in sorted({run["regime"] for run in runs})
    }
    passed_by_regime = {
        regime: any(
            run["gate_passed"] for run in runs if run["regime"] == regime
        )
        for regime in required_regimes
    }
    output = {
        "experiment": "public_tsbm_wp2_calibration",
        "acceptance_gate": "all discrete, endpoint, continuous, and support checks",
        "required_regimes": list(required_regimes),
        "runs": runs,
        "best_canonical_tv_checkpoint_by_regime": by_regime,
        "gate_passed_by_regime": passed_by_regime,
        "gate_passed_any_run": any(run["gate_passed"] for run in runs),
        "gate_passed_all_required_regimes": all(passed_by_regime.values()),
        "decision": (
            "NO-GO for downstream ULM use: retain the public TSBM adapter as a "
            "calibration-tested negative baseline and do not add missed detections "
            "until an approximate sampler recovers the enumerable target."
        ),
    }
    (args.output_dir / "tsbm_wp2_calibration.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    plot_runs(runs, args.output_dir / "tsbm_wp2_calibration.png")


if __name__ == "__main__":
    main()
