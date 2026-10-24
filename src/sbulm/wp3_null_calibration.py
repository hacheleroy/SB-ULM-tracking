"""Calibrate explicit null-association SMC against enumerable ULM targets."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from .null_association import (
    NullAssociationMaternModel,
    NullAssociationPosterior,
    enumerate_null_association_posterior,
    null_and_clutter_marginals,
    reference_null_association,
)
from .null_smc import sample_null_association_posterior
from .simulation import (
    DetectionCorruptionConfig,
    corrupt_detections_with_misses_and_clutter,
    simulate_crossing,
)
from .smc import AnnealedSMCConfig
from .tsbm_reference import (
    compare_tsbm_association_samples,
    sliced_wasserstein_distance,
)

ROI_LOWER = (-0.04, -0.04)
ROI_UPPER = (0.04, 0.04)
REFERENCE_SPEED_MM_S = 6.84
GATE_NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class NullCalibrationRegime:
    name: str
    separation_um: float
    detection_probability: float
    expected_clutter_per_frame: float
    corruption_seed: int = 0


CALIBRATION_REGIMES = (
    NullCalibrationRegime("near_clean", 20.0, 0.95, 0.05),
    NullCalibrationRegime("missed_detections", 0.0, 0.70, 0.05),
    NullCalibrationRegime("false_positives", 0.0, 0.95, 0.75),
    NullCalibrationRegime("joint_corruption", 0.0, 0.70, 0.75),
)


@dataclass(frozen=True)
class _CalibrationCase:
    model: NullAssociationMaternModel
    exact: NullAssociationPosterior
    reference_association: np.ndarray
    coordinate_std: tuple[float, ...]
    length_scale: float
    localization_std: float


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    return (
        mean(values),
        0.0 if len(values) == 1 else stdev(values) / np.sqrt(len(values)),
    )


def _make_case(
    *,
    separation_um: float,
    detection_probability: float,
    expected_clutter_per_frame: float,
    data_seed: int,
    corruption_seed: int,
    max_clutter_per_frame: int = 2,
) -> _CalibrationCase:
    config = physical_crossing_config(
        n_frames=5,
        central_separation_um=separation_um,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=data_seed)
    detections = corrupt_detections_with_misses_and_clutter(
        clean,
        DetectionCorruptionConfig(
            detection_probability=detection_probability,
            expected_clutter_per_frame=expected_clutter_per_frame,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            max_clutter_per_frame=max_clutter_per_frame,
        ),
        seed=corruption_seed,
    )
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std,
        rms_speed=REFERENCE_SPEED_MM_S,
    )
    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=expected_clutter_per_frame,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    exact = enumerate_null_association_posterior(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=expected_clutter_per_frame,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    return _CalibrationCase(
        model=model,
        exact=exact,
        reference_association=reference_null_association(
            detections, n_particles=2
        ),
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        localization_std=config.localization_std,
    )


def _distribution_distances(
    exact: np.ndarray, empirical: np.ndarray
) -> tuple[float, float]:
    midpoint = 0.5 * (exact + empirical)

    def divergence(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0
        return float(
            np.sum(left[positive] * np.log(left[positive] / right[positive]))
        )

    total_variation = 0.5 * float(np.sum(np.abs(exact - empirical)))
    jensen_shannon = 0.5 * (
        divergence(exact, midpoint) + divergence(empirical, midpoint)
    )
    return total_variation, jensen_shannon


def _streaming_finite_sample_baseline(
    posterior: NullAssociationPosterior,
    *,
    n_samples: int,
    n_replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Compute finite-sample thresholds without an R-by-support count matrix."""

    rng = np.random.default_rng(seed)
    tv = np.empty(n_replicates, dtype=float)
    js = np.empty(n_replicates, dtype=float)
    recall = np.empty(n_replicates, dtype=float)
    mass = np.empty(n_replicates, dtype=float)
    order = np.argsort(posterior.probabilities)[::-1]
    credible = order[: posterior.credible_set_size(0.95)]
    for replicate in range(n_replicates):
        counts = rng.multinomial(n_samples, posterior.probabilities)
        empirical = counts / n_samples
        tv[replicate], js[replicate] = _distribution_distances(
            posterior.probabilities, empirical
        )
        observed = counts > 0
        recall[replicate] = float(np.mean(observed[credible]))
        mass[replicate] = float(np.sum(posterior.probabilities[observed]))
    tv_q = np.quantile(tv, [0.05, 0.5, 0.95])
    js_q = np.quantile(js, [0.05, 0.5, 0.95])
    recall_q = np.quantile(recall, [0.05, 0.5, 0.95])
    mass_q = np.quantile(mass, [0.05, 0.5, 0.95])
    return {
        "n_samples": n_samples,
        "n_replicates": n_replicates,
        "total_variation_median": float(tv_q[1]),
        "total_variation_q05": float(tv_q[0]),
        "total_variation_q95": float(tv_q[2]),
        "jensen_shannon_median": float(js_q[1]),
        "jensen_shannon_q05": float(js_q[0]),
        "jensen_shannon_q95": float(js_q[2]),
        "credible_set_recall_95_median": float(recall_q[1]),
        "credible_set_recall_95_q05": float(recall_q[0]),
        "credible_set_recall_95_q95": float(recall_q[2]),
        "exact_mass_covered_median": float(mass_q[1]),
        "exact_mass_covered_q05": float(mass_q[0]),
        "exact_mass_covered_q95": float(mass_q[2]),
    }


def _exact_joint_draws(
    case: _CalibrationCase,
    *,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        case.exact.probabilities.size,
        size=n_samples,
        p=case.exact.probabilities,
    )
    return case.model.sample_conditional_paths(
        case.exact.associations[indices], rng=rng
    )


def _reference_metrics(case: _CalibrationCase) -> dict[str, float | int]:
    matches = np.all(
        case.exact.associations == case.reference_association[None, ...],
        axis=(1, 2),
    )
    if np.count_nonzero(matches) != 1:
        raise RuntimeError("the simulated reference association is absent or duplicated")
    reference_index = int(np.flatnonzero(matches)[0])
    order = np.argsort(case.exact.probabilities)[::-1]
    rank = int(np.flatnonzero(order == reference_index)[0] + 1)
    credible = order[: case.exact.credible_set_size(0.95)]
    reference_ids = case.model.sequence.reference_ids
    if reference_ids is None:
        raise RuntimeError("calibration requires simulated reference IDs")
    realized_clutter = int(
        sum(np.count_nonzero(frame_ids < 0) for frame_ids in reference_ids)
    )
    realized_misses = int(np.count_nonzero(case.reference_association < 0))
    return {
        "reference_hypothesis_mass": float(
            case.exact.probabilities[reference_index]
        ),
        "reference_hypothesis_rank": rank,
        "reference_in_95pct_credible_set": int(reference_index in credible),
        "realized_missed_detections": realized_misses,
        "realized_false_positives": realized_clutter,
    }


def _association_and_event_metrics(
    case: _CalibrationCase,
    sampled_associations: np.ndarray,
    *,
    baseline_seed: int,
    baseline_replicates: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    comparison = compare_tsbm_association_samples(
        sampled_associations, case.exact
    )
    finite_baseline = _streaming_finite_sample_baseline(
        case.exact,
        n_samples=sampled_associations.shape[0],
        n_replicates=baseline_replicates,
        seed=baseline_seed,
    )
    exact_null, exact_clutter = null_and_clutter_marginals(
        case.exact.associations,
        case.model.sequence,
        probabilities=case.exact.probabilities,
    )
    sampled_null, sampled_clutter = null_and_clutter_marginals(
        sampled_associations,
        case.model.sequence,
    )
    max_null_error = float(np.max(np.abs(sampled_null - exact_null)))
    max_clutter_error = float(np.max(np.abs(sampled_clutter - exact_clutter)))
    thresholds = {
        "total_variation": max(
            0.05, 3.0 * float(finite_baseline["total_variation_q95"])
        ),
        "jensen_shannon": max(
            0.01, 3.0 * float(finite_baseline["jensen_shannon_q95"])
        ),
        "credible_set_recall_95": float(
            finite_baseline["credible_set_recall_95_q05"]
        ),
        "exact_mass_covered": max(
            0.95, float(finite_baseline["exact_mass_covered_q05"])
        ),
        "max_null_probability_error": 0.05,
        "max_expected_clutter_error": 0.05,
    }
    checks = {
        "total_variation": (
            comparison.total_variation_distance
            <= thresholds["total_variation"] + GATE_NUMERICAL_TOLERANCE
        ),
        "jensen_shannon": (
            comparison.jensen_shannon_divergence
            <= thresholds["jensen_shannon"] + GATE_NUMERICAL_TOLERANCE
        ),
        "credible_set_recall_95": (
            comparison.credible_set_recall_95 + GATE_NUMERICAL_TOLERANCE
            >= thresholds["credible_set_recall_95"]
        ),
        "exact_mass_covered": (
            comparison.exact_mass_covered_by_empirical_support
            + GATE_NUMERICAL_TOLERANCE
            >= thresholds["exact_mass_covered"]
        ),
        "max_null_probability_error": (
            max_null_error
            <= thresholds["max_null_probability_error"] + GATE_NUMERICAL_TOLERANCE
        ),
        "max_expected_clutter_error": (
            max_clutter_error
            <= thresholds["max_expected_clutter_error"] + GATE_NUMERICAL_TOLERANCE
        ),
    }
    metrics = {
        **comparison.to_dict(),
        "finite_sample_baseline": finite_baseline,
        "max_null_probability_error": max_null_error,
        "max_expected_clutter_error": max_clutter_error,
        "exact_null_probabilities": exact_null.tolist(),
        "sampled_null_probabilities": sampled_null.tolist(),
        "exact_expected_clutter": exact_clutter.tolist(),
        "sampled_expected_clutter": sampled_clutter.tolist(),
        "strict_all_credible_states_observed": (
            comparison.credible_set_recall_95 == 1.0
        ),
    }
    return metrics, thresholds, checks


def _run_sampler(
    case: _CalibrationCase,
    *,
    n_particles: int,
    target_ess_fraction: float,
    mh_sweeps: int,
    final_mh_sweeps: int,
    smc_seed: int,
) -> tuple[Any, float]:
    controls = AnnealedSMCConfig(
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        seed=smc_seed,
    )
    start = time.perf_counter()
    result = sample_null_association_posterior(
        case.model.sequence,
        n_particles=2,
        localization_std=case.localization_std,
        coordinate_std=case.coordinate_std,
        length_scale=case.length_scale,
        detection_probability=case.model.detection_probability,
        expected_clutter_per_frame=case.model.expected_clutter_per_frame,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
        config=controls,
    )
    return result, time.perf_counter() - start


def _evaluate_gate_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        regime,
        smc_seed,
        n_particles,
        target_ess_fraction,
        mh_sweeps,
        final_mh_sweeps,
        n_projections,
        baseline_replicates,
    ) = task
    case = _make_case(
        separation_um=regime.separation_um,
        detection_probability=regime.detection_probability,
        expected_clutter_per_frame=regime.expected_clutter_per_frame,
        data_seed=0,
        corruption_seed=regime.corruption_seed,
        max_clutter_per_frame=2,
    )
    result, runtime = _run_sampler(
        case,
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        smc_seed=smc_seed,
    )
    metrics, thresholds, checks = _association_and_event_metrics(
        case,
        result.associations,
        baseline_seed=10_000 + smc_seed,
        baseline_replicates=baseline_replicates,
    )
    exact_paths = _exact_joint_draws(
        case, n_samples=n_particles, seed=20_000 + smc_seed
    )
    second_exact_paths = _exact_joint_draws(
        case, n_samples=n_particles, seed=30_000 + smc_seed
    )
    path_distance = sliced_wasserstein_distance(
        result.latent_paths.reshape(n_particles, -1),
        exact_paths.reshape(n_particles, -1),
        n_projections=n_projections,
        seed=40_000 + smc_seed,
    )
    baseline_path_distance = sliced_wasserstein_distance(
        exact_paths.reshape(n_particles, -1),
        second_exact_paths.reshape(n_particles, -1),
        n_projections=n_projections,
        seed=40_000 + smc_seed,
    )
    thresholds["path_sliced_wasserstein_mm"] = max(
        0.001, 3.0 * baseline_path_distance
    )
    checks["path_sliced_wasserstein"] = (
        path_distance
        <= thresholds["path_sliced_wasserstein_mm"] + GATE_NUMERICAL_TOLERANCE
    )

    exact_log_evidence = float(
        logsumexp(case.model.log_joint_density(case.exact.associations))
    )
    evidence_error = result.log_evidence_estimate - exact_log_evidence
    thresholds["absolute_log_evidence_error"] = 0.15
    checks["absolute_log_evidence_error"] = (
        abs(evidence_error) <= 0.15 + GATE_NUMERICAL_TOLERANCE
    )
    reference = _reference_metrics(case)
    return {
        "regime": asdict(regime),
        "smc_seed": smc_seed,
        "runtime_seconds": runtime,
        "support_size": int(case.exact.probabilities.size),
        "detection_cardinalities": list(case.model.sequence.cardinalities),
        "association_and_events": metrics,
        "continuous": {
            "sliced_wasserstein_mm": path_distance,
            "finite_sample_sliced_wasserstein_mm": baseline_path_distance,
        },
        "normalizing_constant": {
            "smc_log_evidence": result.log_evidence_estimate,
            "exact_log_evidence": exact_log_evidence,
            "error": evidence_error,
        },
        "reference": reference,
        "smc": {
            "n_tempering_steps": result.n_tempering_steps,
            "mean_mh_acceptance_rate": mean(
                stage.mh_acceptance_rate for stage in result.stages
            ),
            "mean_frame_acceptance_rate": mean(
                stage.frame_acceptance_rate for stage in result.stages
            ),
            "mean_segment_acceptance_rate": mean(
                stage.segment_acceptance_rate for stage in result.stages
            ),
            "stages": [stage.to_dict() for stage in result.stages],
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_map_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        detection_probability,
        expected_clutter_per_frame,
        data_seed,
        n_particles,
        target_ess_fraction,
        mh_sweeps,
        final_mh_sweeps,
        baseline_replicates,
    ) = task
    case = _make_case(
        separation_um=0.0,
        detection_probability=detection_probability,
        expected_clutter_per_frame=expected_clutter_per_frame,
        data_seed=data_seed,
        corruption_seed=100_000 + data_seed,
        max_clutter_per_frame=1,
    )
    result, runtime = _run_sampler(
        case,
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        smc_seed=200_000 + data_seed,
    )
    metrics, thresholds, checks = _association_and_event_metrics(
        case,
        result.associations,
        baseline_seed=300_000 + data_seed,
        baseline_replicates=baseline_replicates,
    )
    exact_log_evidence = float(
        logsumexp(case.model.log_joint_density(case.exact.associations))
    )
    evidence_error = result.log_evidence_estimate - exact_log_evidence
    thresholds["absolute_log_evidence_error"] = 0.15
    checks["absolute_log_evidence_error"] = (
        abs(evidence_error) <= 0.15 + GATE_NUMERICAL_TOLERANCE
    )
    reference = _reference_metrics(case)
    return {
        "detection_probability": detection_probability,
        "expected_clutter_per_frame": expected_clutter_per_frame,
        "data_seed": data_seed,
        "support_size": int(case.exact.probabilities.size),
        "realized_missed_detections": reference["realized_missed_detections"],
        "realized_false_positives": reference["realized_false_positives"],
        "total_variation_distance": metrics["total_variation_distance"],
        "jensen_shannon_divergence": metrics["jensen_shannon_divergence"],
        "credible_set_recall_95": metrics["credible_set_recall_95"],
        "exact_mass_covered": metrics[
            "exact_mass_covered_by_empirical_support"
        ],
        "max_null_probability_error": metrics["max_null_probability_error"],
        "max_expected_clutter_error": metrics["max_expected_clutter_error"],
        "absolute_log_evidence_error": abs(evidence_error),
        "n_tempering_steps": result.n_tempering_steps,
        "mean_mh_acceptance_rate": mean(
            stage.mh_acceptance_rate for stage in result.stages
        ),
        "runtime_seconds": runtime,
        "passed": all(checks.values()),
        "strict_all_credible_states_observed": metrics[
            "strict_all_credible_states_observed"
        ],
        "thresholds": thresholds,
    }


def _run_tasks(function: Any, tasks: list[Any], workers: int) -> list[Any]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, tasks, chunksize=1))


def _aggregate_gate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for regime in CALIBRATION_REGIMES:
        selected = [item for item in runs if item["regime"]["name"] == regime.name]
        metrics = {
            "total_variation": [
                item["association_and_events"]["total_variation_distance"]
                for item in selected
            ],
            "max_null_probability_error": [
                item["association_and_events"]["max_null_probability_error"]
                for item in selected
            ],
            "max_expected_clutter_error": [
                item["association_and_events"]["max_expected_clutter_error"]
                for item in selected
            ],
            "path_sliced_wasserstein_um": [
                1_000.0 * item["continuous"]["sliced_wasserstein_mm"]
                for item in selected
            ],
            "absolute_log_evidence_error": [
                abs(item["normalizing_constant"]["error"]) for item in selected
            ],
            "runtime_seconds": [item["runtime_seconds"] for item in selected],
        }
        summary: dict[str, Any] = {
            "regime": regime.name,
            "n_runs": len(selected),
            "support_size": selected[0]["support_size"],
            "realized_missed_detections": selected[0]["reference"][
                "realized_missed_detections"
            ],
            "realized_false_positives": selected[0]["reference"][
                "realized_false_positives"
            ],
            "all_runs_passed": all(item["passed"] for item in selected),
        }
        for metric, values in metrics.items():
            average, standard_error = _mean_and_standard_error(values)
            summary[f"mean_{metric}"] = average
            summary[f"se_{metric}"] = standard_error
        summaries.append(summary)
    return summaries


def _aggregate_map_runs(
    runs: list[dict[str, Any]],
    detection_probabilities: tuple[float, ...],
    clutter_rates: tuple[float, ...],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metrics = (
        "total_variation_distance",
        "jensen_shannon_divergence",
        "exact_mass_covered",
        "max_null_probability_error",
        "max_expected_clutter_error",
        "absolute_log_evidence_error",
        "n_tempering_steps",
        "mean_mh_acceptance_rate",
        "runtime_seconds",
        "support_size",
        "realized_missed_detections",
        "realized_false_positives",
    )
    arrays = {
        metric: np.empty(
            (len(clutter_rates), len(detection_probabilities)), dtype=float
        )
        for metric in metrics
    }
    arrays["pass_fraction"] = np.empty_like(arrays[metrics[0]])
    rows: list[dict[str, Any]] = []
    for probability_index, probability in enumerate(detection_probabilities):
        for clutter_index, clutter_rate in enumerate(clutter_rates):
            selected = [
                item
                for item in runs
                if item["detection_probability"] == probability
                and item["expected_clutter_per_frame"] == clutter_rate
            ]
            row: dict[str, Any] = {
                "detection_probability": probability,
                "expected_clutter_per_frame": clutter_rate,
                "n_data_seeds": len(selected),
            }
            for metric in metrics:
                values = [float(item[metric]) for item in selected]
                average, standard_error = _mean_and_standard_error(values)
                arrays[metric][clutter_index, probability_index] = average
                row[f"mean_{metric}"] = average
                row[f"se_{metric}"] = standard_error
            pass_fraction = mean(float(item["passed"]) for item in selected)
            arrays["pass_fraction"][clutter_index, probability_index] = pass_fraction
            row["pass_fraction"] = pass_fraction
            rows.append(row)
    return rows, arrays


def _plot_gate_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    labels = [item["regime"].replace("_", " ") for item in summaries]
    indices = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    panels = (
        ("mean_total_variation", "Association TV", 0.05),
        ("mean_max_null_probability_error", "Max null-probability error", 0.05),
        ("mean_max_expected_clutter_error", "Max expected-clutter error", 0.05),
    )
    colors = [
        "#238b45" if item["all_runs_passed"] else "#cb181d" for item in summaries
    ]
    for axis, (metric, title, reference) in zip(axes, panels, strict=True):
        axis.bar(indices, [item[metric] for item in summaries], color=colors)
        axis.axhline(reference, linestyle="--", color="black", linewidth=1)
        axis.set_xticks(indices, labels, rotation=22)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("WP3 explicit null moves: exact-target calibration gate")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _plot_map(
    arrays: dict[str, np.ndarray],
    detection_probabilities: tuple[float, ...],
    clutter_rates: tuple[float, ...],
    output_path: Path,
) -> None:
    panels = (
        ("total_variation_distance", "Association TV", 0.0, None, "magma"),
        (
            "max_null_probability_error",
            "Max null-probability error",
            0.0,
            None,
            "magma",
        ),
        (
            "max_expected_clutter_error",
            "Max expected-clutter error",
            0.0,
            None,
            "magma",
        ),
        ("exact_mass_covered", "Exact mass covered", 0.0, 1.0, "viridis"),
        ("n_tempering_steps", "Tempering steps", 0.0, None, "cividis"),
        ("pass_fraction", "All-check pass fraction", 0.0, 1.0, "viridis"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for axis, (metric, title, lower, upper, color_map) in zip(
        axes.flat, panels, strict=True
    ):
        image = axis.imshow(
            arrays[metric],
            origin="lower",
            aspect="auto",
            vmin=lower,
            vmax=upper,
            cmap=color_map,
        )
        axis.set_title(title)
        axis.set_xticks(
            range(len(detection_probabilities)),
            [f"{value:.2f}" for value in detection_probabilities],
        )
        axis.set_yticks(
            range(len(clutter_rates)), [f"{value:.2f}" for value in clutter_rates]
        )
        axis.set_xlabel("detection probability")
        axis.set_ylabel("expected clutter / frame")
        figure.colorbar(image, ax=axis, shrink=0.88)
    figure.suptitle("Explicit null-association SMC across miss/clutter regimes")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_wp3_null_calibration(
    *,
    output_dir: Path,
    n_particles: int = 8_192,
    n_smc_repeats: int = 3,
    map_n_particles: int = 4_096,
    map_n_data_seeds: int = 4,
    target_ess_fraction: float = 0.80,
    mh_sweeps: int = 10,
    final_mh_sweeps: int = 10,
    n_projections: int = 128,
    baseline_replicates: int = 300,
    map_baseline_replicates: int = 100,
    detection_probabilities: tuple[float, ...] = (0.65, 0.80, 0.95),
    clutter_rates: tuple[float, ...] = (0.05, 0.35, 0.75),
    workers: int = 1,
) -> dict[str, Any]:
    """Run the four-regime gate and a miss/clutter calibration map."""

    positive_counts = (
        n_particles,
        n_smc_repeats,
        map_n_particles,
        map_n_data_seeds,
        mh_sweeps,
        n_projections,
        baseline_replicates,
        map_baseline_replicates,
        workers,
    )
    if any(value < 1 for value in positive_counts):
        raise ValueError("particle, repeat, sweep, projection, baseline, and worker counts must be positive")
    if final_mh_sweeps < 0:
        raise ValueError("final_mh_sweeps must be non-negative")
    if not detection_probabilities or not all(
        0 < value < 1 for value in detection_probabilities
    ):
        raise ValueError("detection probabilities must be non-empty and lie in (0, 1)")
    if not clutter_rates or min(clutter_rates) <= 0:
        raise ValueError("clutter rates must be non-empty and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    gate_tasks = [
        (
            regime,
            smc_seed,
            n_particles,
            target_ess_fraction,
            mh_sweeps,
            final_mh_sweeps,
            n_projections,
            baseline_replicates,
        )
        for regime in CALIBRATION_REGIMES
        for smc_seed in range(n_smc_repeats)
    ]
    gate_runs = _run_tasks(_evaluate_gate_task, gate_tasks, workers)
    gate_summary = _aggregate_gate_runs(gate_runs)

    map_tasks = [
        (
            float(probability),
            float(clutter_rate),
            data_seed,
            map_n_particles,
            target_ess_fraction,
            mh_sweeps,
            final_mh_sweeps,
            map_baseline_replicates,
        )
        for probability in detection_probabilities
        for clutter_rate in clutter_rates
        for data_seed in range(map_n_data_seeds)
    ]
    map_runs = _run_tasks(_evaluate_map_task, map_tasks, workers)
    map_rows, map_arrays = _aggregate_map_runs(
        map_runs, detection_probabilities, clutter_rates
    )
    with (output_dir / "wp3_null_calibration_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)

    first_case = _make_case(
        separation_um=0.0,
        detection_probability=0.80,
        expected_clutter_per_frame=0.35,
        data_seed=0,
        corruption_seed=0,
        max_clutter_per_frame=2,
    )
    result: dict[str, Any] = {
        "experiment": "wp3_explicit_null_association_smc_calibration",
        "method": {
            "latent_cardinality": "two persistent labelled bubbles (fixed)",
            "association_state": (
                "per-bubble detection index or -1; non-null indices are injective"
            ),
            "unassigned_detections": "uniform Poisson clutter",
            "base_distribution": (
                "normalized frame-wise detection/miss/clutter count density"
            ),
            "mutation": (
                "reversible whole-frame partial-injection replacements plus "
                "contiguous label swaps"
            ),
            "anchor": "complete clutter-free first frame fixes global labels",
            "uses_exact_enumeration_during_inference": False,
        },
        "observation_model": {
            "localization_noise_std_um": 5.0,
            "roi_lower_mm": list(ROI_LOWER),
            "roi_upper_mm": list(ROI_UPPER),
            "clutter_spatial_model": "uniform over ROI",
            "gate_simulator_clutter_cap": 2,
            "map_simulator_clutter_cap": 1,
            "poisson_set_density_includes_factorial": False,
        },
        "prior": {
            "family": "Matern-3/2",
            "rms_speed_mm_s": REFERENCE_SPEED_MM_S,
            "coordinate_std_um": [
                1_000.0 * value for value in first_case.coordinate_std
            ],
            "length_scale_ms": 1_000.0 * first_case.length_scale,
        },
        "configuration": {
            "n_particles": n_particles,
            "n_smc_repeats": n_smc_repeats,
            "map_n_particles": map_n_particles,
            "map_n_data_seeds": map_n_data_seeds,
            "target_ess_fraction": target_ess_fraction,
            "mh_sweeps": mh_sweeps,
            "final_mh_sweeps": final_mh_sweeps,
            "n_projections": n_projections,
            "baseline_replicates": baseline_replicates,
            "map_baseline_replicates": map_baseline_replicates,
            "detection_probabilities": list(detection_probabilities),
            "clutter_rates": list(clutter_rates),
        },
        "gate_definition": {
            "association_tv": "at most max(0.05, 3 * exact-draw MC q95)",
            "association_js": "at most max(0.01, 3 * exact-draw MC q95)",
            "credible_set_recall_95": (
                "at least the fifth percentile of equal-size exact draws"
            ),
            "exact_mass_covered": (
                "at least max(0.95, fifth percentile of equal-size exact draws)"
            ),
            "max_null_probability_error": 0.05,
            "max_expected_clutter_error": 0.05,
            "path_sliced_wasserstein": (
                "at most max(1 um, 3 * independent exact-draw baseline)"
            ),
            "absolute_log_evidence_error": 0.15,
            "aggregate_rule": "every repeat in every frozen regime must pass",
        },
        "gate_runs": gate_runs,
        "gate_summary": gate_summary,
        "gate_passed": all(item["all_runs_passed"] for item in gate_summary),
        "map": {
            "n_evaluations": len(map_runs),
            "all_checks_pass_fraction": mean(
                float(item["passed"]) for item in map_runs
            ),
            "max_total_variation_distance": max(
                float(item["total_variation_distance"]) for item in map_runs
            ),
            "max_null_probability_error": max(
                float(item["max_null_probability_error"]) for item in map_runs
            ),
            "max_expected_clutter_error": max(
                float(item["max_expected_clutter_error"]) for item in map_runs
            ),
            "min_exact_mass_covered": min(
                float(item["exact_mass_covered"]) for item in map_runs
            ),
            "max_absolute_log_evidence_error": max(
                float(item["absolute_log_evidence_error"]) for item in map_runs
            ),
            "max_support_size": max(int(item["support_size"]) for item in map_runs),
        },
        "map_runs": map_runs,
    }
    with (output_dir / "wp3_null_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_gate_summary(gate_summary, output_dir / "wp3_null_calibration.png")
    _plot_map(
        map_arrays,
        detection_probabilities,
        clutter_rates,
        output_dir / "wp3_null_calibration_map.png",
    )
    return result


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-particles", type=int, default=8_192)
    parser.add_argument("--n-smc-repeats", type=int, default=3)
    parser.add_argument("--map-n-particles", type=int, default=4_096)
    parser.add_argument("--map-n-data-seeds", type=int, default=4)
    parser.add_argument("--target-ess-fraction", type=float, default=0.80)
    parser.add_argument("--mh-sweeps", type=int, default=10)
    parser.add_argument("--final-mh-sweeps", type=int, default=10)
    parser.add_argument("--n-projections", type=int, default=128)
    parser.add_argument("--baseline-replicates", type=int, default=300)
    parser.add_argument("--map-baseline-replicates", type=int, default=100)
    parser.add_argument(
        "--detection-probabilities",
        type=_parse_float_list,
        default=(0.65, 0.80, 0.95),
    )
    parser.add_argument(
        "--clutter-rates",
        type=_parse_float_list,
        default=(0.05, 0.35, 0.75),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_wp3_null_calibration(
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        n_smc_repeats=args.n_smc_repeats,
        map_n_particles=args.map_n_particles,
        map_n_data_seeds=args.map_n_data_seeds,
        target_ess_fraction=args.target_ess_fraction,
        mh_sweeps=args.mh_sweeps,
        final_mh_sweeps=args.final_mh_sweeps,
        n_projections=args.n_projections,
        baseline_replicates=args.baseline_replicates,
        map_baseline_replicates=args.map_baseline_replicates,
        detection_probabilities=args.detection_probabilities,
        clutter_rates=args.clutter_rates,
        workers=args.workers,
    )
    print(json.dumps({"gate_passed": result["gate_passed"], "map": result["map"]}, indent=2))


if __name__ == "__main__":
    main()
