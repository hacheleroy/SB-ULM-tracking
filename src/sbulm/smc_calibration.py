"""Calibrate collapsed Matérn SMC against exact ULM association targets."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import factorial, log
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .calibration import physical_crossing_config
from .inference import AssociationPosterior
from .localization import (
    MaternLocalizationModel,
    enumerate_matern_localization_posterior,
)
from .physical_sweep import DEFAULT_NOISE_LEVELS_UM, DEFAULT_SEPARATIONS_UM
from .simulation import CrossingConfig, simulate_crossing
from .smc import AnnealedSMCConfig, sample_matern_localization_posterior
from .tsbm_reference import (
    compare_tsbm_association_samples,
    finite_sample_baseline,
    sliced_wasserstein_distance,
)
from .wp2_likelihood import REFERENCE_COORDINATE_STD, REFERENCE_LENGTH_SCALE


@dataclass(frozen=True)
class CalibrationRegime:
    name: str
    separation_um: float
    localization_std_um: float
    data_seed: int = 0


CALIBRATION_REGIMES = (
    CalibrationRegime("easy", 20.0, 2.5),
    CalibrationRegime("ambiguous", 0.0, 5.0),
    CalibrationRegime("high_noise", 0.0, 20.0),
)
GATE_NUMERICAL_TOLERANCE = 1e-12


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    return (
        mean(values),
        0.0 if len(values) == 1 else stdev(values) / np.sqrt(len(values)),
    )


def _exact_joint_draws(
    model: MaternLocalizationModel,
    posterior: AssociationPosterior,
    *,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        posterior.probabilities.size,
        size=n_samples,
        p=posterior.probabilities,
    )
    associations = posterior.associations[indices]
    paths = model.sample_conditional_paths(associations, rng=rng)
    return paths, associations


def _make_case(
    *, separation_um: float, localization_std_um: float, data_seed: int
) -> tuple[CrossingConfig, AssociationPosterior, MaternLocalizationModel]:
    config = physical_crossing_config(
        central_separation_um=separation_um,
        localization_std_um=localization_std_um,
    )
    _, detections = simulate_crossing(config, seed=data_seed)
    exact = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
    )
    model = MaternLocalizationModel(
        detections,
        localization_std=config.localization_std,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
    )
    return config, exact, model


def _association_metrics(
    *,
    exact: AssociationPosterior,
    sampled_associations: np.ndarray,
    baseline_seed: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    comparison = compare_tsbm_association_samples(sampled_associations, exact)
    finite_baseline = finite_sample_baseline(
        exact,
        n_samples=sampled_associations.shape[0],
        n_replicates=500,
        seed=baseline_seed,
    )
    thresholds = {
        "total_variation": max(0.05, 3.0 * finite_baseline.total_variation_q95),
        "jensen_shannon": max(0.01, 3.0 * finite_baseline.jensen_shannon_q95),
        "credible_set_recall_95": finite_baseline.credible_set_recall_95_q05,
        "exact_mass_covered": max(0.95, finite_baseline.exact_mass_covered_q05),
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
    }
    metrics = {
        **comparison.to_dict(),
        "finite_sample_baseline": finite_baseline.to_dict(),
        "strict_all_credible_states_observed": (
            comparison.credible_set_recall_95 == 1.0
        ),
    }
    return metrics, thresholds, checks


def _evaluate_gate_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        regime,
        smc_seed,
        n_particles,
        target_ess_fraction,
        mh_sweeps,
        final_mh_sweeps,
        n_projections,
    ) = task
    config, exact, model = _make_case(
        separation_um=regime.separation_um,
        localization_std_um=regime.localization_std_um,
        data_seed=regime.data_seed,
    )
    controls = AnnealedSMCConfig(
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        seed=smc_seed,
    )
    start = time.perf_counter()
    result = sample_matern_localization_posterior(
        model.sequence,
        localization_std=config.localization_std,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
        config=controls,
    )
    runtime = time.perf_counter() - start
    association, thresholds, checks = _association_metrics(
        exact=exact,
        sampled_associations=result.associations,
        baseline_seed=10_000 + smc_seed,
    )

    exact_paths, _ = _exact_joint_draws(
        model,
        exact,
        n_samples=n_particles,
        seed=20_000 + smc_seed,
    )
    second_exact_paths, _ = _exact_joint_draws(
        model,
        exact,
        n_samples=n_particles,
        seed=30_000 + smc_seed,
    )
    smc_sliced_wasserstein = sliced_wasserstein_distance(
        result.latent_paths.reshape(n_particles, -1),
        exact_paths.reshape(n_particles, -1),
        n_projections=n_projections,
        seed=40_000 + smc_seed,
    )
    baseline_sliced_wasserstein = sliced_wasserstein_distance(
        exact_paths.reshape(n_particles, -1),
        second_exact_paths.reshape(n_particles, -1),
        n_projections=n_projections,
        seed=40_000 + smc_seed,
    )
    path_threshold = max(0.001, 3.0 * baseline_sliced_wasserstein)
    thresholds["path_sliced_wasserstein_mm"] = path_threshold
    checks["path_sliced_wasserstein"] = (
        smc_sliced_wasserstein <= path_threshold + GATE_NUMERICAL_TOLERANCE
    )

    exact_log_evidence = float(
        logsumexp(model.log_marginal_density(exact.associations))
        - (model.sequence.n_frames - 1)
        * log(factorial(model.n_particles))
    )
    evidence_error = result.log_evidence_estimate - exact_log_evidence
    thresholds["absolute_log_evidence_error"] = 0.10
    checks["absolute_log_evidence_error"] = (
        abs(evidence_error) <= 0.10 + GATE_NUMERICAL_TOLERANCE
    )

    return {
        "regime": asdict(regime),
        "smc_seed": smc_seed,
        "runtime_seconds": runtime,
        "association": association,
        "continuous": {
            "sliced_wasserstein_mm": smc_sliced_wasserstein,
            "finite_sample_sliced_wasserstein_mm": baseline_sliced_wasserstein,
        },
        "normalizing_constant": {
            "smc_log_evidence": result.log_evidence_estimate,
            "exact_log_evidence": exact_log_evidence,
            "error": evidence_error,
        },
        "smc": {
            "n_tempering_steps": result.n_tempering_steps,
            "mean_mh_acceptance_rate": mean(
                stage.mh_acceptance_rate for stage in result.stages
            ),
            "stages": [stage.to_dict() for stage in result.stages],
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "strict_all_credible_states_observed": association[
            "strict_all_credible_states_observed"
        ],
    }


def _evaluate_map_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        data_seed,
        separation_um,
        localization_std_um,
        n_particles,
        target_ess_fraction,
        mh_sweeps,
        final_mh_sweeps,
    ) = task
    config, exact, model = _make_case(
        separation_um=separation_um,
        localization_std_um=localization_std_um,
        data_seed=data_seed,
    )
    controls = AnnealedSMCConfig(
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        seed=100_000 + data_seed,
    )
    start = time.perf_counter()
    result = sample_matern_localization_posterior(
        model.sequence,
        localization_std=config.localization_std,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
        config=controls,
    )
    runtime = time.perf_counter() - start
    association, thresholds, checks = _association_metrics(
        exact=exact,
        sampled_associations=result.associations,
        baseline_seed=200_000 + data_seed,
    )
    baseline = association["finite_sample_baseline"]
    return {
        "data_seed": data_seed,
        "central_separation_um": separation_um,
        "localization_std_um": localization_std_um,
        "total_variation_distance": association["total_variation_distance"],
        "jensen_shannon_divergence": association["jensen_shannon_divergence"],
        "finite_sample_tv_q95": baseline["total_variation_q95"],
        "finite_sample_js_q95": baseline["jensen_shannon_q95"],
        "credible_set_recall_95": association["credible_set_recall_95"],
        "exact_mass_covered": association[
            "exact_mass_covered_by_empirical_support"
        ],
        "n_tempering_steps": result.n_tempering_steps,
        "mean_mh_acceptance_rate": mean(
            stage.mh_acceptance_rate for stage in result.stages
        ),
        "runtime_seconds": runtime,
        "passed": all(checks.values()),
        "strict_all_credible_states_observed": association[
            "strict_all_credible_states_observed"
        ],
        "tv_threshold": thresholds["total_variation"],
        "js_threshold": thresholds["jensen_shannon"],
        "credible_set_recall_95_threshold": thresholds[
            "credible_set_recall_95"
        ],
        "exact_mass_covered_threshold": thresholds["exact_mass_covered"],
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
        tv = [item["association"]["total_variation_distance"] for item in selected]
        js = [item["association"]["jensen_shannon_divergence"] for item in selected]
        sw_um = [
            1_000.0 * item["continuous"]["sliced_wasserstein_mm"]
            for item in selected
        ]
        evidence = [
            abs(item["normalizing_constant"]["error"]) for item in selected
        ]
        runtime = [item["runtime_seconds"] for item in selected]
        tv_ratios = [
            item["association"]["total_variation_distance"]
            / item["thresholds"]["total_variation"]
            for item in selected
        ]
        sw_ratios = [
            item["continuous"]["sliced_wasserstein_mm"]
            / item["thresholds"]["path_sliced_wasserstein_mm"]
            for item in selected
        ]
        evidence_ratios = [
            abs(item["normalizing_constant"]["error"])
            / item["thresholds"]["absolute_log_evidence_error"]
            for item in selected
        ]
        tv_mean, tv_se = _mean_and_standard_error(tv)
        js_mean, js_se = _mean_and_standard_error(js)
        sw_mean, sw_se = _mean_and_standard_error(sw_um)
        evidence_mean, evidence_se = _mean_and_standard_error(evidence)
        runtime_mean, runtime_se = _mean_and_standard_error(runtime)
        tv_ratio_mean, tv_ratio_se = _mean_and_standard_error(tv_ratios)
        sw_ratio_mean, sw_ratio_se = _mean_and_standard_error(sw_ratios)
        evidence_ratio_mean, evidence_ratio_se = _mean_and_standard_error(
            evidence_ratios
        )
        summaries.append(
            {
                "regime": regime.name,
                "n_runs": len(selected),
                "mean_total_variation": tv_mean,
                "se_total_variation": tv_se,
                "mean_jensen_shannon": js_mean,
                "se_jensen_shannon": js_se,
                "mean_path_sliced_wasserstein_um": sw_mean,
                "se_path_sliced_wasserstein_um": sw_se,
                "mean_absolute_log_evidence_error": evidence_mean,
                "se_absolute_log_evidence_error": evidence_se,
                "mean_runtime_seconds": runtime_mean,
                "se_runtime_seconds": runtime_se,
                "mean_total_variation_threshold_ratio": tv_ratio_mean,
                "se_total_variation_threshold_ratio": tv_ratio_se,
                "mean_path_threshold_ratio": sw_ratio_mean,
                "se_path_threshold_ratio": sw_ratio_se,
                "mean_evidence_threshold_ratio": evidence_ratio_mean,
                "se_evidence_threshold_ratio": evidence_ratio_se,
                "all_runs_passed": all(item["passed"] for item in selected),
            }
        )
    return summaries


def _aggregate_map_runs(
    runs: list[dict[str, Any]],
    separations_um: tuple[float, ...],
    noise_levels_um: tuple[float, ...],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metrics = (
        "total_variation_distance",
        "jensen_shannon_divergence",
        "credible_set_recall_95",
        "exact_mass_covered",
        "n_tempering_steps",
        "mean_mh_acceptance_rate",
        "runtime_seconds",
    )
    arrays = {
        metric: np.empty((len(noise_levels_um), len(separations_um)), dtype=float)
        for metric in metrics
    }
    arrays["pass_fraction"] = np.empty_like(arrays[metrics[0]])
    arrays["strict_support_fraction"] = np.empty_like(arrays[metrics[0]])
    rows: list[dict[str, Any]] = []
    for separation_index, separation in enumerate(separations_um):
        for noise_index, noise in enumerate(noise_levels_um):
            selected = [
                item
                for item in runs
                if item["central_separation_um"] == separation
                and item["localization_std_um"] == noise
            ]
            row: dict[str, Any] = {
                "central_separation_um": separation,
                "localization_std_um": noise,
                "n_data_seeds": len(selected),
            }
            for metric in metrics:
                values = [float(item[metric]) for item in selected]
                average, standard_error = _mean_and_standard_error(values)
                arrays[metric][noise_index, separation_index] = average
                row[f"mean_{metric}"] = average
                row[f"se_{metric}"] = standard_error
            pass_fraction = mean(float(item["passed"]) for item in selected)
            arrays["pass_fraction"][noise_index, separation_index] = pass_fraction
            row["pass_fraction"] = pass_fraction
            strict_support_fraction = mean(
                float(item["strict_all_credible_states_observed"])
                for item in selected
            )
            arrays["strict_support_fraction"][noise_index, separation_index] = (
                strict_support_fraction
            )
            row["strict_all_credible_states_observed_fraction"] = (
                strict_support_fraction
            )
            rows.append(row)
    return rows, arrays


def _plot_gate_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    labels = [item["regime"].replace("_", " ") for item in summaries]
    indices = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    panels = (
        (
            "mean_total_variation_threshold_ratio",
            "se_total_variation_threshold_ratio",
            "Association TV / threshold",
        ),
        (
            "mean_path_threshold_ratio",
            "se_path_threshold_ratio",
            "Path SW / threshold",
        ),
        (
            "mean_evidence_threshold_ratio",
            "se_evidence_threshold_ratio",
            "Log-evidence error / threshold",
        ),
    )
    for axis, (metric, error, title) in zip(axes, panels, strict=True):
        colors = [
            "#238b45" if item["all_runs_passed"] else "#cb181d"
            for item in summaries
        ]
        axis.bar(
            indices,
            [item[metric] for item in summaries],
            yerr=[item[error] for item in summaries],
            color=colors,
            capsize=3,
        )
        axis.axhline(1.0, linestyle="--", color="black", linewidth=1)
        axis.set_xticks(indices, labels, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Collapsed Matérn SMC: exact-target calibration gate")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _plot_map(
    arrays: dict[str, np.ndarray],
    separations_um: tuple[float, ...],
    noise_levels_um: tuple[float, ...],
    output_path: Path,
) -> None:
    panels = (
        ("total_variation_distance", "Association TV", 0.0, None, "magma"),
        ("jensen_shannon_divergence", "Association JS", 0.0, None, "magma"),
        ("credible_set_recall_95", "95% credible-set recall", 0.0, 1.0, "viridis"),
        ("exact_mass_covered", "Exact mass covered", 0.0, 1.0, "viridis"),
        ("n_tempering_steps", "Tempering steps", 0.0, None, "cividis"),
        (
            "strict_support_fraction",
            "All credible states observed",
            0.0,
            1.0,
            "viridis",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 8.0), constrained_layout=True)
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
            range(len(separations_um)), [f"{value:g}" for value in separations_um]
        )
        axis.set_yticks(
            range(len(noise_levels_um)), [f"{value:g}" for value in noise_levels_um]
        )
        axis.set_xlabel("central separation (µm)")
        axis.set_ylabel("localization-noise SD (µm)")
        figure.colorbar(image, ax=axis, shrink=0.88)
    figure.suptitle("Collapsed Matérn SMC across the frozen ULM regime map")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_smc_calibration(
    *,
    output_dir: Path,
    n_particles: int = 8_192,
    n_smc_repeats: int = 3,
    map_n_particles: int = 4_096,
    map_n_data_seeds: int = 4,
    target_ess_fraction: float = 0.80,
    mh_sweeps: int = 8,
    final_mh_sweeps: int = 8,
    n_projections: int = 128,
    separations_um: tuple[float, ...] = DEFAULT_SEPARATIONS_UM,
    noise_levels_um: tuple[float, ...] = DEFAULT_NOISE_LEVELS_UM,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the frozen three-regime gate and the full calibration map."""

    positive_counts = (
        n_particles,
        n_smc_repeats,
        map_n_particles,
        map_n_data_seeds,
        mh_sweeps,
        n_projections,
        workers,
    )
    if any(value < 1 for value in positive_counts):
        raise ValueError("all particle, repeat, sweep, projection, and worker counts must be positive")
    if final_mh_sweeps < 0:
        raise ValueError("final_mh_sweeps must be non-negative")
    if not separations_um or min(separations_um) < 0:
        raise ValueError("separations must be non-empty and non-negative")
    if not noise_levels_um or min(noise_levels_um) <= 0:
        raise ValueError("noise levels must be non-empty and positive")

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
        )
        for regime in CALIBRATION_REGIMES
        for smc_seed in range(n_smc_repeats)
    ]
    gate_runs = _run_tasks(_evaluate_gate_task, gate_tasks, workers)
    gate_summaries = _aggregate_gate_runs(gate_runs)

    map_tasks = [
        (
            data_seed,
            float(separation),
            float(noise),
            map_n_particles,
            target_ess_fraction,
            mh_sweeps,
            final_mh_sweeps,
        )
        for separation in separations_um
        for noise in noise_levels_um
        for data_seed in range(map_n_data_seeds)
    ]
    map_runs = _run_tasks(_evaluate_map_task, map_tasks, workers)
    map_rows, map_arrays = _aggregate_map_runs(
        map_runs, separations_um, noise_levels_um
    )
    with (output_dir / "smc_matern_calibration_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)

    result: dict[str, Any] = {
        "experiment": "collapsed_matern_association_smc_calibration",
        "method": {
            "association_target": "pi_beta(a) proportional to p(y|a)^beta p0(a)",
            "base_distribution": (
                "uniform frame-wise permutations with frame zero fixing global labels"
            ),
            "tempering": "adaptive ESS; systematic resampling at every stage",
            "mutation": (
                "Metropolis single-frame and contiguous-segment label swaps, "
                "with additional rejuvenation at beta=1"
            ),
            "continuous_paths": "exact Gaussian draws from p(x|y,a)",
            "uses_exact_enumeration_during_inference": False,
        },
        "prior": {
            "family": "Matern-3/2",
            "coordinate_std_um": [
                1_000.0 * value for value in REFERENCE_COORDINATE_STD
            ],
            "length_scale_ms": 1_000.0 * REFERENCE_LENGTH_SCALE,
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
            "separations_um": list(separations_um),
            "noise_levels_um": list(noise_levels_um),
        },
        "gate_definition": {
            "association_tv": "at most max(0.05, 3 * exact-draw MC q95)",
            "association_js": "at most max(0.01, 3 * exact-draw MC q95)",
            "credible_set_recall_95": (
                "at least the 5th percentile of equal-size exact posterior draws"
            ),
            "exact_mass_covered": (
                "at least max(0.95, 5th percentile of exact posterior draws)"
            ),
            "path_sliced_wasserstein": (
                "at most max(1 um, 3 * independent exact-draw baseline)"
            ),
            "absolute_log_evidence_error": 0.10,
            "aggregate_rule": "every repeat in every frozen regime must pass",
            "map_interpretation": (
                "TV, JS, recall, and the absolute 95% mass floor are checked in "
                "every cell; the q05 mass-support check is reported as a "
                "finite-sample diagnostic and is expected to reject about 5% "
                "of exact finite samples"
            ),
            "numerical_tolerance": GATE_NUMERICAL_TOLERANCE,
        },
        "gate_runs": gate_runs,
        "gate_summary": gate_summaries,
        "gate_passed": all(item["all_runs_passed"] for item in gate_summaries),
        "map": {
            "n_evaluations": len(map_runs),
            "cellwise_all_checks_pass_fraction": mean(
                float(item["passed"]) for item in map_runs
            ),
            "total_variation_pass_fraction": mean(
                float(
                    item["total_variation_distance"]
                    <= item["tv_threshold"] + GATE_NUMERICAL_TOLERANCE
                )
                for item in map_runs
            ),
            "jensen_shannon_pass_fraction": mean(
                float(
                    item["jensen_shannon_divergence"]
                    <= item["js_threshold"] + GATE_NUMERICAL_TOLERANCE
                )
                for item in map_runs
            ),
            "calibrated_credible_set_recall_pass_fraction": mean(
                float(
                    item["credible_set_recall_95"]
                    + GATE_NUMERICAL_TOLERANCE
                    >= item["credible_set_recall_95_threshold"]
                )
                for item in map_runs
            ),
            "q05_mass_support_pass_fraction": mean(
                float(
                    item["exact_mass_covered"] + GATE_NUMERICAL_TOLERANCE
                    >= item["exact_mass_covered_threshold"]
                )
                for item in map_runs
            ),
            "absolute_95pct_mass_floor_pass_fraction": mean(
                float(
                    item["exact_mass_covered"] + GATE_NUMERICAL_TOLERANCE
                    >= 0.95
                )
                for item in map_runs
            ),
            "strict_all_credible_states_observed_fraction": mean(
                float(item["strict_all_credible_states_observed"])
                for item in map_runs
            ),
            "max_total_variation_distance": max(
                float(item["total_variation_distance"]) for item in map_runs
            ),
            "max_jensen_shannon_divergence": max(
                float(item["jensen_shannon_divergence"]) for item in map_runs
            ),
            "min_credible_set_recall_95": min(
                float(item["credible_set_recall_95"]) for item in map_runs
            ),
            "min_exact_mass_covered": min(
                float(item["exact_mass_covered"]) for item in map_runs
            ),
        },
        "map_runs": map_runs,
    }
    with (output_dir / "smc_matern_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_gate_summary(
        gate_summaries, output_dir / "smc_matern_calibration.png"
    )
    _plot_map(
        map_arrays,
        separations_um,
        noise_levels_um,
        output_dir / "smc_matern_calibration_map.png",
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
    parser.add_argument("--mh-sweeps", type=int, default=8)
    parser.add_argument("--final-mh-sweeps", type=int, default=8)
    parser.add_argument("--n-projections", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--separations-um", type=_parse_float_list, default=DEFAULT_SEPARATIONS_UM
    )
    parser.add_argument(
        "--noise-levels-um", type=_parse_float_list, default=DEFAULT_NOISE_LEVELS_UM
    )
    args = parser.parse_args()
    result = run_smc_calibration(
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        n_smc_repeats=args.n_smc_repeats,
        map_n_particles=args.map_n_particles,
        map_n_data_seeds=args.map_n_data_seeds,
        target_ess_fraction=args.target_ess_fraction,
        mh_sweeps=args.mh_sweeps,
        final_mh_sweeps=args.final_mh_sweeps,
        n_projections=args.n_projections,
        separations_um=args.separations_um,
        noise_levels_um=args.noise_levels_um,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "gate_passed": result["gate_passed"],
                "map_tv_pass_fraction": result["map"][
                    "total_variation_pass_fraction"
                ],
                "map_js_pass_fraction": result["map"][
                    "jensen_shannon_pass_fraction"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
