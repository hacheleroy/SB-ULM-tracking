"""WP2 comparison of hard marginals and a Gaussian localization likelihood."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from .localization import enumerate_matern_localization_posterior
from .metrics import posterior_reference_metrics, true_in_credible_set
from .physical_sweep import (
    DEFAULT_NOISE_LEVELS_UM,
    DEFAULT_SEPARATIONS_UM,
    REFERENCE_FRAME_RATE_HZ,
    REFERENCE_SPEED_MM_S,
)
from .simulation import simulate_crossing
from .smooth_sb import enumerate_smooth_sb_target

CREDIBLE_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)

MAP_METRICS = (
    "hard_true_mass",
    "likelihood_true_mass",
    "hard_map_accuracy",
    "likelihood_map_accuracy",
    "hard_coverage_95",
    "likelihood_coverage_95",
    "coverage_95_gain",
    "hard_credible_set_size_95",
    "likelihood_credible_set_size_95",
    "hard_normalized_entropy",
    "likelihood_normalized_entropy",
)


def _reference_prior() -> tuple[tuple[float, float], float]:
    config = physical_crossing_config(
        speed_mm_s=REFERENCE_SPEED_MM_S,
        frame_rate_hz=REFERENCE_FRAME_RATE_HZ,
        central_separation_um=0.0,
        localization_std_um=1.0,
    )
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=REFERENCE_SPEED_MM_S, nu=1.5
    )
    return (
        (calibration.coordinate_std[0], calibration.coordinate_std[1]),
        calibration.length_scale,
    )


REFERENCE_COORDINATE_STD, REFERENCE_LENGTH_SCALE = _reference_prior()


def _evaluate_task(
    task: tuple[int, float, float],
) -> tuple[int, float, float, dict[str, float]]:
    seed, separation_um, noise_um = task
    config = physical_crossing_config(
        speed_mm_s=REFERENCE_SPEED_MM_S,
        frame_rate_hz=REFERENCE_FRAME_RATE_HZ,
        central_separation_um=separation_um,
        localization_std_um=noise_um,
    )
    _, detections = simulate_crossing(config, seed=seed)
    hard = enumerate_smooth_sb_target(
        detections,
        model_times=detections.times,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
    )
    likelihood = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=REFERENCE_COORDINATE_STD,
        length_scale=REFERENCE_LENGTH_SCALE,
    )
    hard_metrics = posterior_reference_metrics(detections, hard)
    likelihood_metrics = posterior_reference_metrics(detections, likelihood)
    metrics = {
        "hard_true_mass": float(hard_metrics["true_hypothesis_mass"]),
        "likelihood_true_mass": float(likelihood_metrics["true_hypothesis_mass"]),
        "hard_map_accuracy": float(hard_metrics["map_association_accuracy"]),
        "likelihood_map_accuracy": float(
            likelihood_metrics["map_association_accuracy"]
        ),
        "hard_coverage_95": float(hard_metrics["true_in_95pct_credible_set"]),
        "likelihood_coverage_95": float(
            likelihood_metrics["true_in_95pct_credible_set"]
        ),
        "coverage_95_gain": float(
            likelihood_metrics["true_in_95pct_credible_set"]
            - hard_metrics["true_in_95pct_credible_set"]
        ),
        "hard_credible_set_size_95": float(hard_metrics["credible_set_size_95"]),
        "likelihood_credible_set_size_95": float(
            likelihood_metrics["credible_set_size_95"]
        ),
        "hard_normalized_entropy": float(hard_metrics["normalized_entropy"]),
        "likelihood_normalized_entropy": float(
            likelihood_metrics["normalized_entropy"]
        ),
    }
    for level in CREDIBLE_LEVELS:
        suffix = round(100 * level)
        metrics[f"hard_coverage_{suffix}"] = float(
            true_in_credible_set(detections, hard, level)
        )
        metrics[f"likelihood_coverage_{suffix}"] = float(
            true_in_credible_set(detections, likelihood, level)
        )
    return seed, separation_um, noise_um, metrics


def _run_tasks(
    function: Callable[[Any], Any], tasks: list[Any], workers: int
) -> list[Any]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, tasks, chunksize=1))


def _summary(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    standard_error = 0.0 if len(values) == 1 else stdev(values) / np.sqrt(len(values))
    return average, standard_error


def _plot_map(
    means: dict[str, np.ndarray],
    separations_um: tuple[float, ...],
    noise_levels_um: tuple[float, ...],
    output_path: Path,
) -> None:
    panels = (
        ("hard_coverage_95", "Hard-marginal 95% inclusion", 0.0, 1.0, "viridis", None),
        (
            "likelihood_coverage_95",
            "Gaussian-likelihood 95% inclusion",
            0.0,
            1.0,
            "viridis",
            None,
        ),
        ("coverage_95_gain", "Inclusion gain", -1.0, 1.0, "coolwarm", None),
        (
            "hard_credible_set_size_95",
            "Hard-marginal 95% set size",
            None,
            None,
            "plasma",
            LogNorm(vmin=1.0, vmax=256.0),
        ),
        (
            "likelihood_credible_set_size_95",
            "Gaussian-likelihood 95% set size",
            None,
            None,
            "plasma",
            LogNorm(vmin=1.0, vmax=256.0),
        ),
        (
            "likelihood_normalized_entropy",
            "Gaussian-likelihood entropy",
            0.0,
            1.0,
            "cividis",
            None,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for axis, (metric, title, lower, upper, color_map, norm) in zip(
        axes.flat, panels, strict=True
    ):
        image = axis.imshow(
            means[metric],
            origin="lower",
            aspect="auto",
            vmin=lower,
            vmax=upper,
            cmap=color_map,
            norm=norm,
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
    figure.suptitle(
        "WP2: fixed Matérn prior, hard observations versus Gaussian likelihood"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _plot_coverage_curve(rows: list[dict[str, float]], output_path: Path) -> None:
    levels = np.asarray([row["nominal_mass"] for row in rows])
    hard = np.asarray([row["hard_empirical_inclusion"] for row in rows])
    likelihood = np.asarray([row["likelihood_empirical_inclusion"] for row in rows])
    hard_error = np.asarray([row["hard_standard_error"] for row in rows])
    likelihood_error = np.asarray([row["likelihood_standard_error"] for row in rows])

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    axis.plot([0.45, 1.0], [0.45, 1.0], "--", color="black", label="nominal")
    axis.errorbar(levels, hard, yerr=hard_error, marker="o", capsize=3, label="hard marginals")
    axis.errorbar(
        levels,
        likelihood,
        yerr=likelihood_error,
        marker="o",
        capsize=3,
        label="Gaussian likelihood",
    )
    axis.set_xlim(0.47, 1.01)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("nominal credible mass")
    axis.set_ylabel("reference-path inclusion frequency")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_wp2_likelihood_sweep(
    *,
    output_dir: Path,
    separations_um: tuple[float, ...] = DEFAULT_SEPARATIONS_UM,
    noise_levels_um: tuple[float, ...] = DEFAULT_NOISE_LEVELS_UM,
    n_seeds: int = 64,
    workers: int = 1,
) -> dict[str, Any]:
    """Compare hard intermediate marginals with an explicit noise likelihood."""

    if n_seeds < 1 or workers < 1:
        raise ValueError("n_seeds and workers must be positive")
    if not separations_um or min(separations_um) < 0:
        raise ValueError("separations must be non-empty and non-negative")
    if not noise_levels_um or min(noise_levels_um) <= 0:
        raise ValueError("noise levels must be non-empty and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (seed, float(separation), float(noise))
        for separation in separations_um
        for noise in noise_levels_um
        for seed in range(n_seeds)
    ]
    evaluations = _run_tasks(_evaluate_task, tasks, workers)
    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}
    for _, separation, noise, metrics in evaluations:
        cell = grouped.setdefault(
            (separation, noise), {metric: [] for metric in MAP_METRICS}
        )
        for metric in MAP_METRICS:
            cell[metric].append(metrics[metric])

    shape = (len(noise_levels_um), len(separations_um))
    means = {metric: np.empty(shape, dtype=float) for metric in MAP_METRICS}
    rows: list[dict[str, float]] = []
    for separation_index, separation in enumerate(separations_um):
        for noise_index, noise in enumerate(noise_levels_um):
            row = {
                "central_separation_um": float(separation),
                "localization_std_um": float(noise),
            }
            for metric in MAP_METRICS:
                average, standard_error = _summary(
                    grouped[(float(separation), float(noise))][metric]
                )
                means[metric][noise_index, separation_index] = average
                row[f"mean_{metric}"] = average
                row[f"se_{metric}"] = standard_error
            rows.append(row)

    with (output_dir / "wp2_likelihood_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coverage_rows: list[dict[str, float]] = []
    for level in CREDIBLE_LEVELS:
        suffix = round(100 * level)
        hard_values = [item[3][f"hard_coverage_{suffix}"] for item in evaluations]
        likelihood_values = [
            item[3][f"likelihood_coverage_{suffix}"] for item in evaluations
        ]
        hard_average, hard_se = _summary(hard_values)
        likelihood_average, likelihood_se = _summary(likelihood_values)
        coverage_rows.append(
            {
                "nominal_mass": level,
                "hard_empirical_inclusion": hard_average,
                "hard_standard_error": hard_se,
                "likelihood_empirical_inclusion": likelihood_average,
                "likelihood_standard_error": likelihood_se,
            }
        )
    with (output_dir / "wp2_credible_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    result: dict[str, Any] = {
        "experiment": "wp2_gaussian_localization_likelihood",
        "comparison": (
            "Identical fixed Matérn-3/2 prior; only the observation model changes."
        ),
        "prior": {
            "coordinate_std_um": [
                1_000.0 * value for value in REFERENCE_COORDINATE_STD
            ],
            "length_scale_ms": 1_000.0 * REFERENCE_LENGTH_SCALE,
            "rms_speed_mm_s": REFERENCE_SPEED_MM_S,
            "frame_rate_hz": REFERENCE_FRAME_RATE_HZ,
        },
        "observation_models": {
            "hard": "observed localization is imposed as the path state",
            "likelihood": (
                "latent Matérn path integrated under y|x ~ N(x, sigma_loc^2 I)"
            ),
        },
        "n_seeds_per_cell": n_seeds,
        "n_evaluations": len(evaluations),
        "separations_um": list(separations_um),
        "noise_levels_um": list(noise_levels_um),
        "metric_means": {metric: values.tolist() for metric, values in means.items()},
        "global_credible_coverage": coverage_rows,
    }
    with (output_dir / "wp2_localization_likelihood.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_map(
        means,
        separations_um,
        noise_levels_um,
        output_dir / "wp2_likelihood_map.png",
    )
    _plot_coverage_curve(
        coverage_rows, output_dir / "wp2_credible_coverage.png"
    )
    return result


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--separations-um", type=_parse_float_list, default=DEFAULT_SEPARATIONS_UM
    )
    parser.add_argument(
        "--noise-levels-um", type=_parse_float_list, default=DEFAULT_NOISE_LEVELS_UM
    )
    args = parser.parse_args()
    result = run_wp2_likelihood_sweep(
        output_dir=args.output_dir,
        separations_um=args.separations_um,
        noise_levels_um=args.noise_levels_um,
        n_seeds=args.n_seeds,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "n_evaluations": result["n_evaluations"],
                "grid_shape": [
                    len(result["noise_levels_um"]),
                    len(result["separations_um"]),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
