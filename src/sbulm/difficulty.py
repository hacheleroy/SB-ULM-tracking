"""Difficulty map for two-bubble ULM data association."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .experiment import _run_one
from .simulation import CrossingConfig

DEFAULT_SEPARATIONS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
DEFAULT_NOISE_LEVELS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)

METRICS = (
    "constant_velocity_accuracy",
    "posterior_map_accuracy",
    "true_hypothesis_mass",
    "coverage_95",
    "credible_set_size_95",
    "normalized_entropy",
)


def _evaluate_task(task: tuple[int, float, float]) -> tuple[int, float, float, dict[str, float]]:
    seed, separation, noise = task
    config = CrossingConfig(central_separation=separation, localization_std=noise)
    metrics = _run_one(seed, config)["metrics"]
    velocity = metrics["constant_velocity_hungarian"]
    posterior = metrics["posterior"]
    selected = {
        "constant_velocity_accuracy": float(velocity["association_accuracy"]),
        "posterior_map_accuracy": float(posterior["map_association_accuracy"]),
        "true_hypothesis_mass": float(posterior["true_hypothesis_mass"]),
        "coverage_95": float(posterior["true_in_95pct_credible_set"]),
        "credible_set_size_95": float(posterior["credible_set_size_95"]),
        "normalized_entropy": float(posterior["normalized_entropy"]),
    }
    return seed, separation, noise, selected


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size <= 1:
        return mean, 0.0
    return mean, float(np.std(array, ddof=1) / np.sqrt(array.size))


def run_difficulty_sweep(
    *,
    output_dir: Path,
    separations: tuple[float, ...] = DEFAULT_SEPARATIONS,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    n_seeds: int = 12,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate deterministic and posterior tracking across a 2-D regime map."""

    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not separations or min(separations) < 0:
        raise ValueError("separations must be non-empty and non-negative")
    if not noise_levels or min(noise_levels) <= 0:
        raise ValueError("noise levels must be non-empty and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (seed, float(separation), float(noise))
        for separation in separations
        for noise in noise_levels
        for seed in range(n_seeds)
    ]
    if workers == 1:
        evaluations = [_evaluate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            evaluations = list(executor.map(_evaluate_task, tasks, chunksize=1))

    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}
    for _, separation, noise, metrics in evaluations:
        cell = grouped.setdefault(
            (separation, noise), {metric: [] for metric in METRICS}
        )
        for metric, value in metrics.items():
            cell[metric].append(value)

    shape = (len(noise_levels), len(separations))
    means = {metric: np.empty(shape, dtype=float) for metric in METRICS}
    standard_errors = {metric: np.empty(shape, dtype=float) for metric in METRICS}
    rows: list[dict[str, float]] = []
    for separation_index, separation in enumerate(separations):
        for noise_index, noise in enumerate(noise_levels):
            row: dict[str, float] = {
                "central_separation": float(separation),
                "localization_std": float(noise),
            }
            for metric in METRICS:
                cell_mean, cell_se = _mean_and_standard_error(
                    grouped[(float(separation), float(noise))][metric]
                )
                means[metric][noise_index, separation_index] = cell_mean
                standard_errors[metric][noise_index, separation_index] = cell_se
                row[f"mean_{metric}"] = cell_mean
                row[f"se_{metric}"] = cell_se
            rows.append(row)

    csv_path = output_dir / "difficulty_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result: dict[str, Any] = {
        "experiment": "crossing_difficulty_map",
        "n_seeds": n_seeds,
        "separations": list(separations),
        "noise_levels": list(noise_levels),
        "base_config": asdict(CrossingConfig()),
        "metric_means": {metric: values.tolist() for metric, values in means.items()},
        "metric_standard_errors": {
            metric: values.tolist() for metric, values in standard_errors.items()
        },
    }
    with (output_dir / "difficulty_map.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    _plot_difficulty_map(means, separations, noise_levels, output_dir / "difficulty_map.png")
    return result


def _plot_difficulty_map(
    means: dict[str, np.ndarray],
    separations: tuple[float, ...],
    noise_levels: tuple[float, ...],
    output_path: Path,
) -> None:
    panels = (
        ("constant_velocity_accuracy", "CV Hungarian accuracy", 0.0, 1.0, "viridis"),
        ("posterior_map_accuracy", "Posterior MAP accuracy", 0.0, 1.0, "viridis"),
        ("true_hypothesis_mass", "Mass of true association", 0.0, 1.0, "magma"),
        ("coverage_95", "True association in 95% set", 0.0, 1.0, "viridis"),
        ("credible_set_size_95", "95% credible-set size", 1.0, None, "plasma"),
        ("normalized_entropy", "Normalized posterior entropy", 0.0, 1.0, "cividis"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    for axis, (metric, title, lower, upper, color_map) in zip(
        axes.flat, panels, strict=True
    ):
        image = axis.imshow(
            means[metric],
            origin="lower",
            aspect="auto",
            vmin=lower,
            vmax=upper,
            cmap=color_map,
        )
        axis.set_title(title)
        axis.set_xticks(range(len(separations)), [f"{value:g}" for value in separations])
        axis.set_yticks(range(len(noise_levels)), [f"{value:g}" for value in noise_levels])
        axis.set_xlabel("central bubble separation (a.u.)")
        axis.set_ylabel("localization noise SD (a.u.)")
        figure.colorbar(image, ax=axis, shrink=0.88)
    figure.suptitle("Two-bubble crossing: deterministic decision and posterior uncertainty")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--separations", type=_parse_float_list, default=DEFAULT_SEPARATIONS
    )
    parser.add_argument(
        "--noise-levels", type=_parse_float_list, default=DEFAULT_NOISE_LEVELS
    )
    args = parser.parse_args()
    result = run_difficulty_sweep(
        output_dir=args.output_dir,
        separations=args.separations,
        noise_levels=args.noise_levels,
        n_seeds=args.n_seeds,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "n_seeds": result["n_seeds"],
                "grid_shape": [len(result["noise_levels"]), len(result["separations"])],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

