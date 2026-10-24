"""Physically scaled Matérn-prior sweep for the controlled ULM crossing."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .calibration import enumerate_velocity_calibrated_target, physical_crossing_config
from .metrics import posterior_reference_metrics
from .simulation import simulate_crossing
from .smooth_sb import enumerate_smooth_sb_target

DEFAULT_SEPARATIONS_UM = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0)
DEFAULT_NOISE_LEVELS_UM = (1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0)

# Acquisition-level mean velocities from the 99 rows listed in the ULMShare
# summary at the audited 2026-10-16 repository snapshot.
ULMSHARE_SPEED_SUMMARY_MM_S = {
    "n_acquisitions": 99,
    "minimum": 5.81,
    "q05": 6.219,
    "q25": 6.585,
    "median": 6.84,
    "mean": 7.655858585858586,
    "q75": 7.345,
    "q95": 11.622,
    "maximum": 17.63,
}

REFERENCE_SPEED_MM_S = ULMSHARE_SPEED_SUMMARY_MM_S["median"]
REFERENCE_FRAME_RATE_HZ = 1_000.0
SENSITIVITY_SPEEDS_MM_S = (
    ULMSHARE_SPEED_SUMMARY_MM_S["q05"],
    ULMSHARE_SPEED_SUMMARY_MM_S["median"],
    ULMSHARE_SPEED_SUMMARY_MM_S["q95"],
)

MAP_METRICS = (
    "default_true_mass",
    "calibrated_true_mass",
    "true_mass_change",
    "calibrated_map_accuracy",
    "calibrated_coverage_95",
    "calibrated_credible_set_size_95",
    "calibrated_normalized_entropy",
    "calibrated_length_scale_ms",
)


def _evaluate_map_task(
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
    default = enumerate_smooth_sb_target(detections)
    calibrated, calibration = enumerate_velocity_calibrated_target(
        detections, rms_speed=REFERENCE_SPEED_MM_S
    )
    default_metrics = posterior_reference_metrics(detections, default)
    calibrated_metrics = posterior_reference_metrics(detections, calibrated)
    default_true_mass = float(default_metrics["true_hypothesis_mass"])
    calibrated_true_mass = float(calibrated_metrics["true_hypothesis_mass"])
    metrics = {
        "default_true_mass": default_true_mass,
        "calibrated_true_mass": calibrated_true_mass,
        "true_mass_change": calibrated_true_mass - default_true_mass,
        "calibrated_map_accuracy": float(
            calibrated_metrics["map_association_accuracy"]
        ),
        "calibrated_coverage_95": float(
            calibrated_metrics["true_in_95pct_credible_set"]
        ),
        "calibrated_credible_set_size_95": float(
            calibrated_metrics["credible_set_size_95"]
        ),
        "calibrated_normalized_entropy": float(
            calibrated_metrics["normalized_entropy"]
        ),
        "calibrated_length_scale_ms": 1_000.0 * calibration.length_scale,
    }
    return seed, separation_um, noise_um, metrics


def _evaluate_speed_task(task: tuple[int, float]) -> tuple[int, float, dict[str, float]]:
    seed, assumed_speed = task
    config = physical_crossing_config(
        speed_mm_s=REFERENCE_SPEED_MM_S,
        frame_rate_hz=REFERENCE_FRAME_RATE_HZ,
        central_separation_um=5.0,
        localization_std_um=5.0,
    )
    _, detections = simulate_crossing(config, seed=seed)
    posterior, calibration = enumerate_velocity_calibrated_target(
        detections, rms_speed=assumed_speed
    )
    metrics = posterior_reference_metrics(detections, posterior)
    return seed, assumed_speed, {
        "true_mass": float(metrics["true_hypothesis_mass"]),
        "map_accuracy": float(metrics["map_association_accuracy"]),
        "coverage_95": float(metrics["true_in_95pct_credible_set"]),
        "credible_set_size_95": float(metrics["credible_set_size_95"]),
        "normalized_entropy": float(metrics["normalized_entropy"]),
        "length_scale_ms": 1_000.0 * calibration.length_scale,
    }


def _run_tasks(function: Any, tasks: list[Any], workers: int) -> list[Any]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, tasks, chunksize=1))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_error": 0.0 if len(values) == 1 else stdev(values) / np.sqrt(len(values)),
    }


def _plot_map(
    means: dict[str, np.ndarray],
    separations_um: tuple[float, ...],
    noise_levels_um: tuple[float, ...],
    output_path: Path,
) -> None:
    panels = (
        ("default_true_mass", "Upstream-default true-path mass", 0.0, 1.0, "magma"),
        ("calibrated_true_mass", "Velocity-calibrated true-path mass", 0.0, 1.0, "magma"),
        ("true_mass_change", "Change after calibration", -1.0, 1.0, "coolwarm"),
        ("calibrated_map_accuracy", "Calibrated MAP accuracy", 0.0, 1.0, "viridis"),
        ("calibrated_coverage_95", "True path in calibrated 95% set", 0.0, 1.0, "viridis"),
        ("calibrated_length_scale_ms", "Calibrated length scale (ms)", 0.0, None, "cividis"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), constrained_layout=True)
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
        "Matérn-3/2 target at 1 kHz; 6.84 mm/s reference bubble speed"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _plot_speed_sensitivity(rows: list[dict[str, float]], output_path: Path) -> None:
    speeds = [row["assumed_speed_mm_s"] for row in rows]
    mass = [row["mean_true_mass"] for row in rows]
    mass_error = [row["se_true_mass"] for row in rows]
    length = [row["mean_length_scale_ms"] for row in rows]
    length_error = [row["se_length_scale_ms"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)
    axes[0].errorbar(speeds, mass, yerr=mass_error, marker="o", capsize=4)
    axes[0].set_ylabel("reference-consistent path mass")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].errorbar(speeds, length, yerr=length_error, marker="o", capsize=4)
    axes[1].set_ylabel("calibrated length scale (ms)")
    for axis in axes:
        axis.set_xlabel("assumed RMS speed scale (mm/s)")
        axis.grid(alpha=0.25)
    figure.suptitle("Speed-scale sensitivity at 5 µm separation and 5 µm noise")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_physical_calibration_sweep(
    *,
    output_dir: Path,
    separations_um: tuple[float, ...] = DEFAULT_SEPARATIONS_UM,
    noise_levels_um: tuple[float, ...] = DEFAULT_NOISE_LEVELS_UM,
    n_seeds: int = 16,
    sensitivity_seeds: int = 32,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the physical regime map and a ULMShare speed-scale sensitivity."""

    if n_seeds < 1 or sensitivity_seeds < 1 or workers < 1:
        raise ValueError("seed counts and workers must be positive")
    if not separations_um or min(separations_um) < 0:
        raise ValueError("separations must be non-empty and non-negative")
    if not noise_levels_um or min(noise_levels_um) <= 0:
        raise ValueError("noise levels must be non-empty and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    map_tasks = [
        (seed, float(separation), float(noise))
        for separation in separations_um
        for noise in noise_levels_um
        for seed in range(n_seeds)
    ]
    map_evaluations = _run_tasks(_evaluate_map_task, map_tasks, workers)
    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}
    for _, separation, noise, metrics in map_evaluations:
        cell = grouped.setdefault(
            (separation, noise), {metric: [] for metric in MAP_METRICS}
        )
        for metric, value in metrics.items():
            cell[metric].append(value)

    shape = (len(noise_levels_um), len(separations_um))
    means = {metric: np.empty(shape, dtype=float) for metric in MAP_METRICS}
    map_rows: list[dict[str, float]] = []
    for separation_index, separation in enumerate(separations_um):
        for noise_index, noise in enumerate(noise_levels_um):
            row = {
                "central_separation_um": float(separation),
                "localization_std_um": float(noise),
            }
            for metric in MAP_METRICS:
                summary = _summary(grouped[(float(separation), float(noise))][metric])
                means[metric][noise_index, separation_index] = summary["mean"]
                row[f"mean_{metric}"] = summary["mean"]
                row[f"se_{metric}"] = summary["standard_error"]
            map_rows.append(row)

    with (output_dir / "physical_matern_calibration_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)

    speed_tasks = [
        (seed, float(speed))
        for speed in SENSITIVITY_SPEEDS_MM_S
        for seed in range(sensitivity_seeds)
    ]
    speed_evaluations = _run_tasks(_evaluate_speed_task, speed_tasks, workers)
    speed_grouped: dict[float, dict[str, list[float]]] = {}
    for _, speed, metrics in speed_evaluations:
        cell = speed_grouped.setdefault(speed, {metric: [] for metric in metrics})
        for metric, value in metrics.items():
            cell[metric].append(value)
    speed_rows: list[dict[str, float]] = []
    for speed in SENSITIVITY_SPEEDS_MM_S:
        row = {"assumed_speed_mm_s": float(speed)}
        for metric, values in speed_grouped[float(speed)].items():
            summary = _summary(values)
            row[f"mean_{metric}"] = summary["mean"]
            row[f"se_{metric}"] = summary["standard_error"]
        speed_rows.append(row)
    with (output_dir / "physical_matern_speed_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(speed_rows[0]))
        writer.writeheader()
        writer.writerows(speed_rows)

    result: dict[str, Any] = {
        "experiment": "physical_matern_velocity_calibration",
        "model": {
            "nu": 1.5,
            "calibration_equation": "ell = sqrt(3 * sum_d sigma_d^2) / v_rms",
            "variance_rule": "sigma_d = empirical SD of observed coordinate d",
            "upstream_default": {"t_end": 2.0, "length_scale": 3.0},
        },
        "physical_reference": {
            "position_unit": "mm",
            "time_unit": "s",
            "frame_rate_hz": REFERENCE_FRAME_RATE_HZ,
            "trajectory_speed_mm_s": REFERENCE_SPEED_MM_S,
            "displacement_per_frame_um": (
                1_000.0 * REFERENCE_SPEED_MM_S / REFERENCE_FRAME_RATE_HZ
            ),
            "ulmshare_acquisition_mean_velocity_summary_mm_s": (
                ULMSHARE_SPEED_SUMMARY_MM_S
            ),
            "caveat": (
                "The ULMShare values are acquisition-level mean velocities, used here as "
                "RMS-like design scales; they do not identify an exact population RMS prior."
            ),
        },
        "n_seeds_per_map_cell": n_seeds,
        "sensitivity_seeds": sensitivity_seeds,
        "separations_um": list(separations_um),
        "noise_levels_um": list(noise_levels_um),
        "metric_means": {metric: values.tolist() for metric, values in means.items()},
        "speed_sensitivity": speed_rows,
    }
    with (output_dir / "physical_matern_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_map(
        means,
        separations_um,
        noise_levels_um,
        output_dir / "physical_matern_calibration_map.png",
    )
    _plot_speed_sensitivity(
        speed_rows, output_dir / "physical_matern_speed_sensitivity.png"
    )
    return result


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--sensitivity-seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--separations-um", type=_parse_float_list, default=DEFAULT_SEPARATIONS_UM
    )
    parser.add_argument(
        "--noise-levels-um", type=_parse_float_list, default=DEFAULT_NOISE_LEVELS_UM
    )
    args = parser.parse_args()
    result = run_physical_calibration_sweep(
        output_dir=args.output_dir,
        separations_um=args.separations_um,
        noise_levels_um=args.noise_levels_um,
        n_seeds=args.n_seeds,
        sensitivity_seeds=args.sensitivity_seeds,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "n_seeds_per_map_cell": result["n_seeds_per_map_cell"],
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
