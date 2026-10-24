"""Reproducible controlled-crossing benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from .baselines import hungarian_tracker
from .inference import enumerate_association_posterior
from .metrics import (
    association_accuracy,
    identity_switches,
    posterior_reference_metrics,
    trajectory_rmse,
)
from .simulation import CrossingConfig, simulate_crossing


def _run_one(seed: int, config: CrossingConfig) -> dict[str, object]:
    truth, detections = simulate_crossing(config, seed=seed)
    position_assoc, position_tracks = hungarian_tracker(detections, constant_velocity=False)
    velocity_assoc, velocity_tracks = hungarian_tracker(detections, constant_velocity=True)
    posterior = enumerate_association_posterior(
        detections,
        localization_std=config.localization_std,
        acceleration_std=0.08,
        initial_velocity_std=0.6,
    )
    result: dict[str, object] = {
        "seed": seed,
        "position_hungarian": {
            "association_accuracy": association_accuracy(detections, position_assoc),
            "identity_switches": identity_switches(detections, position_assoc),
            "trajectory_rmse": trajectory_rmse(
                truth, detections, position_assoc, position_tracks
            ),
        },
        "constant_velocity_hungarian": {
            "association_accuracy": association_accuracy(detections, velocity_assoc),
            "identity_switches": identity_switches(detections, velocity_assoc),
            "trajectory_rmse": trajectory_rmse(
                truth, detections, velocity_assoc, velocity_tracks
            ),
        },
        "posterior": posterior_reference_metrics(detections, posterior),
    }
    return {
        "metrics": result,
        "truth": truth,
        "detections": detections,
        "position_tracks": position_tracks,
        "velocity_tracks": velocity_tracks,
        "posterior": posterior,
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method in ("position_hungarian", "constant_velocity_hungarian", "posterior"):
        first_metrics = runs[0]["metrics"][method]  # type: ignore[index]
        output[method] = {}
        for key in first_metrics:  # type: ignore[union-attr]
            values = [float(run["metrics"][method][key]) for run in runs]  # type: ignore[index]
            output[method][f"mean_{key}"] = mean(values)
    return output


def _plot_example(run: dict[str, object], output_path: Path) -> None:
    truth = run["truth"]
    detections = run["detections"]
    posterior = run["posterior"]
    position_tracks = run["position_tracks"]
    velocity_tracks = run["velocity_tracks"]

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), constrained_layout=True)
    colors = ("tab:blue", "tab:orange")
    for particle, color in enumerate(colors):
        axes[0].plot(
            truth.positions[:, particle, 0],
            truth.positions[:, particle, 1],
            color=color,
            linewidth=2.5,
            label=f"true bubble {particle}",
        )
    all_detections = np.concatenate(detections.detections)
    axes[0].scatter(
        all_detections[:, 0], all_detections[:, 1], c="black", s=16, alpha=0.55, label="detections"
    )
    axes[0].set_title("Latent crossing and detections")
    axes[0].legend(fontsize=8)

    for label, tracks, style in (
        ("position", position_tracks, "--"),
        ("constant velocity", velocity_tracks, "-"),
    ):
        for particle, color in enumerate(colors):
            axes[1].plot(
                tracks[:, particle, 0],
                tracks[:, particle, 1],
                linestyle=style,
                color=color,
                linewidth=2.0,
                label=f"{label}, id {particle}",
            )
    axes[1].set_title("Deterministic assignments")
    axes[1].legend(fontsize=7)

    ordered = np.argsort(posterior.probabilities)[::-1]
    leading = [index for index in ordered if posterior.probabilities[index] >= 1e-3][:3]
    hypothesis_styles = ("-", "--", ":")
    for rank, (index, style) in enumerate(
        zip(leading, hypothesis_styles[: len(leading)], strict=True), start=1
    ):
        for particle, color in enumerate(colors):
            label = f"rank {rank}: p={posterior.probabilities[index]:.2f}" if particle == 0 else None
            axes[2].plot(
                posterior.trajectories[index, :, particle, 0],
                posterior.trajectories[index, :, particle, 1],
                linestyle=style,
                color=color,
                alpha=1.0 - 0.2 * (rank - 1),
                linewidth=2.2,
                label=label,
            )
    axes[2].set_title("Leading posterior hypotheses")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel("lateral position (a.u.)")
        axis.set_ylabel("transverse position (a.u.)")
        axis.grid(alpha=0.2)
        axis.set_xlim(-1.15, 1.15)
        axis.set_ylim(-1.15, 1.15)
        axis.set_aspect("equal", adjustable="box")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_experiment(*, output_dir: Path, n_seeds: int = 32) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CrossingConfig()
    runs = [_run_one(seed, config) for seed in range(n_seeds)]
    summary: dict[str, object] = {
        "experiment": "controlled_two_bubble_crossing",
        "n_seeds": n_seeds,
        "config": {
            key: value
            for key, value in config.__dict__.items()
        },
        "aggregate": _aggregate(runs),
        "illustrative_seed": runs[0]["metrics"],
        "interpretation": (
            "The enumerable posterior is the exact association posterior under the stated "
            "linear-Gaussian motion model. It is a calibration target for future Smooth-SB/TSBM "
            "approximations, not itself a learned diffusion model."
        ),
    }
    with (output_dir / "crossing_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _plot_example(runs[0], output_dir / "crossing_example.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-seeds", type=int, default=32)
    args = parser.parse_args()
    summary = run_experiment(output_dir=args.output_dir, n_seeds=args.n_seeds)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
