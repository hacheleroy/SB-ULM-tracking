"""Test blocked birth/death proposals beyond two slots and four frames."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .calibration import calibrate_matern_from_coordinate_std
from .existence_smc import BirthDeathSMCConfig, sample_slot_existence_posterior
from .simulation import (
    DetectionCorruptionConfig,
    MultiSlotCrossingConfig,
    simulate_multislot_crossing,
    simulate_slot_existence_observations,
)
from .slot_existence import (
    BirthDeathPrior,
    SlotEventMarginals,
    SlotExistenceMaternModel,
    SlotExistencePosterior,
    enumerate_slot_existence_support,
    slot_event_marginals,
)
from .tsbm_reference import compare_tsbm_association_samples

ROI_LOWER = (-0.04, -0.04)
ROI_UPPER = (0.04, 0.04)
REFERENCE_SPEED_MM_S = 6.84
DEFAULT_BIRTH_DEATH_PRIOR = BirthDeathPrior(
    initial_probability=0.50,
    birth_probability=0.25,
    survival_probability=0.80,
)


@dataclass(frozen=True)
class ScalingRegime:
    """Frozen three-slot exact calibration case."""

    name: str
    truth_mask: tuple[tuple[bool, bool, bool], ...]
    detection_probability: float
    expected_clutter_per_frame: float
    corruption_seed: int


EXACT_REGIMES = (
    ScalingRegime(
        "three_slots_five_frames",
        (
            (True, False, False),
            (True, True, False),
            (True, True, True),
            (True, True, True),
            (False, True, True),
        ),
        0.35,
        0.08,
        3,
    ),
    ScalingRegime(
        "three_slots_six_frames",
        (
            (True, False, False),
            (True, True, False),
            (True, True, False),
            (True, True, True),
            (False, True, True),
            (False, False, True),
        ),
        0.55,
        0.08,
        0,
    ),
)


@dataclass(frozen=True)
class _ScalingCase:
    model: SlotExistenceMaternModel
    truth_mask: np.ndarray
    localization_std: float
    coordinate_std: tuple[float, ...]
    length_scale: float


def _make_case(regime: ScalingRegime) -> _ScalingCase:
    n_frames = len(regime.truth_mask)
    crossing = MultiSlotCrossingConfig(
        n_slots=3,
        n_frames=n_frames,
        frame_period=0.001,
        half_span=REFERENCE_SPEED_MM_S * 0.001 * (n_frames - 1) / 2.0,
        central_separation=0.0,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=4)
    truth_mask = np.asarray(regime.truth_mask, dtype=bool)
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=regime.detection_probability,
            expected_clutter_per_frame=regime.expected_clutter_per_frame,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=regime.corruption_seed,
    )
    coordinate_std = tuple(
        float(value) for value in np.std(truth.positions, axis=(0, 1))
    )
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=REFERENCE_SPEED_MM_S
    )
    model = SlotExistenceMaternModel(
        detections,
        n_slots=3,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=regime.detection_probability,
        expected_clutter_per_frame=regime.expected_clutter_per_frame,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    return _ScalingCase(
        model=model,
        truth_mask=truth_mask,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )


def _exact_posterior(
    model: SlotExistenceMaternModel,
) -> tuple[SlotExistencePosterior, float]:
    associations, masks = enumerate_slot_existence_support(
        model, max_hypotheses=400_000
    )
    log_scores = model.log_joint_density(associations)
    log_evidence = float(logsumexp(log_scores))
    normalized = log_scores - log_evidence
    return (
        SlotExistencePosterior(
            associations=associations,
            existence_masks=masks,
            log_probabilities=normalized,
            probabilities=np.exp(normalized),
        ),
        log_evidence,
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

    return (
        0.5 * float(np.sum(np.abs(exact - empirical))),
        0.5 * (divergence(exact, midpoint) + divergence(empirical, midpoint)),
    )


def _finite_sample_baseline(
    posterior: SlotExistencePosterior,
    *,
    n_samples: int,
    n_replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Stream multinomial draws to avoid an H-by-replicate allocation."""

    rng = np.random.default_rng(seed)
    tv = np.empty(n_replicates)
    js = np.empty(n_replicates)
    mass = np.empty(n_replicates)
    recall = np.empty(n_replicates)
    order = np.argsort(posterior.probabilities)[::-1]
    credible = order[: posterior.credible_set_size(0.95)]
    for replicate in range(n_replicates):
        counts = rng.multinomial(n_samples, posterior.probabilities)
        empirical = counts / n_samples
        tv[replicate], js[replicate] = _distribution_distances(
            posterior.probabilities, empirical
        )
        observed = counts > 0
        mass[replicate] = float(np.sum(posterior.probabilities[observed]))
        recall[replicate] = float(np.mean(observed[credible]))
    output: dict[str, float | int] = {
        "n_samples": n_samples,
        "n_replicates": n_replicates,
    }
    for name, values in (
        ("total_variation", tv),
        ("jensen_shannon", js),
        ("exact_mass_covered", mass),
        ("credible_set_recall_95", recall),
    ):
        q05, median, q95 = np.quantile(values, [0.05, 0.50, 0.95])
        output[f"{name}_q05"] = float(q05)
        output[f"{name}_median"] = float(median)
        output[f"{name}_q95"] = float(q95)
    return output


def _event_errors(
    exact: SlotEventMarginals, sampled: SlotEventMarginals
) -> dict[str, float]:
    return {
        "existence": float(
            np.max(
                np.abs(
                    exact.existence_probabilities
                    - sampled.existence_probabilities
                )
            )
        ),
        "missed": float(
            np.max(np.abs(exact.missed_probabilities - sampled.missed_probabilities))
        ),
        "clutter": float(
            np.max(np.abs(exact.expected_clutter - sampled.expected_clutter))
        ),
        "birth_time": float(
            np.max(
                np.abs(
                    exact.birth_time_probabilities
                    - sampled.birth_time_probabilities
                )
            )
        ),
        "death_time": float(
            np.max(
                np.abs(
                    exact.death_time_probabilities
                    - sampled.death_time_probabilities
                )
            )
        ),
    }


def _controls(
    method: str,
    *,
    n_particles: int,
    seed: int,
    final_mh_sweeps: int = 8,
    target_ess_fraction: float = 0.80,
) -> BirthDeathSMCConfig:
    common = {
        "n_particles": n_particles,
        "target_ess_fraction": target_ess_fraction,
        "mh_sweeps": 8,
        "final_mh_sweeps": final_mh_sweeps,
        "seed": seed,
    }
    if method == "local":
        return BirthDeathSMCConfig(**common)
    if method == "blocked":
        return BirthDeathSMCConfig(
            **common,
            association_move_probability=0.30,
            interval_move_probability=0.45,
            slot_swap_probability=0.10,
            blocked_slot_move_probability=0.15,
        )
    if method == "paired":
        return BirthDeathSMCConfig(
            **common,
            association_move_probability=0.20,
            interval_move_probability=0.35,
            slot_swap_probability=0.05,
            blocked_slot_move_probability=0.15,
            paired_slot_move_probability=0.25,
            paired_slot_candidates=2,
        )
    raise ValueError(f"unknown method: {method}")


def _sample_case(
    case: _ScalingCase,
    *,
    method: str,
    n_particles: int,
    seed: int,
    final_mh_sweeps: int = 8,
    target_ess_fraction: float = 0.80,
):
    controls = _controls(
        method,
        n_particles=n_particles,
        seed=seed,
        final_mh_sweeps=final_mh_sweeps,
        target_ess_fraction=target_ess_fraction,
    )
    start = time.perf_counter()
    result = sample_slot_existence_posterior(
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
    return result, time.perf_counter() - start


def _run_exact_regime(
    regime: ScalingRegime,
    *,
    n_particles: int,
    n_repeats: int,
    baseline_replicates: int,
    methods: tuple[str, ...] = ("local", "blocked"),
) -> dict[str, Any]:
    case = _make_case(regime)
    exact, exact_log_evidence = _exact_posterior(case.model)
    baseline = _finite_sample_baseline(
        exact,
        n_samples=n_particles,
        n_replicates=baseline_replicates,
        seed=10_000 + regime.corruption_seed,
    )
    exact_events = slot_event_marginals(
        exact.associations,
        case.model.sequence,
        probabilities=exact.probabilities,
    )
    thresholds = {
        "total_variation": min(
            1.0, float(baseline["total_variation_q95"]) + 0.03
        ),
        "jensen_shannon": min(
            float(np.log(2.0)),
            float(baseline["jensen_shannon_q95"]) + 0.02,
        ),
        "exact_mass_covered": max(
            0.0, float(baseline["exact_mass_covered_q05"]) - 0.03
        ),
        "credible_set_recall_95": max(
            0.0, float(baseline["credible_set_recall_95_q05"]) - 0.03
        ),
        "maximum_event_error": 0.06,
    }
    runs: list[dict[str, Any]] = []
    for method in methods:
        for seed in range(n_repeats):
            sampled, runtime = _sample_case(
                case,
                method=method,
                n_particles=n_particles,
                seed=seed,
            )
            comparison = compare_tsbm_association_samples(
                sampled.associations, exact
            )
            events = slot_event_marginals(
                sampled.associations, case.model.sequence
            )
            errors = _event_errors(exact_events, events)
            checks = {
                "valid_support": comparison.invalid_sample_fraction == 0.0,
                "total_variation": (
                    comparison.total_variation_distance
                    <= thresholds["total_variation"]
                ),
                "jensen_shannon": (
                    comparison.jensen_shannon_divergence
                    <= thresholds["jensen_shannon"]
                ),
                "exact_mass_covered": (
                    comparison.exact_mass_covered_by_empirical_support
                    >= thresholds["exact_mass_covered"]
                ),
                "credible_set_recall_95": (
                    comparison.credible_set_recall_95
                    >= thresholds["credible_set_recall_95"]
                ),
                "event_marginals": (
                    max(errors.values()) <= thresholds["maximum_event_error"]
                ),
            }
            stages = sampled.stages
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "runtime_seconds": runtime,
                    "comparison": comparison.to_dict(),
                    "event_errors": errors,
                    "maximum_event_error": max(errors.values()),
                    "log_evidence_error": (
                        sampled.log_evidence_estimate - exact_log_evidence
                    ),
                    "n_tempering_steps": sampled.n_tempering_steps,
                    "mean_acceptance_rate": mean(
                        stage.mh_acceptance_rate for stage in stages
                    ),
                    "mean_blocked_slot_acceptance_rate": mean(
                        stage.blocked_slot_acceptance_rate for stage in stages
                    ),
                    "mean_paired_slot_acceptance_rate": mean(
                        stage.paired_slot_acceptance_rate for stage in stages
                    ),
                    "mean_paired_slot_state_change_rate": mean(
                        stage.paired_slot_state_change_rate for stage in stages
                    ),
                    "final_unique_state_count": stages[-1].unique_state_count,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    method_gates = {
        method: all(run["passed"] for run in runs if run["method"] == method)
        for method in methods
    }
    return {
        "regime": asdict(regime),
        "n_slots": case.model.n_slots,
        "n_frames": case.model.sequence.n_frames,
        "detection_cardinalities": list(case.model.sequence.cardinalities),
        "exact_support_size": exact.probabilities.size,
        "exact_log_evidence": exact_log_evidence,
        "finite_sample_baseline": baseline,
        "thresholds": thresholds,
        "runs": runs,
        "blocked_gate_passed": method_gates.get("blocked", False),
        "method_gates": method_gates,
    }


def _histogram(values: np.ndarray, maximum: int) -> np.ndarray:
    return np.bincount(values, minlength=maximum + 1) / values.size


def _pairwise_maximum(arrays: list[np.ndarray]) -> float:
    return max(
        float(np.max(np.abs(left - right)))
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    )


def _pairwise_histogram_tv(histograms: list[np.ndarray]) -> float:
    return max(
        0.5 * float(np.sum(np.abs(left - right)))
        for index, left in enumerate(histograms)
        for right in histograms[index + 1 :]
    )


def _run_large_stability_case(
    *,
    n_particles: int,
    n_repeats: int,
    method: str = "blocked",
    sampling_seed_start: int = 100,
    name: str = "ambiguous_four_slots_eight_frames",
    central_separation: float = 0.0,
    detection_probability: float = 0.70,
    clutter_rate: float = 0.15,
    corruption_seed: int = 5,
    final_mh_sweeps: int = 8,
    target_ess_fraction: float = 0.80,
    stability_threshold: float = 0.10,
) -> dict[str, Any]:
    n_slots = 4
    n_frames = 8
    crossing = MultiSlotCrossingConfig(
        n_slots=n_slots,
        n_frames=n_frames,
        frame_period=0.001,
        half_span=REFERENCE_SPEED_MM_S * 0.001 * (n_frames - 1) / 2.0,
        central_separation=central_separation,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=9)
    truth_mask = np.asarray(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=detection_probability,
            expected_clutter_per_frame=clutter_rate,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=corruption_seed,
    )
    coordinate_std = tuple(
        float(value) for value in np.std(truth.positions, axis=(0, 1))
    )
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=REFERENCE_SPEED_MM_S
    )
    model = SlotExistenceMaternModel(
        detections,
        n_slots=n_slots,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=clutter_rate,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    interval_combinations = model.n_interval_masks**n_slots
    runs: list[dict[str, Any]] = []
    event_summaries: list[SlotEventMarginals] = []
    alive_histograms: list[np.ndarray] = []
    assigned_histograms: list[np.ndarray] = []
    for seed in range(n_repeats):
        sampled, runtime = _sample_case(
            _ScalingCase(
                model=model,
                truth_mask=truth_mask,
                localization_std=crossing.localization_std,
                coordinate_std=calibration.coordinate_std,
                length_scale=calibration.length_scale,
            ),
            method=method,
            n_particles=n_particles,
            seed=sampling_seed_start + seed,
            final_mh_sweeps=final_mh_sweeps,
            target_ess_fraction=target_ess_fraction,
        )
        events = slot_event_marginals(sampled.associations, detections)
        event_summaries.append(events)
        alive_counts = np.count_nonzero(sampled.associations != -2, axis=(1, 2))
        assigned_counts = np.count_nonzero(sampled.associations >= 0, axis=(1, 2))
        alive_histograms.append(_histogram(alive_counts, n_slots * n_frames))
        assigned_histograms.append(_histogram(assigned_counts, n_slots * n_frames))
        runs.append(
            {
                "seed": sampling_seed_start + seed,
                "runtime_seconds": runtime,
                "n_tempering_steps": sampled.n_tempering_steps,
                "mean_acceptance_rate": mean(
                    stage.mh_acceptance_rate for stage in sampled.stages
                ),
                "mean_blocked_slot_acceptance_rate": mean(
                    stage.blocked_slot_acceptance_rate for stage in sampled.stages
                ),
                "mean_paired_slot_acceptance_rate": mean(
                    stage.paired_slot_acceptance_rate for stage in sampled.stages
                ),
                "mean_paired_slot_state_change_rate": mean(
                    stage.paired_slot_state_change_rate for stage in sampled.stages
                ),
                "final_unique_state_count": sampled.stages[-1].unique_state_count,
                "mean_alive_slot_frames": float(np.mean(alive_counts)),
                "mean_assigned_detections": float(np.mean(assigned_counts)),
                "log_evidence_estimate": sampled.log_evidence_estimate,
            }
        )
    cross_run = {
        "maximum_existence_marginal_difference": _pairwise_maximum(
            [events.existence_probabilities for events in event_summaries]
        ),
        "maximum_missed_marginal_difference": _pairwise_maximum(
            [events.missed_probabilities for events in event_summaries]
        ),
        "maximum_clutter_marginal_difference": _pairwise_maximum(
            [events.expected_clutter for events in event_summaries]
        ),
        "maximum_birth_marginal_difference": _pairwise_maximum(
            [events.birth_time_probabilities for events in event_summaries]
        ),
        "maximum_death_marginal_difference": _pairwise_maximum(
            [events.death_time_probabilities for events in event_summaries]
        ),
        "maximum_alive_count_tv": _pairwise_histogram_tv(alive_histograms),
        "maximum_assigned_count_tv": _pairwise_histogram_tv(assigned_histograms),
    }
    threshold = stability_threshold
    checks = {
        name: value <= threshold for name, value in cross_run.items()
    }
    return {
        "validation_level": (
            "independent-run stability only; no exact calibration claim"
        ),
        "name": name,
        "method": method,
        "n_slots": n_slots,
        "n_frames": n_frames,
        "central_separation_mm": central_separation,
        "detection_probability": detection_probability,
        "expected_clutter_per_frame": clutter_rate,
        "corruption_seed": corruption_seed,
        "sampling_seed_start": sampling_seed_start,
        "detection_cardinalities": list(detections.cardinalities),
        "interval_mask_combinations_not_enumerated": interval_combinations,
        "joint_hmm_state_count": 3**n_slots,
        "final_mh_sweeps": final_mh_sweeps,
        "target_ess_fraction": target_ess_fraction,
        "runs": runs,
        "cross_run_metrics": cross_run,
        "stability_threshold": threshold,
        "checks": checks,
        "stability_passed": all(checks.values()),
    }


def _plot_summary(result: dict[str, Any], output_path: Path) -> None:
    exact_cases = result["exact_cases"]
    labels: list[str] = []
    tv_values: list[float] = []
    event_values: list[float] = []
    runtime_values: list[float] = []
    colors: list[str] = []
    for case in exact_cases:
        short = case["regime"]["name"].replace("three_slots_", "")
        for method in ("local", "blocked"):
            selected = [run for run in case["runs"] if run["method"] == method]
            labels.append(f"{short}\n{method}")
            tv_values.append(mean(run["comparison"]["total_variation_distance"] for run in selected))
            event_values.append(mean(run["maximum_event_error"] for run in selected))
            runtime_values.append(mean(run["runtime_seconds"] for run in selected))
            colors.append("#2166ac" if method == "blocked" else "#bdbdbd")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    indices = np.arange(len(labels))
    for axis, values, title in (
        (axes[0, 0], tv_values, "Exact-target total variation"),
        (axes[0, 1], event_values, "Maximum event-marginal error"),
        (axes[1, 0], runtime_values, "Runtime (seconds)"),
    ):
        axis.bar(indices, values, color=colors)
        axis.set_xticks(indices, labels, rotation=15, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    large_metrics = result["large_case"]["cross_run_metrics"]
    large_labels = [
        "existence",
        "missed",
        "clutter",
        "birth",
        "death",
        "alive count",
        "assigned count",
    ]
    stability_threshold = result["large_case"]["stability_threshold"]
    large_values = list(large_metrics.values())
    axes[1, 1].bar(
        np.arange(len(large_labels)),
        large_values,
        color=[
            "#238b45" if value <= stability_threshold else "#cb181d"
            for value in large_values
        ],
    )
    axes[1, 1].axhline(
        stability_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 1].set_xticks(
        np.arange(len(large_labels)), large_labels, rotation=25, ha="right"
    )
    axes[1, 1].set_title("4-slot/8-frame cross-run stability")
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle("WP5 scaling: sequential base and blocked slot proposals")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_wp5_scaling_calibration(
    *,
    output_dir: Path,
    n_particles: int = 2_048,
    n_repeats: int = 3,
    baseline_replicates: int = 100,
) -> dict[str, Any]:
    """Run exact three-slot gates and a larger independent-run stress test."""

    if n_particles < 2 or n_repeats < 2 or baseline_replicates < 1:
        raise ValueError(
            "n_particles must be at least two, n_repeats at least two, and "
            "baseline_replicates positive"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_cases = [
        _run_exact_regime(
            regime,
            n_particles=n_particles,
            n_repeats=n_repeats,
            baseline_replicates=baseline_replicates,
        )
        for regime in EXACT_REGIMES
    ]
    large_case = _run_large_stability_case(
        n_particles=n_particles, n_repeats=n_repeats
    )
    result: dict[str, Any] = {
        "experiment": "wp5_scaling_blocked_birth_death_slots",
        "method": {
            "base_sampler": (
                "exact forward-filtering/backward-sampling on the joint "
                "unborn-alive-dead HMM"
            ),
            "blocked_move": (
                "one slot's complete birth/death interval and all temporal "
                "associations drawn from its exact conditional base density"
            ),
            "metropolis_property": (
                "existence and count terms cancel analytically; acceptance at "
                "beta zero is one, and only the tempered Matern path factor "
                "remains thereafter"
            ),
            "blocked_hybrid_move_probabilities": {
                "whole_frame_association": 0.30,
                "interval_replacement": 0.45,
                "global_slot_swap": 0.10,
                "conditional_base_blocked_slot": 0.15,
            },
            "oracle_used_during_inference": False,
        },
        "configuration": {
            "n_particles": n_particles,
            "n_repeats": n_repeats,
            "baseline_replicates": baseline_replicates,
            "target_ess_fraction": 0.80,
            "mh_sweeps": 8,
            "final_mh_sweeps": 8,
        },
        "exact_cases": exact_cases,
        "large_case": large_case,
        "exact_blocked_gate_passed": all(
            case["blocked_gate_passed"] for case in exact_cases
        ),
        "large_stability_passed": large_case["stability_passed"],
        "validated_ceiling": "three slots and six frames",
        "observed_boundary": (
            "four slots and eight frames run without interval enumeration, "
            "but independent posterior marginals are not stable"
        ),
    }
    result["milestone_passed"] = bool(
        result["exact_blocked_gate_passed"]
        and result["large_stability_passed"]
    )
    with (output_dir / "wp5_scaling_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_summary(result, output_dir / "wp5_scaling_calibration.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-particles", type=int, default=2_048)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--baseline-replicates", type=int, default=100)
    args = parser.parse_args()
    result = run_wp5_scaling_calibration(
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        n_repeats=args.n_repeats,
        baseline_replicates=args.baseline_replicates,
    )
    print(
        json.dumps(
            {
                "exact_blocked_gate_passed": result[
                    "exact_blocked_gate_passed"
                ],
                "large_stability_passed": result["large_stability_passed"],
                "milestone_passed": result["milestone_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
