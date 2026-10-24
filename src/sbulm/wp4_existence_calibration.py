"""Calibrate explicit birth/death-slot SMC against enumerable ULM targets."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.special import logsumexp

from .calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from .existence_smc import BirthDeathSMCConfig, sample_slot_existence_posterior
from .simulation import (
    DetectionCorruptionConfig,
    simulate_crossing,
    simulate_slot_existence_observations,
)
from .slot_existence import (
    BirthDeathPrior,
    SlotEventMarginals,
    SlotExistenceMaternModel,
    SlotExistencePosterior,
    enumerate_slot_existence_posterior,
    reference_slot_association,
    slot_event_marginals,
)
from .tsbm_reference import compare_tsbm_association_samples, sliced_wasserstein_distance

ROI_LOWER = (-0.04, -0.04)
ROI_UPPER = (0.04, 0.04)
REFERENCE_SPEED_MM_S = 6.84
GATE_NUMERICAL_TOLERANCE = 1e-12
DEFAULT_BIRTH_DEATH_PRIOR = BirthDeathPrior(
    initial_probability=0.50,
    birth_probability=0.25,
    survival_probability=0.80,
)


@dataclass(frozen=True)
class ExistenceCalibrationRegime:
    """Frozen truth and corruption parameters for one exact-target case."""

    name: str
    truth_mask: tuple[tuple[bool, bool], ...]
    separation_um: float
    detection_probability: float
    expected_clutter_per_frame: float
    corruption_seed: int


CALIBRATION_REGIMES = (
    ExistenceCalibrationRegime(
        "persistent_near_clean",
        ((True, True), (True, True), (True, True), (True, True)),
        20.0,
        0.95,
        0.05,
        0,
    ),
    ExistenceCalibrationRegime(
        "delayed_birth",
        ((True, False), (True, False), (True, True), (True, True)),
        0.0,
        0.80,
        0.15,
        1,
    ),
    ExistenceCalibrationRegime(
        "early_death",
        ((True, True), (True, True), (False, True), (False, True)),
        0.0,
        0.80,
        0.15,
        2,
    ),
    ExistenceCalibrationRegime(
        "turnover_with_misses_and_clutter",
        ((True, False), (True, True), (True, True), (False, True)),
        0.0,
        0.70,
        0.55,
        3,
    ),
)


@dataclass(frozen=True)
class _CalibrationCase:
    model: SlotExistenceMaternModel
    exact: SlotExistencePosterior
    reference_association: np.ndarray
    truth_mask: np.ndarray
    coordinate_std: tuple[float, ...]
    length_scale: float
    localization_std: float


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    return mean(values), 0.0 if len(values) == 1 else stdev(values) / np.sqrt(len(values))


def _make_case(regime: ExistenceCalibrationRegime, *, data_seed: int = 0) -> _CalibrationCase:
    config = physical_crossing_config(
        n_frames=4,
        central_separation_um=regime.separation_um,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=data_seed)
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
    calibration = calibrate_matern_from_coordinate_std(
        crossing_design_coordinate_std(config), rms_speed=REFERENCE_SPEED_MM_S
    )
    kwargs = {
        "n_slots": 2,
        "localization_std": config.localization_std,
        "coordinate_std": calibration.coordinate_std,
        "length_scale": calibration.length_scale,
        "detection_probability": regime.detection_probability,
        "expected_clutter_per_frame": regime.expected_clutter_per_frame,
        "roi_lower": ROI_LOWER,
        "roi_upper": ROI_UPPER,
        "birth_death_prior": DEFAULT_BIRTH_DEATH_PRIOR,
    }
    model = SlotExistenceMaternModel(detections, **kwargs)
    exact = enumerate_slot_existence_posterior(detections, **kwargs)
    return _CalibrationCase(
        model=model,
        exact=exact,
        reference_association=reference_slot_association(detections, truth_mask),
        truth_mask=truth_mask,
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
        return float(np.sum(left[positive] * np.log(left[positive] / right[positive])))

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
    output: dict[str, float | int] = {
        "n_samples": n_samples,
        "n_replicates": n_replicates,
    }
    for name, values in (
        ("total_variation", tv),
        ("jensen_shannon", js),
        ("credible_set_recall_95", recall),
        ("exact_mass_covered", mass),
    ):
        quantiles = np.quantile(values, [0.05, 0.5, 0.95])
        output[f"{name}_q05"] = float(quantiles[0])
        output[f"{name}_median"] = float(quantiles[1])
        output[f"{name}_q95"] = float(quantiles[2])
    return output


def _max_event_errors(
    exact: SlotEventMarginals, sampled: SlotEventMarginals
) -> dict[str, float]:
    return {
        "max_existence_probability_error": float(
            np.max(
                np.abs(
                    sampled.existence_probabilities - exact.existence_probabilities
                )
            )
        ),
        "max_missed_probability_error": float(
            np.max(np.abs(sampled.missed_probabilities - exact.missed_probabilities))
        ),
        "max_expected_clutter_error": float(
            np.max(np.abs(sampled.expected_clutter - exact.expected_clutter))
        ),
        "max_birth_time_probability_error": float(
            np.max(
                np.abs(
                    sampled.birth_time_probabilities - exact.birth_time_probabilities
                )
            )
        ),
        "max_death_time_probability_error": float(
            np.max(
                np.abs(
                    sampled.death_time_probabilities - exact.death_time_probabilities
                )
            )
        ),
    }


def _association_and_event_metrics(
    case: _CalibrationCase,
    sampled_associations: np.ndarray,
    *,
    baseline_seed: int,
    baseline_replicates: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    comparison = compare_tsbm_association_samples(sampled_associations, case.exact)
    finite_baseline = _finite_sample_baseline(
        case.exact,
        n_samples=sampled_associations.shape[0],
        n_replicates=baseline_replicates,
        seed=baseline_seed,
    )
    exact_events = slot_event_marginals(
        case.exact.associations,
        case.model.sequence,
        probabilities=case.exact.probabilities,
    )
    sampled_events = slot_event_marginals(
        sampled_associations, case.model.sequence
    )
    errors = _max_event_errors(exact_events, sampled_events)
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
        "max_existence_probability_error": 0.05,
        "max_missed_probability_error": 0.05,
        "max_expected_clutter_error": 0.05,
        "max_birth_time_probability_error": 0.05,
        "max_death_time_probability_error": 0.05,
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
        **{
            name: value <= thresholds[name] + GATE_NUMERICAL_TOLERANCE
            for name, value in errors.items()
        },
    }
    metrics = {
        **comparison.to_dict(),
        **errors,
        "finite_sample_baseline": finite_baseline,
        "exact_events": {
            "existence_probabilities": exact_events.existence_probabilities.tolist(),
            "missed_probabilities": exact_events.missed_probabilities.tolist(),
            "expected_clutter": exact_events.expected_clutter.tolist(),
            "birth_time_probabilities": exact_events.birth_time_probabilities.tolist(),
            "death_time_probabilities": exact_events.death_time_probabilities.tolist(),
        },
        "sampled_events": {
            "existence_probabilities": sampled_events.existence_probabilities.tolist(),
            "missed_probabilities": sampled_events.missed_probabilities.tolist(),
            "expected_clutter": sampled_events.expected_clutter.tolist(),
            "birth_time_probabilities": sampled_events.birth_time_probabilities.tolist(),
            "death_time_probabilities": sampled_events.death_time_probabilities.tolist(),
        },
        "strict_all_credible_states_observed": comparison.credible_set_recall_95 == 1.0,
    }
    return metrics, thresholds, checks


def _path_embedding(paths: np.ndarray, associations: np.ndarray, scale: float) -> np.ndarray:
    masks = associations != -2
    filled = np.nan_to_num(paths, nan=0.0)
    mask_feature = scale * masks[..., None]
    return np.concatenate([filled, mask_feature], axis=-1).reshape(paths.shape[0], -1)


def _exact_joint_draws(
    case: _CalibrationCase, *, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        case.exact.probabilities.size, size=n_samples, p=case.exact.probabilities
    )
    associations = case.exact.associations[indices]
    paths = case.model.sample_conditional_paths(associations, rng=rng)
    return paths, associations


def _reference_metrics(case: _CalibrationCase) -> dict[str, Any]:
    matches = np.all(
        case.exact.associations == case.reference_association[None], axis=(1, 2)
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
    return {
        "truth_mask": case.truth_mask.tolist(),
        "reference_hypothesis_mass": float(case.exact.probabilities[reference_index]),
        "reference_hypothesis_rank": rank,
        "reference_in_95pct_credible_set": int(reference_index in credible),
        "realized_missed_detections": int(
            np.count_nonzero(case.reference_association == -1)
        ),
        "realized_false_positives": int(
            sum(np.count_nonzero(frame_ids < 0) for frame_ids in reference_ids)
        ),
    }


def _run_sampler(
    case: _CalibrationCase,
    *,
    n_particles: int,
    target_ess_fraction: float,
    mh_sweeps: int,
    final_mh_sweeps: int,
    smc_seed: int,
) -> tuple[Any, float]:
    controls = BirthDeathSMCConfig(
        n_particles=n_particles,
        target_ess_fraction=target_ess_fraction,
        mh_sweeps=mh_sweeps,
        final_mh_sweeps=final_mh_sweeps,
        seed=smc_seed,
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
    case = _make_case(regime)
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

    exact_paths, exact_associations = _exact_joint_draws(
        case, n_samples=n_particles, seed=20_000 + smc_seed
    )
    second_paths, second_associations = _exact_joint_draws(
        case, n_samples=n_particles, seed=30_000 + smc_seed
    )
    embedding_scale = float(np.mean(case.coordinate_std))
    path_distance = sliced_wasserstein_distance(
        _path_embedding(result.latent_paths, result.associations, embedding_scale),
        _path_embedding(exact_paths, exact_associations, embedding_scale),
        n_projections=n_projections,
        seed=40_000 + smc_seed,
    )
    baseline_path_distance = sliced_wasserstein_distance(
        _path_embedding(exact_paths, exact_associations, embedding_scale),
        _path_embedding(second_paths, second_associations, embedding_scale),
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
    thresholds["absolute_log_evidence_error"] = 0.20
    checks["absolute_log_evidence_error"] = (
        abs(evidence_error) <= 0.20 + GATE_NUMERICAL_TOLERANCE
    )
    return {
        "regime": asdict(regime),
        "smc_seed": smc_seed,
        "runtime_seconds": runtime,
        "support_size": int(case.exact.probabilities.size),
        "detection_cardinalities": list(case.model.sequence.cardinalities),
        "association_and_events": metrics,
        "continuous": {
            "masked_path_sliced_wasserstein_mm": path_distance,
            "finite_sample_sliced_wasserstein_mm": baseline_path_distance,
        },
        "normalizing_constant": {
            "smc_log_evidence": result.log_evidence_estimate,
            "exact_log_evidence": exact_log_evidence,
            "error": evidence_error,
        },
        "reference": _reference_metrics(case),
        "smc": {
            "n_tempering_steps": result.n_tempering_steps,
            "mean_mh_acceptance_rate": mean(
                stage.mh_acceptance_rate for stage in result.stages
            ),
            "mean_association_acceptance_rate": mean(
                stage.association_acceptance_rate for stage in result.stages
            ),
            "mean_interval_acceptance_rate": mean(
                stage.interval_acceptance_rate for stage in result.stages
            ),
            "mean_slot_swap_acceptance_rate": mean(
                stage.slot_swap_acceptance_rate for stage in result.stages
            ),
            "stages": [stage.to_dict() for stage in result.stages],
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_tasks(tasks: list[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_evaluate_gate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate_gate_task, tasks, chunksize=1))


def _aggregate_gate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for regime in CALIBRATION_REGIMES:
        selected = [item for item in runs if item["regime"]["name"] == regime.name]
        metric_values = {
            "total_variation": [
                item["association_and_events"]["total_variation_distance"]
                for item in selected
            ],
            "max_existence_probability_error": [
                item["association_and_events"]["max_existence_probability_error"]
                for item in selected
            ],
            "max_birth_time_probability_error": [
                item["association_and_events"]["max_birth_time_probability_error"]
                for item in selected
            ],
            "max_death_time_probability_error": [
                item["association_and_events"]["max_death_time_probability_error"]
                for item in selected
            ],
            "max_expected_clutter_error": [
                item["association_and_events"]["max_expected_clutter_error"]
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
            "detection_cardinalities": selected[0]["detection_cardinalities"],
            "realized_missed_detections": selected[0]["reference"][
                "realized_missed_detections"
            ],
            "realized_false_positives": selected[0]["reference"][
                "realized_false_positives"
            ],
            "all_runs_passed": all(item["passed"] for item in selected),
        }
        for metric, values in metric_values.items():
            average, standard_error = _mean_and_standard_error(values)
            summary[f"mean_{metric}"] = average
            summary[f"se_{metric}"] = standard_error
        summaries.append(summary)
    return summaries


def _gate_component_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    distribution_event_path_checks = (
        "total_variation",
        "jensen_shannon",
        "credible_set_recall_95",
        "max_existence_probability_error",
        "max_missed_probability_error",
        "max_expected_clutter_error",
        "max_birth_time_probability_error",
        "max_death_time_probability_error",
        "path_sliced_wasserstein",
    )
    support_checks = ("credible_set_recall_95", "exact_mass_covered")
    pooled_evidence_errors: dict[str, float] = {}
    for regime in CALIBRATION_REGIMES:
        selected = [item for item in runs if item["regime"]["name"] == regime.name]
        pooled_log_evidence = float(
            logsumexp(
                [item["normalizing_constant"]["smc_log_evidence"] for item in selected]
            )
            - np.log(len(selected))
        )
        exact_log_evidence = float(
            selected[0]["normalizing_constant"]["exact_log_evidence"]
        )
        pooled_evidence_errors[regime.name] = pooled_log_evidence - exact_log_evidence
    return {
        "distribution_event_path_checks_passed": all(
            run["checks"][name]
            for run in runs
            for name in distribution_event_path_checks
        ),
        "finite_sample_support_checks_passed": all(
            run["checks"][name] for run in runs for name in support_checks
        ),
        "individual_evidence_checks_passed": all(
            run["checks"]["absolute_log_evidence_error"] for run in runs
        ),
        "pooled_evidence_errors_by_regime": pooled_evidence_errors,
        "pooled_evidence_checks_passed": all(
            abs(error) <= 0.20 + GATE_NUMERICAL_TOLERANCE
            for error in pooled_evidence_errors.values()
        ),
    }


def _plot_gate_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    labels = [item["regime"].replace("_", " ") for item in summaries]
    indices = np.arange(len(labels))
    panels = (
        ("mean_total_variation", "Association + mask TV", None),
        ("mean_max_existence_probability_error", "Existence error", 0.05),
        ("mean_max_birth_time_probability_error", "Birth-time error", 0.05),
        ("mean_max_death_time_probability_error", "Death-time error", 0.05),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    colors = [
        "#238b45" if item["all_runs_passed"] else "#cb181d" for item in summaries
    ]
    for axis, (metric, title, reference) in zip(axes.flat, panels, strict=True):
        axis.bar(indices, [item[metric] for item in summaries], color=colors)
        if reference is not None:
            axis.axhline(reference, linestyle="--", color="black", linewidth=1)
        axis.set_xticks(indices, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        handles=(
            Patch(color="#238b45", label="all strict checks pass"),
            Patch(color="#cb181d", label="at least one strict check fails"),
        ),
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    figure.suptitle("WP4 explicit birth/death slots: exact-target calibration")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def run_wp4_existence_calibration(
    *,
    output_dir: Path,
    n_particles: int = 8_192,
    n_smc_repeats: int = 3,
    target_ess_fraction: float = 0.80,
    mh_sweeps: int = 10,
    final_mh_sweeps: int = 10,
    n_projections: int = 128,
    baseline_replicates: int = 300,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the four-regime exact calibration gate."""

    positive_counts = (
        n_particles,
        n_smc_repeats,
        mh_sweeps,
        n_projections,
        baseline_replicates,
        workers,
    )
    if any(value < 1 for value in positive_counts):
        raise ValueError("particle, repeat, sweep, projection, baseline and worker counts must be positive")
    if final_mh_sweeps < 0:
        raise ValueError("final_mh_sweeps must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
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
    runs = _run_tasks(tasks, workers)
    summaries = _aggregate_gate_runs(runs)
    first_case = _make_case(CALIBRATION_REGIMES[0])
    result: dict[str, Any] = {
        "experiment": "wp4_explicit_birth_death_slot_smc_calibration",
        "method": {
            "maximum_slots": 2,
            "association_states": {
                "-2": "slot absent",
                "-1": "slot exists but is missed",
                ">=0": "assigned detection index",
            },
            "existence_constraint": "one contiguous alive interval per slot, including never alive",
            "unassigned_detections": "uniform Poisson clutter",
            "base_distribution": "exact normalized existence-prior times count density",
            "mutation": (
                "reversible whole-frame association replacement, birth/death interval "
                "replacement with exact Hastings correction, and global slot swaps"
            ),
            "label_treatment": "exchangeable labelled slots; no first-frame anchor",
            "uses_exact_enumeration_during_inference": False,
        },
        "observation_model": {
            "n_frames": 4,
            "localization_noise_std_um": 5.0,
            "roi_lower_mm": list(ROI_LOWER),
            "roi_upper_mm": list(ROI_UPPER),
            "simulator_clutter_cap": 1,
            "poisson_set_density_includes_factorial": False,
        },
        "existence_prior": asdict(DEFAULT_BIRTH_DEATH_PRIOR),
        "motion_prior": {
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
            "target_ess_fraction": target_ess_fraction,
            "mh_sweeps": mh_sweeps,
            "final_mh_sweeps": final_mh_sweeps,
            "n_projections": n_projections,
            "baseline_replicates": baseline_replicates,
        },
        "gate_definition": {
            "association_and_mask_tv": "at most max(0.05, 3 * exact-draw MC q95)",
            "association_and_mask_js": "at most max(0.01, 3 * exact-draw MC q95)",
            "credible_set_recall_95": "at least the fifth percentile of equal-size exact draws",
            "exact_mass_covered": "at least max(0.95, exact-draw fifth percentile)",
            "event_marginal_errors": 0.05,
            "masked_path_sliced_wasserstein": "at most max(1 um, 3 * exact-draw baseline)",
            "absolute_log_evidence_error": 0.20,
            "aggregate_rule": "every repeat in every frozen regime must pass",
        },
        "gate_runs": runs,
        "gate_summary": summaries,
        "gate_components": _gate_component_summary(runs),
        "gate_passed": all(item["all_runs_passed"] for item in summaries),
    }
    with (output_dir / "wp4_existence_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    _plot_gate_summary(summaries, output_dir / "wp4_existence_calibration.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-particles", type=int, default=8_192)
    parser.add_argument("--n-smc-repeats", type=int, default=3)
    parser.add_argument("--target-ess-fraction", type=float, default=0.80)
    parser.add_argument("--mh-sweeps", type=int, default=10)
    parser.add_argument("--final-mh-sweeps", type=int, default=10)
    parser.add_argument("--n-projections", type=int, default=128)
    parser.add_argument("--baseline-replicates", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_wp4_existence_calibration(
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        n_smc_repeats=args.n_smc_repeats,
        target_ess_fraction=args.target_ess_fraction,
        mh_sweeps=args.mh_sweeps,
        final_mh_sweeps=args.final_mh_sweeps,
        n_projections=args.n_projections,
        baseline_replicates=args.baseline_replicates,
        workers=args.workers,
    )
    print(json.dumps({"gate_passed": result["gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
