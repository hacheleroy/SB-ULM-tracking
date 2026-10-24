import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.null_association import (
    NullAssociationMaternModel,
    enumerate_null_association_posterior,
    null_and_clutter_marginals,
)
from sbulm.null_smc import sample_null_association_posterior
from sbulm.simulation import (
    DetectionCorruptionConfig,
    corrupt_detections_with_misses_and_clutter,
    simulate_crossing,
)
from sbulm.smc import AnnealedSMCConfig
from sbulm.tsbm_reference import (
    compare_tsbm_association_samples,
    finite_sample_baseline,
)


def _problem() -> tuple[object, object, tuple[float, ...], float]:
    config = physical_crossing_config(
        n_frames=5,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=5)
    detections = corrupt_detections_with_misses_and_clutter(
        clean,
        DetectionCorruptionConfig(
            detection_probability=0.72,
            expected_clutter_per_frame=0.55,
            roi_lower=(-0.04, -0.04),
            roi_upper=(0.04, 0.04),
            max_clutter_per_frame=2,
        ),
        seed=11,
    )
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(coordinate_std, rms_speed=6.84)
    return config, detections, calibration.coordinate_std, calibration.length_scale


def test_null_association_smc_recovers_exact_posterior() -> None:
    config, detections, coordinate_std, length_scale = _problem()
    exact = enumerate_null_association_posterior(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.55,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )
    controls = AnnealedSMCConfig(
        n_particles=4_096,
        target_ess_fraction=0.80,
        mh_sweeps=10,
        final_mh_sweeps=10,
        seed=27,
    )
    result = sample_null_association_posterior(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.55,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
        config=controls,
    )
    comparison = compare_tsbm_association_samples(result.associations, exact)
    finite_floor = finite_sample_baseline(
        exact, n_samples=controls.n_particles, n_replicates=200, seed=6
    )
    assert comparison.invalid_sample_fraction == 0.0
    assert comparison.total_variation_distance <= max(0.05, 3.0 * finite_floor.total_variation_q95)
    assert comparison.exact_mass_covered_by_empirical_support >= 0.95
    assert result.latent_paths.shape == (*result.associations.shape, 2)
    assert np.all(np.isfinite(result.latent_paths))
    assert result.stages[-1].beta == 1.0

    exact_null, exact_clutter = null_and_clutter_marginals(
        exact.associations,
        detections,
        probabilities=exact.probabilities,
    )
    sampled_null, sampled_clutter = null_and_clutter_marginals(result.associations, detections)
    assert np.max(np.abs(sampled_null - exact_null)) < 0.04
    assert np.max(np.abs(sampled_clutter - exact_clutter)) < 0.05

    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.55,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )
    exact_log_evidence = float(logsumexp(model.log_joint_density(exact.associations)))
    assert result.log_evidence_estimate == pytest.approx(exact_log_evidence, abs=0.15)


def test_count_only_base_is_normalized() -> None:
    config, detections, coordinate_std, length_scale = _problem()
    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.55,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )
    exact = enumerate_null_association_posterior(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.55,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )
    base_logs = model.log_base_probability(exact.associations)
    assert logsumexp(base_logs) == pytest.approx(0.0, abs=1e-12)
    expected = (
        logsumexp(model.log_count_density(exact.associations, include_poisson_void=False))
        - detections.n_frames * model.expected_clutter_per_frame
    )
    assert model.base_log_normalizer == pytest.approx(expected)
