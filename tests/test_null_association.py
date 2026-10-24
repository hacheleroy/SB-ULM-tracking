from math import log

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.localization import MaternLocalizationModel
from sbulm.null_association import (
    NullAssociationMaternModel,
    enumerate_frame_null_associations,
    enumerate_null_association_posterior,
    enumerate_null_association_support,
    null_and_clutter_marginals,
    reference_null_association,
)
from sbulm.simulation import (
    DetectionCorruptionConfig,
    corrupt_detections_with_misses_and_clutter,
    simulate_crossing,
)
from sbulm.smooth_sb import matern_covariance

ROI_LOWER = (-0.04, -0.04)
ROI_UPPER = (0.04, 0.04)


def _corrupted_problem() -> tuple[object, object, tuple[float, ...], float]:
    config = physical_crossing_config(
        n_frames=5,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=3)
    detections = corrupt_detections_with_misses_and_clutter(
        clean,
        DetectionCorruptionConfig(
            detection_probability=0.70,
            expected_clutter_per_frame=0.75,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            max_clutter_per_frame=2,
        ),
        seed=0,
    )
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(coordinate_std, rms_speed=6.84)
    return config, detections, calibration.coordinate_std, calibration.length_scale


def test_frame_null_association_support_sizes_and_injectivity() -> None:
    assert enumerate_frame_null_associations(2, 0).shape == (1, 2)
    assert enumerate_frame_null_associations(2, 1).shape == (3, 2)
    assert enumerate_frame_null_associations(2, 2).shape == (7, 2)
    states = enumerate_frame_null_associations(2, 3)
    assert states.shape == (13, 2)
    assert all(row[0] < 0 or row[1] < 0 or row[0] != row[1] for row in states)


def test_corruption_uses_clean_anchor_and_auditable_clutter_ids() -> None:
    _, detections, _, _ = _corrupted_problem()
    np.testing.assert_array_equal(detections.reference_ids[0], np.array([0, 1]))
    all_ids = np.concatenate(detections.reference_ids)
    assert set(all_ids).issubset({-1, 0, 1})
    assert np.count_nonzero(all_ids == -1) > 0
    truth = reference_null_association(detections, n_particles=2)
    np.testing.assert_array_equal(truth[0], np.array([0, 1]))
    assert np.count_nonzero(truth < 0) > 0


def test_full_association_gp_score_matches_fixed_cardinality_model() -> None:
    config = physical_crossing_config(
        n_frames=5,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, detections = simulate_crossing(config, seed=4)
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(coordinate_std, rms_speed=6.84)
    fixed = MaternLocalizationModel(
        detections,
        localization_std=config.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )
    partial = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=0.8,
        expected_clutter_per_frame=0.5,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    association = np.repeat(np.arange(2)[None, None, :], 5, axis=1)
    np.testing.assert_allclose(
        partial.log_marginal_density(association),
        fixed.log_marginal_density(association),
        atol=1e-10,
    )
    expected_count = 10 * log(0.8) - 5 * 0.5
    assert partial.log_count_density(association)[0] == pytest.approx(expected_count)


def test_exact_null_posterior_and_marginals_are_normalized() -> None:
    config, detections, coordinate_std, length_scale = _corrupted_problem()
    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.70,
        expected_clutter_per_frame=0.75,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    support = enumerate_null_association_support(model)
    posterior = enumerate_null_association_posterior(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.70,
        expected_clutter_per_frame=0.75,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    assert posterior.associations.shape == support.shape
    assert logsumexp(posterior.log_probabilities) == pytest.approx(0.0)
    null, clutter = null_and_clutter_marginals(
        posterior.associations,
        detections,
        probabilities=posterior.probabilities,
    )
    assert null.shape == (detections.n_frames, 2)
    assert clutter.shape == (detections.n_frames,)
    assert np.all((null >= -1e-12) & (null <= 1 + 1e-12))
    assert np.all(
        (clutter >= -1e-12)
        & (clutter <= np.asarray(detections.cardinalities) + 1e-12)
    )


def test_partial_conditional_mean_matches_direct_gp_conditioning() -> None:
    config, detections, coordinate_std, length_scale = _corrupted_problem()
    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.70,
        expected_clutter_per_frame=0.75,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    association = reference_null_association(detections, n_particles=2)[None]
    means = model.conditional_path_means(association)

    particle = 0
    dimension = 0
    observed_times = np.flatnonzero(association[0, :, particle] >= 0)
    observed_values = np.asarray(
        [
            detections.detections[time_index][
                association[0, time_index, particle], dimension
            ]
            for time_index in observed_times
        ]
    )
    covariance = matern_covariance(
        detections.times,
        variance=coordinate_std[dimension] ** 2,
        length_scale=length_scale,
        nu=1.5,
    )
    covariance.flat[:: detections.n_frames + 1] += (
        model.relative_jitter * coordinate_std[dimension] ** 2
    )
    observed_covariance = covariance[np.ix_(observed_times, observed_times)]
    observed_covariance.flat[:: observed_times.size + 1] += config.localization_std**2
    expected = covariance[:, observed_times] @ np.linalg.solve(
        observed_covariance, observed_values
    )
    np.testing.assert_allclose(
        means[0, :, particle, dimension], expected, atol=1e-14
    )


def test_model_rejects_duplicate_non_null_assignment() -> None:
    config, detections, coordinate_std, length_scale = _corrupted_problem()
    model = NullAssociationMaternModel(
        detections,
        n_particles=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.70,
        expected_clutter_per_frame=0.75,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    invalid = np.full((1, detections.n_frames, 2), -1, dtype=np.int64)
    invalid[0, 0] = (0, 1)
    frame = next(index for index, size in enumerate(detections.cardinalities) if size > 0)
    invalid[0, frame] = 0
    with pytest.raises(ValueError, match="injective"):
        model.log_joint_density(invalid)
