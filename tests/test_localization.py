import numpy as np
import pytest

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.localization import (
    enumerate_matern_localization_posterior,
    unordered_gaussian_localization_gradient,
    unordered_gaussian_localization_potential,
)
from sbulm.simulation import simulate_crossing
from sbulm.smooth_sb import enumerate_smooth_sb_target, to_smooth_sb_array


def test_unordered_potential_is_invariant_to_particle_and_detection_order() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=3)
    latent = to_smooth_sb_array(detections) + 0.001
    reference = unordered_gaussian_localization_potential(
        latent, detections, localization_std=config.localization_std
    )
    permuted_latent = latent[:, ::-1]
    particle_permuted = unordered_gaussian_localization_potential(
        permuted_latent, detections, localization_std=config.localization_std
    )
    reordered_detections = type(detections)(
        times=detections.times,
        detections=tuple(frame[::-1] for frame in detections.detections),
    )
    detection_permuted = unordered_gaussian_localization_potential(
        latent, reordered_detections, localization_std=config.localization_std
    )
    np.testing.assert_allclose(particle_permuted, reference)
    np.testing.assert_allclose(detection_permuted, reference)


def test_unordered_potential_gradient_matches_finite_difference() -> None:
    config = physical_crossing_config(n_frames=4)
    _, detections = simulate_crossing(config, seed=1)
    latent = to_smooth_sb_array(detections) + 0.002
    analytical = unordered_gaussian_localization_gradient(
        latent, detections, localization_std=config.localization_std
    )
    epsilon = 1e-7
    numerical = np.empty_like(latent)
    for index in np.ndindex(latent.shape):
        forward = latent.copy()
        backward = latent.copy()
        forward[index] += epsilon
        backward[index] -= epsilon
        forward_value = unordered_gaussian_localization_potential(
            forward, detections, localization_std=config.localization_std
        ).sum()
        backward_value = unordered_gaussian_localization_potential(
            backward, detections, localization_std=config.localization_std
        ).sum()
        numerical[index] = (forward_value - backward_value) / (2.0 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-5)


def test_marginalized_posterior_is_normalized_and_softens_noisy_case() -> None:
    config = physical_crossing_config(
        n_frames=5, central_separation_um=5.0, localization_std_um=20.0
    )
    _, detections = simulate_crossing(config, seed=2)
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    marginalized = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )
    hard = enumerate_smooth_sb_target(
        detections,
        model_times=detections.times,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )
    assert np.sum(marginalized.probabilities) == pytest.approx(1.0)
    assert marginalized.normalized_entropy > hard.normalized_entropy


def test_marginalized_posterior_converges_to_hard_target_at_tiny_noise() -> None:
    config = physical_crossing_config(n_frames=5, localization_std_um=1.0)
    _, detections = simulate_crossing(config, seed=0)
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    hard = enumerate_smooth_sb_target(
        detections,
        model_times=detections.times,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )
    marginalized = enumerate_matern_localization_posterior(
        detections,
        localization_std=1e-9,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
    )
    np.testing.assert_allclose(
        marginalized.probabilities, hard.probabilities, rtol=2e-5, atol=1e-9
    )
