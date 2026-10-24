import numpy as np
import pytest

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    calibrate_matern_length_scale,
    crossing_design_coordinate_std,
    enumerate_velocity_calibrated_target,
    physical_crossing_config,
)
from sbulm.simulation import simulate_crossing


def test_physical_crossing_uses_mm_and_seconds() -> None:
    config = physical_crossing_config(
        speed_mm_s=8.0,
        frame_rate_hz=1_000.0,
        n_frames=9,
        central_separation_um=10.0,
        localization_std_um=2.0,
    )
    assert config.frame_period == pytest.approx(0.001)
    assert config.half_span == pytest.approx(0.032)
    assert config.central_separation == pytest.approx(0.010)
    assert config.localization_std == pytest.approx(0.002)


@pytest.mark.parametrize(
    ("nu", "factor"),
    [(1.5, 3.0), (2.5, 5.0 / 3.0)],
)
def test_calibration_matches_matern_derivative_variance(nu: float, factor: float) -> None:
    config = physical_crossing_config(localization_std_um=2.0)
    _, detections = simulate_crossing(config, seed=2)
    speed = 7.5
    calibration = calibrate_matern_length_scale(
        detections, rms_speed=speed, nu=nu
    )
    recovered_speed = np.sqrt(
        factor * np.sum(np.asarray(calibration.coordinate_std) ** 2)
    ) / calibration.length_scale
    assert recovered_speed == pytest.approx(speed)


def test_velocity_calibrated_target_is_normalized() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=4)
    posterior, calibration = enumerate_velocity_calibrated_target(
        detections, rms_speed=6.84
    )
    assert posterior.associations.shape == (16, 5, 2)
    assert np.sum(posterior.probabilities) == pytest.approx(1.0)
    assert calibration.length_scale > 0


def test_calibration_rejects_nonphysical_speed() -> None:
    _, detections = simulate_crossing(seed=0)
    with pytest.raises(ValueError, match="rms_speed"):
        calibrate_matern_length_scale(detections, rms_speed=0.0)


def test_frozen_crossing_scale_is_independent_of_noise_and_separation() -> None:
    reference = physical_crossing_config(
        central_separation_um=0.0, localization_std_um=1.0
    )
    difficult = physical_crossing_config(
        central_separation_um=20.0, localization_std_um=20.0
    )
    assert crossing_design_coordinate_std(reference) == pytest.approx(
        crossing_design_coordinate_std(difficult)
    )


def test_fixed_scale_calibration_recovers_requested_speed() -> None:
    coordinate_std = (0.012, 0.018)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84, nu=1.5
    )
    recovered = np.sqrt(3.0 * np.sum(np.asarray(coordinate_std) ** 2))
    recovered /= calibration.length_scale
    assert recovered == pytest.approx(6.84)
