"""Physical calibration helpers for the Smooth-SB Matérn motion prior."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import DetectionSequence
from .inference import AssociationPosterior
from .simulation import CrossingConfig
from .smooth_sb import enumerate_smooth_sb_target, to_smooth_sb_array


@dataclass(frozen=True)
class MaternVelocityCalibration:
    """Matérn parameters whose derivative RMS matches a speed scale.

    All quantities share the units of the supplied sequence. For example, if
    positions are in millimetres and times in seconds, ``rms_speed`` is in
    millimetres per second and ``length_scale`` is in seconds.
    """

    nu: float
    rms_speed: float
    coordinate_std: tuple[float, ...]
    scale_multiplier: float
    length_scale: float

    def to_dict(self) -> dict[str, float | list[float]]:
        output = asdict(self)
        output["coordinate_std"] = list(self.coordinate_std)
        return output


def calibrate_matern_length_scale(
    sequence: DetectionSequence,
    *,
    rms_speed: float,
    nu: float = 1.5,
    scale_multiplier: float = 1.0,
) -> MaternVelocityCalibration:
    r"""Match the stationary Matérn prior's RMS speed to ``rms_speed``.

    For independent coordinates with marginal standard deviations
    :math:`\sigma_d`, the Matérn-3/2 prior satisfies

    :math:`E\|x'(t)\|^2 = 3\sum_d\sigma_d^2/\ell^2`.

    The corresponding factor is :math:`5/3` for Matérn-5/2. The coordinate
    scales deliberately match the upstream notebook: empirical standard
    deviations of all observed points, multiplied by ``scale_multiplier``.
    """

    if scale_multiplier <= 0 or not np.isfinite(scale_multiplier):
        raise ValueError("scale_multiplier must be finite and positive")
    observations = to_smooth_sb_array(sequence)
    coordinate_std = np.std(observations, axis=(0, 1), ddof=0)
    return calibrate_matern_from_coordinate_std(
        coordinate_std,
        rms_speed=rms_speed,
        nu=nu,
        scale_multiplier=scale_multiplier,
    )


def calibrate_matern_from_coordinate_std(
    coordinate_std: tuple[float, ...] | list[float] | np.ndarray,
    *,
    rms_speed: float,
    nu: float = 1.5,
    scale_multiplier: float = 1.0,
) -> MaternVelocityCalibration:
    """Calibrate a Matérn time scale from fixed latent spatial scales."""

    if rms_speed <= 0 or not np.isfinite(rms_speed):
        raise ValueError("rms_speed must be finite and positive")
    if scale_multiplier <= 0 or not np.isfinite(scale_multiplier):
        raise ValueError("scale_multiplier must be finite and positive")
    if nu == 1.5:
        derivative_variance_factor = 3.0
    elif nu == 2.5:
        derivative_variance_factor = 5.0 / 3.0
    else:
        raise ValueError("the upstream notebook supports only nu=1.5 or nu=2.5")

    coordinate_scales = np.asarray(coordinate_std, dtype=float) * scale_multiplier
    if coordinate_scales.ndim != 1 or coordinate_scales.size == 0:
        raise ValueError("coordinate_std must be a non-empty one-dimensional array")
    if np.any(coordinate_scales <= 0) or not np.all(np.isfinite(coordinate_scales)):
        raise ValueError("each coordinate standard deviation must be finite and positive")
    length_scale = float(
        np.sqrt(derivative_variance_factor * np.sum(coordinate_scales**2))
        / rms_speed
    )
    return MaternVelocityCalibration(
        nu=nu,
        rms_speed=float(rms_speed),
        coordinate_std=tuple(float(value) for value in coordinate_scales),
        scale_multiplier=float(scale_multiplier),
        length_scale=length_scale,
    )


def crossing_design_coordinate_std(config: CrossingConfig) -> tuple[float, float]:
    """Return frozen coordinate scales for the centered crossing design.

    Each coordinate contains one linearly moving bubble and one bubble at zero.
    The scale therefore depends on the time window and reference speed, but not
    on a particular noise realization or central separation.
    """

    normalized = np.linspace(-1.0, 1.0, config.n_frames)
    scale = config.half_span * float(np.sqrt(np.mean(normalized**2) / 2.0))
    return (scale, scale)


def enumerate_velocity_calibrated_target(
    sequence: DetectionSequence,
    *,
    rms_speed: float,
    nu: float = 1.5,
    scale_multiplier: float = 1.0,
    relative_jitter: float = 1e-10,
    max_hypotheses: int = 250_000,
) -> tuple[AssociationPosterior, MaternVelocityCalibration]:
    """Enumerate the Matérn target after velocity and time calibration."""

    calibration = calibrate_matern_length_scale(
        sequence,
        rms_speed=rms_speed,
        nu=nu,
        scale_multiplier=scale_multiplier,
    )
    posterior = enumerate_smooth_sb_target(
        sequence,
        model_times=sequence.times,
        nu=nu,
        length_scale=calibration.length_scale,
        scale_multiplier=scale_multiplier,
        relative_jitter=relative_jitter,
        max_hypotheses=max_hypotheses,
    )
    return posterior, calibration


def physical_crossing_config(
    *,
    speed_mm_s: float = 6.84,
    frame_rate_hz: float = 1_000.0,
    n_frames: int = 9,
    central_separation_um: float = 5.0,
    localization_std_um: float = 5.0,
) -> CrossingConfig:
    """Create a crossing in millimetres and seconds from ULM-scale inputs."""

    if speed_mm_s <= 0 or frame_rate_hz <= 0:
        raise ValueError("speed_mm_s and frame_rate_hz must be positive")
    if n_frames < 4:
        raise ValueError("n_frames must be at least four")
    frame_period = 1.0 / frame_rate_hz
    half_span = speed_mm_s * frame_period * (n_frames - 1) / 2.0
    return CrossingConfig(
        n_frames=n_frames,
        frame_period=frame_period,
        half_span=half_span,
        central_separation=central_separation_um / 1_000.0,
        localization_std=localization_std_um / 1_000.0,
    )
