"""Compatibility helpers for the public Smooth-SB particle-tracking notebooks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .data import DetectionSequence, TrajectorySet
from .inference import AssociationPosterior, enumerate_association_support
from .simulation import CrossingConfig, simulate_crossing

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def to_smooth_sb_array(sequence: DetectionSequence) -> FloatArray:
    """Convert constant-cardinality detections to Smooth-SB's ``(T, N, D)`` input.

    Detection order is deliberately preserved. It is arbitrary within each
    frame, so the exported array contains observations rather than hidden
    trajectory identities.
    """

    if not sequence.has_constant_cardinality:
        raise ValueError("Smooth SB requires the same number of observations in every frame")
    if sequence.cardinalities[0] == 0:
        raise ValueError("Smooth SB requires at least one observation per frame")
    if sequence.dimension > 5:
        raise ValueError("the public Smooth-SB implementation supports at most five dimensions")
    return np.stack(sequence.detections).astype(float, copy=False)


def export_smooth_sb_npz(
    sequence: DetectionSequence,
    path: str | Path,
    *,
    truth: TrajectorySet | None = None,
) -> Path:
    """Write a notebook-compatible NPZ plus optional audit-only ground truth.

    The public notebook reads only ``arr_0``. ``times``, ``reference_ids``, and
    ``truth_positions`` are included when available so sampled index paths can
    later be evaluated without relying on the row order as an identity label.
    """

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("the Smooth-SB export path must end in .npz")
    array = to_smooth_sb_array(sequence)
    payload: dict[str, np.ndarray] = {
        "arr_0": array,
        "times": sequence.times,
    }
    if sequence.reference_ids is not None:
        payload["reference_ids"] = np.stack(sequence.reference_ids)
    if truth is not None:
        if truth.times.shape != sequence.times.shape or not np.allclose(
            truth.times, sequence.times
        ):
            raise ValueError("truth and detections must use the same acquisition times")
        if truth.positions.shape != array.shape:
            raise ValueError("truth and detections must have the same (T, N, D) shape")
        payload["truth_positions"] = truth.positions
        payload["truth_particle_ids"] = truth.particle_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


@dataclass(frozen=True)
class SampleComparison:
    """Agreement between sampled association paths and an exact posterior."""

    n_samples: int
    invalid_sample_fraction: float
    total_variation_distance: float
    jensen_shannon_divergence: float
    valid_conditioned_total_variation_distance: float | None
    valid_conditioned_jensen_shannon_divergence: float | None
    exact_map_sample_frequency: float
    empirical_support_size: int


def _jensen_shannon(probability: FloatArray, approximation: FloatArray) -> float:
    midpoint = 0.5 * (probability + approximation)

    def divergence(left: FloatArray, right: FloatArray) -> float:
        positive = left > 0
        return float(np.sum(left[positive] * np.log(left[positive] / right[positive])))

    return 0.5 * divergence(probability, midpoint) + 0.5 * divergence(
        approximation, midpoint
    )


def compare_smooth_sb_samples(
    samples: NDArray[np.integer] | NDArray[np.floating],
    posterior: AssociationPosterior,
) -> SampleComparison:
    """Compare Smooth-SB index-path samples with an enumerable posterior.

    ``samples`` must have shape ``(S, T, N)`` and follow the convention used by
    ``D_sample`` in the upstream one-by-one-matching notebook. Non-permutation
    frames are retained as invalid mass rather than silently discarded.
    """

    raw = np.asarray(samples)
    expected_shape = posterior.associations.shape[1:]
    if raw.ndim != 3 or raw.shape[1:] != expected_shape:
        raise ValueError(f"samples must have shape (S, {expected_shape[0]}, {expected_shape[1]})")
    if raw.shape[0] == 0:
        raise ValueError("at least one Smooth-SB sample is required")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.round(raw)):
        raise ValueError("sampled association indices must be finite integers")
    integer_samples = raw.astype(np.int64)

    n_particles = expected_shape[1]
    expected_permutation = np.arange(n_particles)
    valid = np.all(
        np.sort(integer_samples, axis=2) == expected_permutation[None, None, :],
        axis=(1, 2),
    )
    hypothesis_lookup = {
        association.tobytes(): index
        for index, association in enumerate(posterior.associations)
    }
    empirical = np.zeros(posterior.probabilities.size + 1, dtype=float)
    for sample, is_valid in zip(integer_samples, valid, strict=True):
        index = hypothesis_lookup.get(sample.tobytes()) if is_valid else None
        empirical[-1 if index is None else index] += 1.0
    empirical /= integer_samples.shape[0]
    exact = np.concatenate([posterior.probabilities, np.zeros(1, dtype=float)])
    valid_mass = float(np.sum(empirical[:-1]))
    if valid_mass > 0:
        empirical_given_valid = empirical[:-1] / valid_mass
        valid_tv: float | None = 0.5 * float(
            np.sum(np.abs(posterior.probabilities - empirical_given_valid))
        )
        valid_js: float | None = _jensen_shannon(
            posterior.probabilities, empirical_given_valid
        )
    else:
        valid_tv = None
        valid_js = None

    exact_map = posterior.map_index
    return SampleComparison(
        n_samples=integer_samples.shape[0],
        invalid_sample_fraction=float(empirical[-1]),
        total_variation_distance=0.5 * float(np.sum(np.abs(exact - empirical))),
        jensen_shannon_divergence=_jensen_shannon(exact, empirical),
        valid_conditioned_total_variation_distance=valid_tv,
        valid_conditioned_jensen_shannon_divergence=valid_js,
        exact_map_sample_frequency=float(empirical[exact_map]),
        empirical_support_size=int(np.count_nonzero(empirical[:-1])),
    )


def matern_covariance(
    times: FloatArray,
    *,
    variance: float,
    length_scale: float,
    nu: float,
) -> FloatArray:
    distances = np.abs(times[:, None] - times[None, :])
    if nu == 1.5:
        scaled = np.sqrt(3.0) * distances / length_scale
        correlation = (1.0 + scaled) * np.exp(-scaled)
    elif nu == 2.5:
        scaled = np.sqrt(5.0) * distances / length_scale
        correlation = (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)
    else:
        raise ValueError("the upstream notebook supports only nu=1.5 or nu=2.5")
    return variance * correlation


def enumerate_smooth_sb_target(
    sequence: DetectionSequence,
    *,
    t_end: float = 2.0,
    model_times: FloatArray | None = None,
    nu: float = 1.5,
    length_scale: float = 3.0,
    scale_multiplier: float = 1.0,
    coordinate_std: ArrayLike | None = None,
    relative_jitter: float = 1e-10,
    max_hypotheses: int = 250_000,
) -> AssociationPosterior:
    """Enumerate the discrete Matérn-GP path target used by Smooth SB.

    This is an exact small-problem reference for the upstream notebook's target,
    not the localization-noise posterior used by :mod:`sbulm.inference`. The
    notebook rescales acquisition times uniformly to ``[0, t_end]`` and sets a
    stationary Matérn variance independently for every spatial coordinate.
    Passing ``model_times`` instead preserves physical acquisition times; this
    is required when the length scale has physical time units. Passing
    ``coordinate_std`` freezes the latent spatial scales instead of estimating
    them from each noisy sequence.
    """

    if length_scale <= 0 or scale_multiplier <= 0:
        raise ValueError("length_scale and scale_multiplier must be positive")
    if relative_jitter < 0:
        raise ValueError("relative_jitter cannot be negative")
    associations, trajectories = enumerate_association_support(
        sequence, max_hypotheses=max_hypotheses
    )
    observations = to_smooth_sb_array(sequence)
    if coordinate_std is None:
        coordinate_scales = np.std(observations, axis=(0, 1), ddof=0)
    else:
        coordinate_scales = np.asarray(coordinate_std, dtype=float)
        if coordinate_scales.shape != (sequence.dimension,):
            raise ValueError("coordinate_std must have one value per spatial dimension")
    if np.any(coordinate_scales <= 0) or not np.all(np.isfinite(coordinate_scales)):
        raise ValueError("each coordinate standard deviation must be finite and positive")
    if model_times is None:
        if t_end <= 0:
            raise ValueError("t_end must be positive")
        target_times = np.linspace(0.0, t_end, sequence.n_frames)
    else:
        target_times = np.asarray(model_times, dtype=float)
        if target_times.shape != (sequence.n_frames,):
            raise ValueError("model_times must have one value per acquisition frame")
        if not np.all(np.isfinite(target_times)) or np.any(np.diff(target_times) <= 0):
            raise ValueError("model_times must be finite and strictly increasing")
    precision_factors: list[tuple[np.ndarray, bool]] = []
    for dimension in range(sequence.dimension):
        variance = float(coordinate_scales[dimension] ** 2) * scale_multiplier**2
        covariance = matern_covariance(
            target_times,
            variance=variance,
            length_scale=length_scale,
            nu=nu,
        )
        covariance.flat[:: sequence.n_frames + 1] += relative_jitter * variance
        precision_factors.append(cho_factor(covariance, lower=True, check_finite=False))

    log_scores = np.empty(trajectories.shape[0], dtype=float)
    for hypothesis_index, hypothesis in enumerate(trajectories):
        quadratic = 0.0
        for dimension, factor in enumerate(precision_factors):
            for particle in range(hypothesis.shape[1]):
                path = hypothesis[:, particle, dimension]
                quadratic += float(
                    path @ cho_solve(factor, path, check_finite=False)
                )
        log_scores[hypothesis_index] = -0.5 * quadratic
    normalized = log_scores - logsumexp(log_scores)
    return AssociationPosterior(
        associations=associations,
        log_probabilities=normalized,
        probabilities=np.exp(normalized),
        trajectories=trajectories,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sbulm_crossing.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--central-separation", type=float, default=0.05)
    parser.add_argument("--localization-std", type=float, default=0.075)
    args = parser.parse_args()
    config = CrossingConfig(
        central_separation=args.central_separation,
        localization_std=args.localization_std,
    )
    truth, detections = simulate_crossing(config, seed=args.seed)
    path = export_smooth_sb_npz(detections, args.output, truth=truth)
    print(
        json.dumps(
            {
                "path": str(path),
                "shape": list(to_smooth_sb_array(detections).shape),
                "seed": args.seed,
                "central_separation": args.central_separation,
                "localization_std": args.localization_std,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
