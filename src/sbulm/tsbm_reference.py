"""Exact small-problem reference target matched to TSBM's Brownian bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import log, pi

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .data import DetectionSequence
from .inference import AssociationPosterior, enumerate_association_support


@dataclass(frozen=True)
class TSBMSampleComparison:
    """Distributional comparison of inferred association paths."""

    n_samples: int
    invalid_sample_fraction: float
    total_variation_distance: float
    jensen_shannon_divergence: float
    exact_map_sample_frequency: float
    empirical_support_size: int
    exact_mass_covered_by_empirical_support: float
    credible_set_recall_95: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteSampleBaseline:
    """Expected empirical-distribution error from a finite exact sample."""

    n_samples: int
    n_replicates: int
    total_variation_median: float
    total_variation_q05: float
    total_variation_q95: float
    jensen_shannon_median: float
    jensen_shannon_q05: float
    jensen_shannon_q95: float
    credible_set_recall_95_median: float
    credible_set_recall_95_q05: float
    credible_set_recall_95_q95: float
    exact_mass_covered_median: float
    exact_mass_covered_q05: float
    exact_mass_covered_q95: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationProbabilityComparison:
    """Comparison of an estimated association distribution with the target."""

    n_path_samples: int
    total_variation_distance: float
    jensen_shannon_divergence: float
    exact_map_mass: float
    estimated_map_mass: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _distribution_distances(
    exact: np.ndarray, empirical: np.ndarray
) -> tuple[float, float]:
    midpoint = 0.5 * (exact + empirical)

    def divergence(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0
        return float(
            np.sum(left[positive] * np.log(left[positive] / right[positive]))
        )

    total_variation = 0.5 * float(np.sum(np.abs(exact - empirical)))
    jensen_shannon = 0.5 * (
        divergence(exact, midpoint) + divergence(empirical, midpoint)
    )
    return total_variation, jensen_shannon


def finite_sample_baseline(
    posterior: AssociationPosterior,
    *,
    n_samples: int,
    n_replicates: int = 500,
    seed: int = 0,
) -> FiniteSampleBaseline:
    """Estimate the error floor produced by drawing finitely from the target."""

    if n_samples <= 0 or n_replicates <= 0:
        raise ValueError("n_samples and n_replicates must be positive")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_samples, posterior.probabilities, size=n_replicates)
    empirical = counts / n_samples
    total_variations = np.empty(n_replicates, dtype=float)
    jensen_shannons = np.empty(n_replicates, dtype=float)
    for index, frequencies in enumerate(empirical):
        total_variations[index], jensen_shannons[index] = _distribution_distances(
            posterior.probabilities, frequencies
        )
    observed = counts > 0
    order = np.argsort(posterior.probabilities)[::-1]
    credible_indices = order[: posterior.credible_set_size(0.95)]
    credible_set_recalls = np.mean(observed[:, credible_indices], axis=1)
    exact_masses_covered = observed @ posterior.probabilities
    tv_quantiles = np.quantile(total_variations, [0.05, 0.5, 0.95])
    js_quantiles = np.quantile(jensen_shannons, [0.05, 0.5, 0.95])
    recall_quantiles = np.quantile(credible_set_recalls, [0.05, 0.5, 0.95])
    mass_quantiles = np.quantile(exact_masses_covered, [0.05, 0.5, 0.95])
    return FiniteSampleBaseline(
        n_samples=n_samples,
        n_replicates=n_replicates,
        total_variation_median=float(tv_quantiles[1]),
        total_variation_q05=float(tv_quantiles[0]),
        total_variation_q95=float(tv_quantiles[2]),
        jensen_shannon_median=float(js_quantiles[1]),
        jensen_shannon_q05=float(js_quantiles[0]),
        jensen_shannon_q95=float(js_quantiles[2]),
        credible_set_recall_95_median=float(recall_quantiles[1]),
        credible_set_recall_95_q05=float(recall_quantiles[0]),
        credible_set_recall_95_q95=float(recall_quantiles[2]),
        exact_mass_covered_median=float(mass_quantiles[1]),
        exact_mass_covered_q05=float(mass_quantiles[0]),
        exact_mass_covered_q95=float(mass_quantiles[2]),
    )


def collapse_global_label_symmetry(
    posterior: AssociationPosterior,
) -> AssociationPosterior:
    """Quotient out arbitrary global particle labels using the first frame."""

    canonical_associations = np.empty_like(posterior.associations)
    canonical_trajectories = np.empty_like(posterior.trajectories)
    for index, association in enumerate(posterior.associations):
        particle_order = np.argsort(association[0])
        canonical_associations[index] = association[:, particle_order]
        canonical_trajectories[index] = posterior.trajectories[index][
            :, particle_order
        ]
    support, first_indices, inverse = np.unique(
        canonical_associations, axis=0, return_index=True, return_inverse=True
    )
    probabilities = np.zeros(support.shape[0], dtype=float)
    np.add.at(probabilities, inverse, posterior.probabilities)
    return AssociationPosterior(
        associations=support,
        log_probabilities=np.log(probabilities),
        probabilities=probabilities,
        trajectories=canonical_trajectories[first_indices],
    )


def canonicalize_global_particle_labels(associations: np.ndarray) -> np.ndarray:
    """Relabel each sampled path so particle labels match frame-zero detections."""

    associations = np.asarray(associations)
    if associations.ndim != 3:
        raise ValueError("associations must have shape (S, T, N)")
    canonical = np.empty_like(associations)
    for index, association in enumerate(associations):
        canonical[index] = association[:, np.argsort(association[0])]
    return canonical


def collapse_global_label_probabilities(
    associations: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate arbitrary association masses after quotienting global labels."""

    associations = np.asarray(associations)
    probabilities = np.asarray(probabilities, dtype=float)
    if associations.ndim != 3 or probabilities.shape != (associations.shape[0],):
        raise ValueError("associations/probabilities have incompatible shapes")
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and non-negative")
    canonical = canonicalize_global_particle_labels(associations)
    support, inverse = np.unique(canonical, axis=0, return_inverse=True)
    collapsed = np.zeros(support.shape[0], dtype=float)
    np.add.at(collapsed, inverse, probabilities)
    return support, collapsed


def brownian_bridge_covariance(
    times: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    diffusion_std: float,
) -> np.ndarray:
    """Return covariance of a scalar Brownian bridge at interior times."""

    times = np.asarray(times, dtype=float)
    duration = end_time - start_time
    if duration <= 0 or diffusion_std <= 0:
        raise ValueError("duration and diffusion_std must be positive")
    relative = times - start_time
    if np.any(relative <= 0) or np.any(relative >= duration):
        raise ValueError("bridge evaluation times must lie strictly inside the boundaries")
    return diffusion_std**2 * (
        np.minimum(relative[:, None], relative[None, :])
        - relative[:, None] * relative[None, :] / duration
    )


def enumerate_tsbm_brownian_localization_posterior(
    sequence: DetectionSequence,
    *,
    localization_std: float,
    diffusion_std: float,
    relative_jitter: float = 1e-10,
    max_hypotheses: int = 250_000,
) -> AssociationPosterior:
    r"""Enumerate the exact association target represented by the TSBM test.

    The first and final unordered detection sets define exchangeable, discrete
    boundary marginals. Conditional on each ordered endpoint pair, the latent
    coordinates follow a Brownian bridge with diffusion ``diffusion_std``.
    Every interior localization follows

    .. math:: y_t\mid x_t\sim\mathcal N(x_t,\sigma_{\rm loc}^2 I).

    Both endpoint permutations are enumerated. The Brownian transition kernel
    and the integrated intermediate likelihood define a finite endpoint kernel.
    Log-domain Sinkhorn scaling then enforces the prescribed uniform initial
    and final permutation marginals. Conditional association weights within
    each endpoint pair retain their exact Gaussian-integral probabilities.

    This target assumes that the TSBM potential is ``sigma**2`` times the
    negative log likelihood, because the twisted reference weights a discrete
    potential as ``exp(-V / sigma**2)``.
    """

    if localization_std <= 0 or not np.isfinite(localization_std):
        raise ValueError("localization_std must be finite and positive")
    if diffusion_std <= 0 or not np.isfinite(diffusion_std):
        raise ValueError("diffusion_std must be finite and positive")
    if relative_jitter < 0 or not np.isfinite(relative_jitter):
        raise ValueError("relative_jitter must be finite and non-negative")
    if sequence.n_frames < 3:
        raise ValueError("at least one interior observation time is required")

    associations, trajectories = enumerate_association_support(
        sequence,
        fix_first=False,
        max_hypotheses=max_hypotheses,
    )
    interior_times = sequence.times[1:-1]
    bridge_covariance = brownian_bridge_covariance(
        interior_times,
        start_time=float(sequence.times[0]),
        end_time=float(sequence.times[-1]),
        diffusion_std=diffusion_std,
    )
    observed_covariance = bridge_covariance + localization_std**2 * np.eye(
        interior_times.size
    )
    jitter_scale = max(
        float(np.max(np.diag(observed_covariance))), np.finfo(float).tiny
    )
    observed_covariance.flat[:: interior_times.size + 1] += (
        relative_jitter * jitter_scale
    )
    factor = cho_factor(observed_covariance, lower=True, check_finite=False)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    log_normalizer = interior_times.size * log(2.0 * pi) + log_determinant

    duration = float(sequence.times[-1] - sequence.times[0])
    fraction = ((interior_times - sequence.times[0]) / duration)[:, None]
    log_scores = np.empty(trajectories.shape[0], dtype=float)
    for hypothesis_index, hypothesis in enumerate(trajectories):
        score = 0.0
        for particle in range(hypothesis.shape[1]):
            endpoint_mean = (
                (1.0 - fraction) * hypothesis[0, particle]
                + fraction * hypothesis[-1, particle]
            )
            residual = hypothesis[1:-1, particle] - endpoint_mean
            for dimension in range(hypothesis.shape[2]):
                coordinate = residual[:, dimension]
                quadratic = float(
                    coordinate
                    @ cho_solve(factor, coordinate, check_finite=False)
                )
                score -= 0.5 * (quadratic + log_normalizer)
        log_scores[hypothesis_index] = score

    initial_permutations, initial_ids = np.unique(
        associations[:, 0], axis=0, return_inverse=True
    )
    final_permutations, final_ids = np.unique(
        associations[:, -1], axis=0, return_inverse=True
    )
    n_initial = initial_permutations.shape[0]
    n_final = final_permutations.shape[0]
    group_log_partition = np.full((n_initial, n_final), -np.inf, dtype=float)
    log_transition = np.full_like(group_log_partition, -np.inf)
    for initial_index in range(n_initial):
        for final_index in range(n_final):
            selected = (initial_ids == initial_index) & (final_ids == final_index)
            group_log_partition[initial_index, final_index] = logsumexp(
                log_scores[selected]
            )
            representative = int(np.flatnonzero(selected)[0])
            displacement = (
                trajectories[representative, -1] - trajectories[representative, 0]
            )
            log_transition[initial_index, final_index] = -0.5 * float(
                np.sum(displacement**2)
            ) / (diffusion_std**2 * duration)

    log_kernel = log_transition + group_log_partition
    target_initial = np.full(n_initial, 1.0 / n_initial)
    target_final = np.full(n_final, 1.0 / n_final)
    log_u = np.zeros(n_initial, dtype=float)
    log_v = np.zeros(n_final, dtype=float)
    for _ in range(10_000):
        previous = log_v.copy()
        log_u = np.log(target_initial) - logsumexp(
            log_kernel + log_v[None, :], axis=1
        )
        log_v = np.log(target_final) - logsumexp(
            log_kernel + log_u[:, None], axis=0
        )
        if np.max(np.abs(log_v - previous)) < 1e-13:
            break
    log_endpoint_coupling = log_u[:, None] + log_kernel + log_v[None, :]
    endpoint_coupling = np.exp(log_endpoint_coupling)
    if not (
        np.allclose(endpoint_coupling.sum(axis=1), target_initial, atol=1e-11)
        and np.allclose(endpoint_coupling.sum(axis=0), target_final, atol=1e-11)
    ):
        raise RuntimeError("endpoint Sinkhorn scaling did not converge")

    normalized = np.empty_like(log_scores)
    for hypothesis_index in range(log_scores.size):
        initial_index = initial_ids[hypothesis_index]
        final_index = final_ids[hypothesis_index]
        normalized[hypothesis_index] = (
            log_endpoint_coupling[initial_index, final_index]
            + log_scores[hypothesis_index]
            - group_log_partition[initial_index, final_index]
        )
    normalized -= logsumexp(normalized)
    return AssociationPosterior(
        associations=associations,
        log_probabilities=normalized,
        probabilities=np.exp(normalized),
        trajectories=trajectories,
    )


def endpoint_permutation_marginals(
    posterior: AssociationPosterior,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact probability masses for each endpoint permutation."""

    permutations = np.unique(posterior.associations[:, 0], axis=0)
    initial = np.asarray(
        [
            posterior.probabilities[
                np.all(posterior.associations[:, 0] == permutation, axis=1)
            ].sum()
            for permutation in permutations
        ]
    )
    final = np.asarray(
        [
            posterior.probabilities[
                np.all(posterior.associations[:, -1] == permutation, axis=1)
            ].sum()
            for permutation in permutations
        ]
    )
    return initial, final


def endpoint_permutation_joint(
    posterior: AssociationPosterior,
) -> np.ndarray:
    """Return the joint probability table of initial and final permutations."""

    initial_permutations = np.unique(posterior.associations[:, 0], axis=0)
    final_permutations = np.unique(posterior.associations[:, -1], axis=0)
    joint = np.empty(
        (initial_permutations.shape[0], final_permutations.shape[0]), dtype=float
    )
    for initial_index, initial in enumerate(initial_permutations):
        for final_index, final in enumerate(final_permutations):
            selected = np.all(posterior.associations[:, 0] == initial, axis=1)
            selected &= np.all(posterior.associations[:, -1] == final, axis=1)
            joint[initial_index, final_index] = posterior.probabilities[selected].sum()
    return joint


def sample_exact_tsbm_paths(
    sequence: DetectionSequence,
    posterior: AssociationPosterior,
    *,
    localization_std: float,
    diffusion_std: float,
    n_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent paths from the finite exact matched TSBM target.

    The discrete association hypothesis is sampled first. Conditional on it,
    each particle coordinate has the conjugate Gaussian posterior obtained by
    combining its Brownian bridge with the assigned localization observations.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if localization_std <= 0 or diffusion_std <= 0:
        raise ValueError("localization_std and diffusion_std must be positive")
    interior_times = sequence.times[1:-1]
    prior_covariance = brownian_bridge_covariance(
        interior_times,
        start_time=float(sequence.times[0]),
        end_time=float(sequence.times[-1]),
        diffusion_std=diffusion_std,
    )
    observed_covariance = prior_covariance + localization_std**2 * np.eye(
        interior_times.size
    )
    gain = np.linalg.solve(observed_covariance, prior_covariance).T
    posterior_covariance = prior_covariance - gain @ prior_covariance
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    cholesky = np.linalg.cholesky(
        posterior_covariance
        + np.finfo(float).eps * np.eye(posterior_covariance.shape[0])
    )

    rng = np.random.default_rng(seed)
    hypothesis_indices = rng.choice(
        posterior.probabilities.size, size=n_samples, p=posterior.probabilities
    )
    assigned = posterior.trajectories[hypothesis_indices]
    duration = float(sequence.times[-1] - sequence.times[0])
    fraction = ((interior_times - sequence.times[0]) / duration)[None, :, None, None]
    prior_mean = (1.0 - fraction) * assigned[:, :1] + fraction * assigned[:, -1:]
    residual = assigned[:, 1:-1] - prior_mean
    posterior_mean = prior_mean + np.einsum("kl,slnd->sknd", gain, residual)
    standard_normal = rng.normal(size=posterior_mean.shape)
    noise = np.einsum("kl,slnd->sknd", cholesky, standard_normal)

    paths = np.empty_like(assigned)
    paths[:, 0] = assigned[:, 0]
    paths[:, -1] = assigned[:, -1]
    paths[:, 1:-1] = posterior_mean + noise
    return paths, posterior.associations[hypothesis_indices]


def canonicalize_continuous_paths(
    paths: np.ndarray, associations: np.ndarray
) -> np.ndarray:
    """Apply the association-derived global canonical labeling to paths."""

    paths = np.asarray(paths, dtype=float)
    associations = np.asarray(associations)
    if paths.ndim != 4 or associations.ndim != 3:
        raise ValueError("paths and associations must have shapes (S,T,N,D)/(S,T,N)")
    if paths.shape[:3] != associations.shape:
        raise ValueError("paths and associations have incompatible shapes")
    canonical = np.empty_like(paths)
    for index, association in enumerate(associations):
        canonical[index] = paths[index][:, np.argsort(association[0])]
    return canonical


def sliced_wasserstein_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    """Return an equal-sample sliced Wasserstein-1 distance."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("left and right must have the same shape (S, D)")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, left.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    left_projected = np.sort(left @ directions.T, axis=0)
    right_projected = np.sort(right @ directions.T, axis=0)
    return float(np.mean(np.abs(left_projected - right_projected)))


def observation_set_rms(
    paths: np.ndarray, sequence: DetectionSequence
) -> np.ndarray:
    """Return optimal unordered-set RMS residuals with shape ``(S, T)``."""

    paths = np.asarray(paths, dtype=float)
    associations = nearest_detection_associations(paths, sequence)
    observations = np.stack(sequence.detections)
    frame_indices = np.arange(sequence.n_frames)[None, :, None]
    assigned = observations[frame_indices, associations]
    squared = np.sum((paths - assigned) ** 2, axis=(-1, -2))
    return np.sqrt(squared / sequence.cardinalities[0])


def nearest_detection_associations(
    sampled_paths: np.ndarray,
    sequence: DetectionSequence,
) -> np.ndarray:
    """Project continuous labelled-particle paths onto detection permutations."""

    paths = np.asarray(sampled_paths, dtype=float)
    observations = np.stack(sequence.detections)
    expected = (sequence.n_frames, sequence.cardinalities[0], sequence.dimension)
    if paths.ndim != 4 or paths.shape[1:] != expected:
        raise ValueError(f"sampled_paths must have shape (S, {expected})")
    if not np.all(np.isfinite(paths)):
        raise ValueError("sampled paths must be finite")
    frame_permutations = np.asarray(
        tuple(permutations(range(sequence.cardinalities[0]))), dtype=np.int64
    )
    assigned = observations[:, frame_permutations, :]
    squared_distances = np.sum(
        (paths[:, :, None, :, :] - assigned[None, :, :, :, :]) ** 2,
        axis=(-1, -2),
    )
    best = np.argmin(squared_distances, axis=-1)
    return frame_permutations[best]


def rao_blackwellized_association_probabilities(
    sampled_paths: np.ndarray,
    sequence: DetectionSequence,
    support: np.ndarray,
    *,
    localization_std: float,
    batch_size: int = 256,
) -> np.ndarray:
    r"""Estimate association masses from continuous paths without hard assignment.

    For every sampled latent path, the interior-frame permutation variable is
    marginalized under the unordered Gaussian likelihood. Boundary labels are
    assigned to their nearest prescribed endpoint permutation. Averaging these
    conditional probabilities is the Rao--Blackwellized estimator of the
    discrete association posterior represented by the continuous sampler.
    """

    paths = np.asarray(sampled_paths, dtype=float)
    support = np.asarray(support)
    expected = (sequence.n_frames, sequence.cardinalities[0], sequence.dimension)
    if paths.ndim != 4 or paths.shape[1:] != expected:
        raise ValueError(f"sampled_paths must have shape (S, {expected})")
    if support.ndim != 3 or support.shape[1:] != expected[:2]:
        raise ValueError("support has an incompatible association shape")
    if paths.shape[0] == 0 or batch_size <= 0:
        raise ValueError("at least one path and a positive batch_size are required")
    if localization_std <= 0 or not np.isfinite(localization_std):
        raise ValueError("localization_std must be finite and positive")
    if not sequence.has_constant_cardinality:
        raise ValueError("Rao--Blackwellization currently requires fixed cardinality")

    n_particles = sequence.cardinalities[0]
    frame_permutations = np.asarray(
        tuple(permutations(range(n_particles))), dtype=np.int64
    )
    permutation_lookup = {
        permutation.tobytes(): index
        for index, permutation in enumerate(frame_permutations)
    }
    support_ids = np.empty(support.shape[:2], dtype=np.int64)
    for hypothesis_index, association in enumerate(support.astype(np.int64)):
        for frame_index, permutation in enumerate(association):
            try:
                support_ids[hypothesis_index, frame_index] = permutation_lookup[
                    permutation.tobytes()
                ]
            except KeyError as error:
                raise ValueError("support contains a non-permutation frame") from error

    observations = np.stack(sequence.detections)
    assigned = observations[:, frame_permutations, :]
    accumulated = np.zeros(support.shape[0], dtype=float)
    for start in range(0, paths.shape[0], batch_size):
        path_batch = paths[start : start + batch_size]
        logits = -0.5 * np.sum(
            (path_batch[:, :, None, :, :] - assigned[None, :, :, :, :]) ** 2,
            axis=(-1, -2),
        ) / localization_std**2
        frame_log_probabilities = logits - logsumexp(logits, axis=-1, keepdims=True)
        log_contributions = np.zeros(
            (path_batch.shape[0], support.shape[0]), dtype=float
        )
        for frame_index in range(1, sequence.n_frames - 1):
            log_contributions += frame_log_probabilities[:, frame_index][
                :, support_ids[:, frame_index]
            ]

        initial_ids = np.argmax(logits[:, 0], axis=1)
        final_ids = np.argmax(logits[:, -1], axis=1)
        valid_endpoints = support_ids[None, :, 0] == initial_ids[:, None]
        valid_endpoints &= support_ids[None, :, -1] == final_ids[:, None]
        log_contributions[~valid_endpoints] = -np.inf
        contributions = np.exp(log_contributions)
        if not np.allclose(contributions.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("support does not exhaust the conditional associations")
        accumulated += contributions.sum(axis=0)
    return accumulated / paths.shape[0]


def compare_association_probabilities(
    estimated: np.ndarray,
    posterior: AssociationPosterior,
    *,
    n_path_samples: int,
) -> AssociationProbabilityComparison:
    """Compare estimated and exact association probability vectors."""

    estimated = np.asarray(estimated, dtype=float)
    if estimated.shape != posterior.probabilities.shape:
        raise ValueError("estimated probabilities have the wrong shape")
    if n_path_samples <= 0:
        raise ValueError("n_path_samples must be positive")
    if np.any(estimated < 0) or not np.all(np.isfinite(estimated)):
        raise ValueError("estimated probabilities must be finite and non-negative")
    if not np.isclose(estimated.sum(), 1.0, atol=1e-10):
        raise ValueError("estimated probabilities must sum to one")
    total_variation, jensen_shannon = _distribution_distances(
        posterior.probabilities, estimated
    )
    return AssociationProbabilityComparison(
        n_path_samples=n_path_samples,
        total_variation_distance=total_variation,
        jensen_shannon_divergence=jensen_shannon,
        exact_map_mass=float(posterior.probabilities[posterior.map_index]),
        estimated_map_mass=float(estimated[posterior.map_index]),
    )


def compare_tsbm_association_samples(
    samples: np.ndarray,
    posterior: AssociationPosterior,
) -> TSBMSampleComparison:
    """Compare sampled permutation paths with the exact matched target."""

    associations = np.asarray(samples)
    expected_shape = posterior.associations.shape[1:]
    if associations.ndim != 3 or associations.shape[1:] != expected_shape:
        raise ValueError(
            f"samples must have shape (S, {expected_shape[0]}, {expected_shape[1]})"
        )
    if associations.shape[0] == 0:
        raise ValueError("at least one sample is required")
    lookup = {
        association.tobytes(): index
        for index, association in enumerate(posterior.associations)
    }
    empirical = np.zeros(posterior.probabilities.size + 1, dtype=float)
    for association in associations.astype(np.int64, copy=False):
        index = lookup.get(association.tobytes())
        empirical[-1 if index is None else index] += 1.0
    empirical /= associations.shape[0]
    exact = np.concatenate([posterior.probabilities, np.zeros(1)])
    total_variation, jensen_shannon = _distribution_distances(exact, empirical)
    observed = empirical[:-1] > 0
    order = np.argsort(posterior.probabilities)[::-1]
    credible_size = posterior.credible_set_size(0.95)
    credible_indices = order[:credible_size]
    return TSBMSampleComparison(
        n_samples=associations.shape[0],
        invalid_sample_fraction=float(empirical[-1]),
        total_variation_distance=total_variation,
        jensen_shannon_divergence=jensen_shannon,
        exact_map_sample_frequency=float(empirical[posterior.map_index]),
        empirical_support_size=int(np.count_nonzero(empirical[:-1])),
        exact_mass_covered_by_empirical_support=float(
            posterior.probabilities[observed].sum()
        ),
        credible_set_recall_95=float(np.mean(observed[credible_indices])),
    )
