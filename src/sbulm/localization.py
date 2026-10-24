"""Gaussian localization likelihoods for latent unordered ULM trajectories."""

from __future__ import annotations

from itertools import permutations
from math import factorial, log, pi

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .data import DetectionSequence
from .inference import AssociationPosterior, enumerate_association_support
from .smooth_sb import matern_covariance, to_smooth_sb_array

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_localization_std(localization_std: float) -> None:
    if localization_std <= 0 or not np.isfinite(localization_std):
        raise ValueError("localization_std must be finite and positive")


def _frame_permutations(n_particles: int) -> NDArray[np.int64]:
    return np.asarray(tuple(permutations(range(n_particles))), dtype=np.int64)


class MaternLocalizationModel:
    r"""Collapsed Gaussian model for fixed-cardinality ULM associations.

    For a proposed association path, each labelled coordinate follows a
    Matérn Gaussian-process prior and the assigned localization is observed
    with independent Gaussian noise. The latent paths can consequently be
    integrated out when scoring associations and sampled exactly after an
    association has been selected.
    """

    def __init__(
        self,
        sequence: DetectionSequence,
        *,
        localization_std: float,
        coordinate_std: ArrayLike,
        length_scale: float,
        nu: float = 1.5,
        relative_jitter: float = 1e-10,
    ) -> None:
        _validate_localization_std(localization_std)
        if not sequence.has_constant_cardinality:
            raise ValueError("the Matérn localization model requires constant cardinality")
        if length_scale <= 0 or not np.isfinite(length_scale):
            raise ValueError("length_scale must be finite and positive")
        if relative_jitter < 0 or not np.isfinite(relative_jitter):
            raise ValueError("relative_jitter must be finite and non-negative")
        coordinate_scales = np.asarray(coordinate_std, dtype=float)
        if coordinate_scales.shape != (sequence.dimension,):
            raise ValueError("coordinate_std must have one value per spatial dimension")
        if np.any(coordinate_scales <= 0) or not np.all(np.isfinite(coordinate_scales)):
            raise ValueError(
                "each coordinate standard deviation must be finite and positive"
            )

        self.sequence = sequence
        self.localization_std = float(localization_std)
        self.coordinate_std = coordinate_scales
        self.length_scale = float(length_scale)
        self.nu = float(nu)
        self.relative_jitter = float(relative_jitter)
        self.observations = np.stack(sequence.detections)
        self._observed_factors: list[tuple[np.ndarray, bool]] = []
        self._log_normalizers: list[float] = []
        self._conditional_gains: list[FloatArray] = []
        self._conditional_cholesky: list[FloatArray] = []

        identity = np.eye(sequence.n_frames)
        for coordinate_scale in coordinate_scales:
            variance = float(coordinate_scale**2)
            prior_covariance = matern_covariance(
                sequence.times,
                variance=variance,
                length_scale=length_scale,
                nu=nu,
            )
            prior_covariance.flat[:: sequence.n_frames + 1] += (
                relative_jitter * variance
            )
            observed_covariance = prior_covariance + localization_std**2 * identity
            factor = cho_factor(observed_covariance, lower=True, check_finite=False)
            self._observed_factors.append(factor)
            log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
            self._log_normalizers.append(
                sequence.n_frames * log(2.0 * pi) + log_determinant
            )

            gain = cho_solve(factor, prior_covariance.T, check_finite=False).T
            conditional_covariance = prior_covariance - gain @ prior_covariance
            conditional_covariance = 0.5 * (
                conditional_covariance + conditional_covariance.T
            )
            conditional_scale = max(
                float(np.max(np.diag(conditional_covariance))),
                np.finfo(float).tiny,
            )
            conditional_covariance.flat[:: sequence.n_frames + 1] += (
                relative_jitter * conditional_scale
            )
            self._conditional_gains.append(np.asarray(gain, dtype=float))
            self._conditional_cholesky.append(
                np.linalg.cholesky(conditional_covariance)
            )

    @property
    def n_particles(self) -> int:
        return self.sequence.cardinalities[0]

    def _validate_associations(self, associations: ArrayLike) -> IntArray:
        raw = np.asarray(associations)
        expected = (self.sequence.n_frames, self.n_particles)
        if raw.ndim != 3 or raw.shape[1:] != expected:
            raise ValueError(f"associations must have shape (S, {expected[0]}, {expected[1]})")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("associations must contain integer detection indices")
        integer = raw.astype(np.int64, copy=False)
        ordered = np.sort(integer, axis=-1)
        reference = np.arange(self.n_particles, dtype=np.int64)
        if not np.all(ordered == reference):
            raise ValueError("every association frame must be a permutation")
        return integer

    def assigned_detections(self, associations: ArrayLike) -> FloatArray:
        """Return observed detections reordered by batched association paths."""

        integer = self._validate_associations(associations)
        time_indices = np.arange(self.sequence.n_frames)[None, :, None]
        return np.asarray(self.observations[time_indices, integer], dtype=float)

    def log_marginal_density(self, associations: ArrayLike) -> FloatArray:
        """Score association paths after analytically integrating latent paths."""

        assigned = self.assigned_detections(associations)
        n_samples = assigned.shape[0]
        scores = np.zeros(n_samples, dtype=float)
        for dimension, (factor, log_normalizer) in enumerate(
            zip(self._observed_factors, self._log_normalizers, strict=True)
        ):
            coordinates = assigned[..., dimension].transpose(1, 0, 2).reshape(
                self.sequence.n_frames, -1
            )
            solved = cho_solve(factor, coordinates, check_finite=False)
            quadratics = np.sum(coordinates * solved, axis=0).reshape(
                n_samples, self.n_particles
            )
            scores -= 0.5 * (
                np.sum(quadratics, axis=1) + self.n_particles * log_normalizer
            )
        return scores

    def sample_conditional_paths(
        self,
        associations: ArrayLike,
        *,
        rng: np.random.Generator,
    ) -> FloatArray:
        r"""Draw latent paths from :math:`p(x\mid y,a)` for each association."""

        assigned = self.assigned_detections(associations)
        n_samples = assigned.shape[0]
        paths = np.empty_like(assigned)
        for dimension, (gain, cholesky) in enumerate(
            zip(
                self._conditional_gains,
                self._conditional_cholesky,
                strict=True,
            )
        ):
            coordinates = assigned[..., dimension].transpose(1, 0, 2).reshape(
                self.sequence.n_frames, -1
            )
            conditional_mean = gain @ coordinates
            innovations = cholesky @ rng.standard_normal(coordinates.shape)
            samples = (conditional_mean + innovations).reshape(
                self.sequence.n_frames, n_samples, self.n_particles
            )
            paths[..., dimension] = samples.transpose(1, 0, 2)
        return paths

    def conditional_path_moments(
        self, associations: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Return means and coordinate-wise covariances of ``p(x | y, a)``."""

        assigned = self.assigned_detections(associations)
        n_samples = assigned.shape[0]
        means = np.empty_like(assigned)
        covariances = np.empty(
            (self.sequence.dimension, self.sequence.n_frames, self.sequence.n_frames),
            dtype=float,
        )
        for dimension, (gain, cholesky) in enumerate(
            zip(
                self._conditional_gains,
                self._conditional_cholesky,
                strict=True,
            )
        ):
            coordinates = assigned[..., dimension].transpose(1, 0, 2).reshape(
                self.sequence.n_frames, -1
            )
            dimension_means = (gain @ coordinates).reshape(
                self.sequence.n_frames, n_samples, self.n_particles
            )
            means[..., dimension] = dimension_means.transpose(1, 0, 2)
            covariances[dimension] = cholesky @ cholesky.T
        return means, covariances


def unordered_gaussian_localization_potential(
    latent_positions: ArrayLike,
    sequence: DetectionSequence,
    *,
    localization_std: float,
    include_normalization: bool = True,
) -> FloatArray:
    r"""Evaluate a differentiable set likelihood at every acquisition time.

    For latent labelled particles :math:`x_{t,i}` and an unordered detection
    set :math:`Y_t`, the likelihood averages over every one-to-one assignment:

    .. math::

       p(Y_t\mid X_t)=\frac{1}{N!}\sum_{\pi\in S_N}
       \prod_i \mathcal N(y_{t,\pi(i)};x_{t,i},\sigma_{\rm loc}^2 I)

    The returned value is ``-log p(Y_t | X_t)`` with shape ``(..., T)``.
    Leading batch dimensions on ``latent_positions`` are supported. The
    factorial sum is intended for the small fixed-cardinality benchmark; it is
    not a scalable data-association algorithm.
    """

    _validate_localization_std(localization_std)
    if not sequence.has_constant_cardinality:
        raise ValueError("the unordered potential requires constant cardinality")
    observations = to_smooth_sb_array(sequence)
    latent = np.asarray(latent_positions, dtype=float)
    expected = observations.shape
    if latent.ndim < 3 or latent.shape[-3:] != expected:
        raise ValueError(f"latent_positions must end with shape {expected}")
    if not np.all(np.isfinite(latent)):
        raise ValueError("latent_positions must be finite")

    n_particles = sequence.cardinalities[0]
    permutations_array = _frame_permutations(n_particles)
    assigned = observations[:, permutations_array, :]
    residual = latent[..., :, None, :, :] - assigned
    costs = 0.5 * np.sum(residual**2, axis=(-1, -2)) / localization_std**2
    potential = -logsumexp(-costs, axis=-1) + log(factorial(n_particles))
    if include_normalization:
        potential = potential + n_particles * sequence.dimension * log(
            localization_std * np.sqrt(2.0 * pi)
        )
    return np.asarray(potential, dtype=float)


def unordered_gaussian_localization_gradient(
    latent_positions: ArrayLike,
    sequence: DetectionSequence,
    *,
    localization_std: float,
) -> FloatArray:
    """Return the analytical spatial gradient of the unordered potential."""

    _validate_localization_std(localization_std)
    if not sequence.has_constant_cardinality:
        raise ValueError("the unordered potential requires constant cardinality")
    observations = to_smooth_sb_array(sequence)
    latent = np.asarray(latent_positions, dtype=float)
    expected = observations.shape
    if latent.ndim < 3 or latent.shape[-3:] != expected:
        raise ValueError(f"latent_positions must end with shape {expected}")
    if not np.all(np.isfinite(latent)):
        raise ValueError("latent_positions must be finite")

    permutations_array = _frame_permutations(sequence.cardinalities[0])
    assigned = observations[:, permutations_array, :]
    residual = latent[..., :, None, :, :] - assigned
    costs = 0.5 * np.sum(residual**2, axis=(-1, -2)) / localization_std**2
    log_weights = -costs - logsumexp(-costs, axis=-1, keepdims=True)
    weights = np.exp(log_weights)[..., :, :, None, None]
    return np.sum(weights * residual, axis=-3) / localization_std**2


def enumerate_matern_localization_posterior(
    sequence: DetectionSequence,
    *,
    localization_std: float,
    coordinate_std: ArrayLike,
    length_scale: float,
    nu: float = 1.5,
    relative_jitter: float = 1e-10,
    max_hypotheses: int = 250_000,
) -> AssociationPosterior:
    r"""Exactly marginalize latent Matérn paths for every association.

    Under a latent path prior :math:`x_d\sim\mathcal N(0,K_d)` and localization
    model :math:`y_d\mid x_d\sim\mathcal N(x_d,\sigma_{\rm loc}^2I)`, an assigned
    observed path follows :math:`y_d\sim\mathcal N(0,K_d+\sigma_{\rm loc}^2I)`.
    Exact enumeration therefore supplies gold-standard association weights for
    the small benchmark while properly acknowledging localization noise.
    """

    associations, trajectories = enumerate_association_support(
        sequence, max_hypotheses=max_hypotheses
    )
    model = MaternLocalizationModel(
        sequence,
        localization_std=localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        nu=nu,
        relative_jitter=relative_jitter,
    )
    log_scores = model.log_marginal_density(associations)

    normalized = log_scores - logsumexp(log_scores)
    return AssociationPosterior(
        associations=associations,
        log_probabilities=normalized,
        probabilities=np.exp(normalized),
        trajectories=trajectories,
    )
