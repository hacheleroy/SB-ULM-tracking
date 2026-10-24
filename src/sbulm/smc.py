"""Annealed SMC for the collapsed Matérn ULM localization posterior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .data import DetectionSequence
from .localization import MaternLocalizationModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AnnealedSMCConfig:
    """Numerical controls for collapsed association SMC."""

    n_particles: int = 8_192
    target_ess_fraction: float = 0.80
    mh_sweeps: int = 8
    final_mh_sweeps: int = 8
    frame_move_probability: float = 0.50
    max_tempering_steps: int = 128
    bisection_iterations: int = 48
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least two")
        if not 0 < self.target_ess_fraction < 1:
            raise ValueError("target_ess_fraction must lie in (0, 1)")
        if self.mh_sweeps < 0 or self.final_mh_sweeps < 0:
            raise ValueError("mh_sweeps and final_mh_sweeps must be non-negative")
        if not 0 <= self.frame_move_probability <= 1:
            raise ValueError("frame_move_probability must lie in [0, 1]")
        if self.max_tempering_steps < 1 or self.bisection_iterations < 1:
            raise ValueError("tempering and bisection iteration counts must be positive")


@dataclass(frozen=True)
class SMCStageDiagnostics:
    """Diagnostics recorded before and after one tempering transition."""

    step: int
    beta: float
    beta_increment: float
    effective_sample_size: float
    unique_parent_count: int
    unique_state_count: int
    mh_sweeps: int
    mh_acceptance_rate: float
    frame_acceptance_rate: float
    segment_acceptance_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MaternSMCResult:
    """Unweighted joint posterior draws and annealing diagnostics."""

    associations: IntArray
    latent_paths: FloatArray
    log_marginal_densities: FloatArray
    log_evidence_estimate: float
    stages: tuple[SMCStageDiagnostics, ...]
    config: AnnealedSMCConfig

    def __post_init__(self) -> None:
        n_samples = self.config.n_particles
        if self.associations.ndim != 3 or self.associations.shape[0] != n_samples:
            raise ValueError("associations must contain one path per SMC particle")
        if self.latent_paths.shape[:3] != self.associations.shape:
            raise ValueError("latent_paths and associations have incompatible shapes")
        if self.log_marginal_densities.shape != (n_samples,):
            raise ValueError("log_marginal_densities has the wrong shape")
        if not self.stages or not np.isclose(self.stages[-1].beta, 1.0):
            raise ValueError("the SMC result must terminate at beta=1")

    @property
    def n_tempering_steps(self) -> int:
        return len(self.stages)


def _normalized_weights(log_weights: FloatArray) -> tuple[FloatArray, float]:
    normalizer = float(logsumexp(log_weights))
    weights = np.exp(log_weights - normalizer)
    return weights, normalizer


def effective_sample_size(log_weights: ArrayLike) -> float:
    """Return ESS after normalizing an arbitrary vector of log weights."""

    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_weights must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("log_weights must be finite")
    weights, _ = _normalized_weights(values)
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights: ArrayLike, rng: np.random.Generator) -> IntArray:
    """Draw equally weighted ancestor indices by systematic resampling."""

    probabilities = np.asarray(weights, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    probabilities = probabilities / total
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(probabilities.size)) / probabilities.size
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _incremental_ess(
    log_marginal_densities: FloatArray,
    beta_increment: float,
) -> float:
    return effective_sample_size(beta_increment * log_marginal_densities)


def _next_beta(
    log_marginal_densities: FloatArray,
    current_beta: float,
    *,
    target_ess: float,
    bisection_iterations: int,
) -> float:
    remaining = 1.0 - current_beta
    if _incremental_ess(log_marginal_densities, remaining) >= target_ess:
        return 1.0

    lower = current_beta
    upper = 1.0
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (lower + upper)
        ess = _incremental_ess(
            log_marginal_densities, midpoint - current_beta
        )
        if ess < target_ess:
            upper = midpoint
        else:
            lower = midpoint
    if lower <= current_beta:
        raise RuntimeError("adaptive tempering failed to make positive progress")
    return lower


def _initial_associations(
    sequence: DetectionSequence,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> IntArray:
    n_particles = sequence.cardinalities[0]
    frame_permutations = np.asarray(
        tuple(permutations(range(n_particles))), dtype=np.int64
    )
    draws = rng.integers(
        frame_permutations.shape[0],
        size=(n_samples, sequence.n_frames - 1),
    )
    associations = np.empty(
        (n_samples, sequence.n_frames, n_particles), dtype=np.int64
    )
    associations[:, 0] = np.arange(n_particles)
    associations[:, 1:] = frame_permutations[draws]
    return associations


def _propose_swaps(
    associations: IntArray,
    *,
    frame_move_probability: float,
    rng: np.random.Generator,
) -> tuple[IntArray, NDArray[np.bool_]]:
    n_samples, n_frames, n_particles = associations.shape
    proposed = associations.copy()
    frame_moves = rng.random(n_samples) < frame_move_probability
    starts = rng.integers(1, n_frames, size=n_samples)
    ends = starts.copy()
    segment_rows = np.flatnonzero(~frame_moves)
    if segment_rows.size:
        available_lengths = n_frames - starts[segment_rows]
        ends[segment_rows] += np.floor(
            rng.random(segment_rows.size) * available_lengths
        ).astype(np.int64)
    first_labels = rng.integers(n_particles, size=n_samples)
    offsets = rng.integers(1, n_particles, size=n_samples)
    second_labels = (first_labels + offsets) % n_particles
    for row in range(n_samples):
        interval = slice(starts[row], ends[row] + 1)
        first = first_labels[row]
        second = second_labels[row]
        temporary = proposed[row, interval, first].copy()
        proposed[row, interval, first] = proposed[row, interval, second]
        proposed[row, interval, second] = temporary
    return proposed, frame_moves


def _metropolis_rejuvenation(
    associations: IntArray,
    log_marginal_densities: FloatArray,
    model: MaternLocalizationModel,
    *,
    beta: float,
    n_sweeps: int,
    frame_move_probability: float,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, tuple[float, float, float]]:
    if n_sweeps == 0 or model.n_particles == 1:
        return associations, log_marginal_densities, (0.0, 0.0, 0.0)

    accepted_total = 0
    proposed_total = 0
    accepted_by_move = np.zeros(2, dtype=np.int64)
    proposed_by_move = np.zeros(2, dtype=np.int64)
    current = associations
    current_scores = log_marginal_densities
    for _ in range(n_sweeps):
        proposed, frame_moves = _propose_swaps(
            current,
            frame_move_probability=frame_move_probability,
            rng=rng,
        )
        proposed_scores = model.log_marginal_density(proposed)
        accepted = np.log(rng.random(current.shape[0])) < beta * (
            proposed_scores - current_scores
        )
        current[accepted] = proposed[accepted]
        current_scores[accepted] = proposed_scores[accepted]
        accepted_total += int(np.count_nonzero(accepted))
        proposed_total += accepted.size
        for move_index, selected in enumerate((frame_moves, ~frame_moves)):
            proposed_by_move[move_index] += int(np.count_nonzero(selected))
            accepted_by_move[move_index] += int(
                np.count_nonzero(accepted & selected)
            )

    rates = np.divide(
        accepted_by_move,
        proposed_by_move,
        out=np.zeros(2, dtype=float),
        where=proposed_by_move > 0,
    )
    return (
        current,
        current_scores,
        (
            accepted_total / proposed_total,
            float(rates[0]),
            float(rates[1]),
        ),
    )


def sample_matern_localization_posterior(
    sequence: DetectionSequence,
    *,
    localization_std: float,
    coordinate_std: ArrayLike,
    length_scale: float,
    nu: float = 1.5,
    relative_jitter: float = 1e-10,
    config: AnnealedSMCConfig | None = None,
) -> MaternSMCResult:
    r"""Sample the full fixed-cardinality posterior with collapsed SMC.

    The tempered association targets are

    .. math:: \pi_\beta(a)\propto p(y\mid a)^\beta p_0(a),

    where ``p0`` is uniform over association paths with the first frame used to
    fix global labels. At each temperature, systematic resampling is followed
    by Metropolis swaps of one frame or a contiguous frame segment. Once
    ``beta=1``, latent Matérn paths are drawn exactly from ``p(x | y, a)``.

    Exact enumeration is not called by this routine; it is reserved for the
    small-problem calibration harness.
    """

    controls = AnnealedSMCConfig() if config is None else config
    if not sequence.has_constant_cardinality:
        raise ValueError("collapsed association SMC requires constant cardinality")
    n_frame_permutations = factorial(sequence.cardinalities[0])
    if n_frame_permutations > 100_000:
        raise ValueError("frame permutation support is too large for this benchmark sampler")

    model = MaternLocalizationModel(
        sequence,
        localization_std=localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        nu=nu,
        relative_jitter=relative_jitter,
    )
    rng = np.random.default_rng(controls.seed)
    associations = _initial_associations(
        sequence, n_samples=controls.n_particles, rng=rng
    )
    log_densities = model.log_marginal_density(associations)
    beta = 0.0
    log_evidence = 0.0
    stages: list[SMCStageDiagnostics] = []
    target_ess = controls.target_ess_fraction * controls.n_particles

    for step in range(1, controls.max_tempering_steps + 1):
        next_beta = _next_beta(
            log_densities,
            beta,
            target_ess=target_ess,
            bisection_iterations=controls.bisection_iterations,
        )
        increment = next_beta - beta
        incremental_log_weights = increment * log_densities
        weights, log_weight_sum = _normalized_weights(incremental_log_weights)
        ess = float(1.0 / np.sum(weights**2))
        log_evidence += log_weight_sum - np.log(controls.n_particles)

        ancestors = systematic_resample(weights, rng)
        associations = associations[ancestors].copy()
        log_densities = log_densities[ancestors].copy()
        stage_mh_sweeps = controls.mh_sweeps + (
            controls.final_mh_sweeps if next_beta == 1.0 else 0
        )
        associations, log_densities, acceptance = _metropolis_rejuvenation(
            associations,
            log_densities,
            model,
            beta=next_beta,
            n_sweeps=stage_mh_sweeps,
            frame_move_probability=controls.frame_move_probability,
            rng=rng,
        )
        unique_states = np.unique(
            associations.reshape(controls.n_particles, -1), axis=0
        ).shape[0]
        stages.append(
            SMCStageDiagnostics(
                step=step,
                beta=float(next_beta),
                beta_increment=float(increment),
                effective_sample_size=ess,
                unique_parent_count=int(np.unique(ancestors).size),
                unique_state_count=int(unique_states),
                mh_sweeps=stage_mh_sweeps,
                mh_acceptance_rate=acceptance[0],
                frame_acceptance_rate=acceptance[1],
                segment_acceptance_rate=acceptance[2],
            )
        )
        beta = next_beta
        if beta == 1.0:
            break
    else:
        raise RuntimeError(
            "SMC did not reach beta=1 within max_tempering_steps; "
            "increase the limit or inspect target degeneracy"
        )

    latent_paths = model.sample_conditional_paths(associations, rng=rng)
    return MaternSMCResult(
        associations=associations,
        latent_paths=latent_paths,
        log_marginal_densities=log_densities,
        log_evidence_estimate=float(log_evidence),
        stages=tuple(stages),
        config=controls,
    )
