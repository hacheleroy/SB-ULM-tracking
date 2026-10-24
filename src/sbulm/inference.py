"""Exact association posterior for small fixed-cardinality benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial, log, pi

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .data import DetectionSequence, stack_assigned_detections

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AssociationPosterior:
    associations: IntArray
    log_probabilities: FloatArray
    probabilities: FloatArray
    trajectories: FloatArray

    def __post_init__(self) -> None:
        if self.associations.ndim != 3:
            raise ValueError("associations must have shape (H, T, N)")
        if self.trajectories.ndim != 4:
            raise ValueError("trajectories must have shape (H, T, N, D)")
        n_hypotheses = self.associations.shape[0]
        if self.log_probabilities.shape != (n_hypotheses,):
            raise ValueError("log_probabilities has the wrong shape")
        if self.probabilities.shape != (n_hypotheses,):
            raise ValueError("probabilities has the wrong shape")

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def normalized_entropy(self) -> float:
        positive = self.probabilities[self.probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        return 0.0 if self.probabilities.size <= 1 else entropy / log(self.probabilities.size)

    def credible_set_size(self, mass: float = 0.95) -> int:
        if not 0 < mass <= 1:
            raise ValueError("mass must lie in (0, 1]")
        ordered = np.sort(self.probabilities)[::-1]
        return int(np.searchsorted(np.cumsum(ordered), mass, side="left") + 1)


def _constant_velocity_matrices(
    dt: float, dimension: int, acceleration_std: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    identity = np.eye(dimension)
    transition = np.block([[identity, dt * identity], [np.zeros_like(identity), identity]])
    observation = np.block([identity, np.zeros_like(identity)])
    q = acceleration_std**2
    process = q * np.block(
        [
            [(dt**4 / 4.0) * identity, (dt**3 / 2.0) * identity],
            [(dt**3 / 2.0) * identity, (dt**2) * identity],
        ]
    )
    return transition, observation, process


def _kalman_track_log_likelihood(
    observations: FloatArray,
    times: FloatArray,
    *,
    localization_std: float,
    acceleration_std: float,
    initial_velocity_std: float,
) -> float:
    dimension = observations.shape[1]
    observation_covariance = localization_std**2 * np.eye(dimension)
    mean = np.concatenate([observations[0], np.zeros(dimension)])
    covariance = np.block(
        [
            [observation_covariance, np.zeros((dimension, dimension))],
            [np.zeros((dimension, dimension)), initial_velocity_std**2 * np.eye(dimension)],
        ]
    )
    total = 0.0
    for time_index in range(1, observations.shape[0]):
        dt = float(times[time_index] - times[time_index - 1])
        transition, observation_matrix, process_covariance = _constant_velocity_matrices(
            dt, dimension, acceleration_std
        )
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process_covariance
        innovation = observations[time_index] - observation_matrix @ mean
        innovation_covariance = (
            observation_matrix @ covariance @ observation_matrix.T + observation_covariance
        )
        sign, log_determinant = np.linalg.slogdet(innovation_covariance)
        if sign <= 0:
            return -np.inf
        solved = np.linalg.solve(innovation_covariance, innovation)
        total += -0.5 * (
            dimension * log(2.0 * pi) + log_determinant + float(innovation @ solved)
        )
        gain = (
            np.linalg.solve(
                innovation_covariance,
                observation_matrix @ covariance,
            ).T
        )
        mean = mean + gain @ innovation
        covariance = covariance - gain @ observation_matrix @ covariance
        covariance = 0.5 * (covariance + covariance.T)
    return total


def enumerate_association_support(
    sequence: DetectionSequence,
    *,
    fix_first: bool = True,
    max_hypotheses: int = 250_000,
) -> tuple[IntArray, FloatArray]:
    """Enumerate association arrays and their assigned trajectories.

    By default, the first frame fixes arbitrary particle labels, removing the
    global label symmetry. Set ``fix_first=False`` when a method explicitly
    represents exchangeable boundary marginals, as in the matched TSBM
    benchmark.
    """

    if not sequence.has_constant_cardinality:
        raise ValueError("exact enumeration currently requires constant cardinality")
    n_particles = sequence.cardinalities[0]
    n_variable_frames = sequence.n_frames - 1 if fix_first else sequence.n_frames
    n_hypotheses = factorial(n_particles) ** n_variable_frames
    if n_hypotheses > max_hypotheses:
        raise ValueError(
            f"enumeration would create {n_hypotheses:,} hypotheses; "
            f"increase max_hypotheses deliberately or use an approximate sampler"
        )

    frame_permutations = tuple(permutations(range(n_particles)))
    association_list = []
    trajectory_list = []
    if fix_first:
        prefix = (tuple(range(n_particles)),)
        continuations = product(frame_permutations, repeat=sequence.n_frames - 1)
    else:
        prefix = ()
        continuations = product(frame_permutations, repeat=sequence.n_frames)
    for continuation in continuations:
        association = np.asarray((*prefix, *continuation), dtype=np.int64)
        association_list.append(association)
        trajectory_list.append(stack_assigned_detections(sequence, association))
    return np.stack(association_list), np.stack(trajectory_list)


def enumerate_association_posterior(
    sequence: DetectionSequence,
    *,
    localization_std: float,
    acceleration_std: float = 0.08,
    initial_velocity_std: float = 1.0,
    max_hypotheses: int = 250_000,
) -> AssociationPosterior:
    """Enumerate and normalize every association path for a small problem.

    The first frame fixes arbitrary identity labels. Every later frame is
    permuted, and each complete hypothesis is scored using independent
    constant-velocity Kalman filters. This produces exact discrete posterior
    weights under the stated model.
    """

    if localization_std <= 0 or acceleration_std <= 0 or initial_velocity_std <= 0:
        raise ValueError("all scale parameters must be positive")
    associations, trajectories = enumerate_association_support(
        sequence, max_hypotheses=max_hypotheses
    )
    n_particles = sequence.cardinalities[0]
    log_likelihoods = []
    for hypothesis in trajectories:
        log_likelihood = sum(
            _kalman_track_log_likelihood(
                hypothesis[:, particle],
                sequence.times,
                localization_std=localization_std,
                acceleration_std=acceleration_std,
                initial_velocity_std=initial_velocity_std,
            )
            for particle in range(n_particles)
        )
        log_likelihoods.append(log_likelihood)

    log_likelihood_array = np.asarray(log_likelihoods, dtype=float)
    normalized_logs = log_likelihood_array - logsumexp(log_likelihood_array)
    return AssociationPosterior(
        associations=associations,
        log_probabilities=normalized_logs,
        probabilities=np.exp(normalized_logs),
        trajectories=trajectories,
    )

