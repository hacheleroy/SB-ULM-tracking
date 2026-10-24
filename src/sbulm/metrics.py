"""Metrics for deterministic tracks and association posteriors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .data import DetectionSequence, TrajectorySet
from .inference import AssociationPosterior

IntArray = NDArray[np.int64]


def assigned_reference_ids(sequence: DetectionSequence, associations: IntArray) -> IntArray:
    if sequence.reference_ids is None:
        raise ValueError("reference IDs are required for association metrics")
    return np.stack(
        [ids[index] for ids, index in zip(sequence.reference_ids, associations, strict=True)]
    )


def correct_association_mask(sequence: DetectionSequence, associations: IntArray) -> np.ndarray:
    assigned_ids = assigned_reference_ids(sequence, associations)
    return assigned_ids == assigned_ids[0][None, :]


def association_accuracy(sequence: DetectionSequence, associations: IntArray) -> float:
    mask = correct_association_mask(sequence, associations)
    return float(mask[1:].mean())


def identity_switches(sequence: DetectionSequence, associations: IntArray) -> int:
    assigned_ids = assigned_reference_ids(sequence, associations)
    return int(np.sum(assigned_ids[1:] != assigned_ids[:-1]))


def trajectory_rmse(
    truth: TrajectorySet, sequence: DetectionSequence, associations: IntArray, trajectories: np.ndarray
) -> float:
    if sequence.reference_ids is None:
        raise ValueError("reference IDs are required to align trajectories")
    initial_ids = sequence.reference_ids[0][associations[0]]
    truth_indices = [int(np.flatnonzero(truth.particle_ids == item)[0]) for item in initial_ids]
    aligned_truth = truth.positions[:, truth_indices]
    return float(np.sqrt(np.mean((trajectories - aligned_truth) ** 2)))


def true_hypothesis_mask(sequence: DetectionSequence, posterior: AssociationPosterior) -> np.ndarray:
    return np.asarray(
        [
            bool(np.all(correct_association_mask(sequence, association)))
            for association in posterior.associations
        ]
    )


def true_in_credible_set(
    sequence: DetectionSequence,
    posterior: AssociationPosterior,
    mass: float,
) -> bool:
    """Return whether a reference-consistent association is in a credible set."""

    if not 0 < mass <= 1:
        raise ValueError("mass must lie in (0, 1]")
    mask = true_hypothesis_mask(sequence, posterior)
    order = np.argsort(posterior.probabilities)[::-1]
    cumulative = np.cumsum(posterior.probabilities[order])
    size = int(np.searchsorted(cumulative, mass, side="left") + 1)
    return bool(np.any(mask[order[:size]]))


def posterior_reference_metrics(
    sequence: DetectionSequence, posterior: AssociationPosterior
) -> dict[str, float | int]:
    mask = true_hypothesis_mask(sequence, posterior)
    true_mass = float(np.sum(posterior.probabilities[mask]))
    order = np.argsort(posterior.probabilities)[::-1]
    true_ranks = np.flatnonzero(mask[order])
    rank = int(true_ranks[0] + 1) if true_ranks.size else posterior.probabilities.size + 1
    return {
        "true_hypothesis_mass": true_mass,
        "true_hypothesis_rank": rank,
        "true_in_95pct_credible_set": int(
            true_in_credible_set(sequence, posterior, 0.95)
        ),
        "normalized_entropy": posterior.normalized_entropy,
        "credible_set_size_95": posterior.credible_set_size(0.95),
        "map_association_accuracy": association_accuracy(
            sequence, posterior.associations[posterior.map_index]
        ),
        "map_identity_switches": identity_switches(
            sequence, posterior.associations[posterior.map_index]
        ),
    }

