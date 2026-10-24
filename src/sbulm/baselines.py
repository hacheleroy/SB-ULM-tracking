"""Deterministic localization-level tracking baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .data import DetectionSequence, stack_assigned_detections

IntArray = NDArray[np.int64]


def hungarian_tracker(
    sequence: DetectionSequence, *, constant_velocity: bool = False, velocity_memory: float = 0.5
) -> tuple[IntArray, np.ndarray]:
    """Associate each frame with Hungarian matching.

    Identities are defined by the order of detections in the first frame.
    ``constant_velocity=False`` gives nearest-neighbour frame-to-frame matching;
    ``True`` matches to a smoothed constant-velocity prediction.
    """

    if not sequence.has_constant_cardinality:
        raise ValueError("Hungarian baseline currently requires constant cardinality")
    n_particles = sequence.cardinalities[0]
    if n_particles == 0:
        raise ValueError("frames must contain at least one detection")
    if not 0 <= velocity_memory < 1:
        raise ValueError("velocity_memory must lie in [0, 1)")

    associations = np.empty((sequence.n_frames, n_particles), dtype=np.int64)
    associations[0] = np.arange(n_particles, dtype=np.int64)
    assigned = np.empty((sequence.n_frames, n_particles, sequence.dimension), dtype=float)
    assigned[0] = sequence.detections[0]
    velocity = np.zeros_like(assigned[0])

    for time_index in range(1, sequence.n_frames):
        dt = sequence.times[time_index] - sequence.times[time_index - 1]
        prediction = assigned[time_index - 1]
        if constant_velocity:
            prediction = prediction + velocity * dt
        cost = cdist(prediction, sequence.detections[time_index], metric="sqeuclidean")
        rows, columns = linear_sum_assignment(cost)
        order = np.empty(n_particles, dtype=np.int64)
        order[rows] = columns
        associations[time_index] = order
        assigned[time_index] = sequence.detections[time_index][order]
        observed_velocity = (assigned[time_index] - assigned[time_index - 1]) / dt
        velocity = velocity_memory * velocity + (1.0 - velocity_memory) * observed_velocity

    return associations, stack_assigned_detections(sequence, associations)

