"""Likelihood-guided conditional SMC for a full pair of ULM slot histories.

The Feynman--Kac target at frame t is the exact birth/death, finite-set count,
and tempered collapsed Matérn density restricted to frames 0..t.  Its
localization increment is the Gaussian predictive density conditioned on all
previously assigned observations of the same slot.  A fixed reference particle
turns the fully-adapted particle filter into a particle-Gibbs update.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve
from scipy.special import logsumexp

from .paired_moves import PairConditionalBase, _pair_markov_model
from .slot_existence import SlotExistenceMaternModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ConditionalPairPath:
    """Fully adapted pair proposal, conditional on the unselected slots."""

    def __init__(
        self, model: SlotExistenceMaternModel, current: IntArray, slots: tuple[int, int]
    ) -> None:
        self.base = PairConditionalBase(model, current, slots)
        self.model = model
        self.slots = slots
        _, self.initial, self.transitions = _pair_markov_model(model.birth_death_prior)

    def _predictive_log_density(
        self, prefix: IntArray, slot: int, time_index: int, available: IntArray
    ) -> FloatArray:
        """Exact GP predictive log density for each available new localization."""

        if available.size == 0:
            return np.empty(0, dtype=float)
        gp = self.model._gp_model
        previous = tuple(int(t) for t in np.flatnonzero(prefix[:time_index, slot] >= 0))
        candidates = self.model.sequence.detections[time_index][available]
        scores = np.zeros(available.size, dtype=float)
        for dimension in range(self.model.sequence.dimension):
            covariance = gp._prior_covariances[dimension]
            variance = covariance[time_index, time_index] + gp.localization_std**2
            mean = 0.0
            if previous:
                factor, _ = gp._observation_factor(dimension, previous)
                cross = covariance[time_index, list(previous)]
                observations = np.asarray(
                    [
                        self.model.sequence.detections[t][prefix[t, slot], dimension]
                        for t in previous
                    ]
                )
                mean = float(cross @ cho_solve(factor, observations, check_finite=False))
                variance -= float(cross @ cho_solve(factor, cross, check_finite=False))
            if variance <= 0:
                raise FloatingPointError("non-positive GP predictive variance")
            residual = candidates[:, dimension] - mean
            scores -= 0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)
        return scores

    def options(
        self, prefix: IntArray, previous_state: int | None, time_index: int, beta: float
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Enumerate one-frame proposal support and exact incremental log weights."""

        if not 0 <= time_index < self.model.sequence.n_frames or not 0 <= beta <= 1:
            raise ValueError("invalid frame or tempering parameter")
        if (time_index == 0) != (previous_state is None):
            raise ValueError("previous_state is required exactly after the first frame")
        available = self.base.available_by_frame[time_index]
        predictive = (
            self._predictive_log_density(prefix, self.slots[0], time_index, available),
            self._predictive_log_density(prefix, self.slots[1], time_index, available),
        )
        lookup = {int(detection): i for i, detection in enumerate(available)}
        transition = (
            self.initial if previous_state is None else self.transitions[previous_state]
        )
        states: list[int] = []
        assignments: list[IntArray] = []
        log_weights: list[float] = []
        for state_index, state in enumerate(self.base.states):
            if not np.isfinite(transition[state_index]):
                continue
            frame_options, count_weights = self.base._frame_options(
                tuple(state == 1), available.size
            )
            for option, count_weight in zip(frame_options, count_weights, strict=True):
                assignment = option.copy()
                assigned = assignment >= 0
                assignment[assigned] = available[assignment[assigned]]
                increment = 0.0
                for local_slot, detection in enumerate(assignment):
                    if detection >= 0:
                        increment += predictive[local_slot][lookup[int(detection)]]
                states.append(state_index)
                assignments.append(assignment)
                log_weights.append(float(transition[state_index] + count_weight + beta * increment))
        return (
            np.asarray(states, dtype=np.int64),
            np.stack(assignments),
            np.asarray(log_weights, dtype=float),
        )

    def sample(
        self,
        *,
        beta: float,
        n_particles: int,
        rng: np.random.Generator,
    ) -> IntArray:
        """One exact-target conditional SMC/particle-Gibbs transition.

        Resampling is multinomial; the reference particle retains its own
        ancestor.  Proposal weights are local normalizers because proposals
        are fully adapted to the incremental collapsed likelihood.
        """

        if n_particles < 2:
            raise ValueError("conditional SMC requires at least two particles")
        reference = self.base.current
        particles = np.broadcast_to(reference, (n_particles, *reference.shape)).copy()
        state_indices = np.empty(n_particles, dtype=np.int64)
        weights = np.full(n_particles, 1.0 / n_particles)
        fixed = n_particles - 1
        for time_index in range(self.model.sequence.n_frames):
            if time_index:
                ancestors = rng.choice(n_particles, size=fixed, p=weights)
                parents = np.concatenate((ancestors, np.asarray([fixed])))
                particles = particles[parents].copy()
                state_indices = state_indices[parents]
            incremental = np.empty(n_particles, dtype=float)
            for particle_index in range(n_particles):
                previous = None if time_index == 0 else int(state_indices[particle_index])
                states, options, log_scores = self.options(
                    particles[particle_index], previous, time_index, beta
                )
                normalizer = float(logsumexp(log_scores))
                incremental[particle_index] = normalizer
                if particle_index == fixed:
                    matching = np.flatnonzero(np.all(
                        options == reference[time_index, list(self.slots)], axis=1
                    ))
                    if matching.size != 1:
                        raise ValueError("reference pair is outside conditional support")
                    selected = int(matching[0])
                else:
                    selected = int(rng.choice(
                        len(states), p=np.exp(log_scores - normalizer)
                    ))
                particles[particle_index, time_index, list(self.slots)] = options[selected]
                state_indices[particle_index] = states[selected]
            weights = np.exp(incremental - logsumexp(incremental))
        chosen = int(rng.choice(n_particles, p=weights))
        return particles[chosen]


def conditional_pair_pg_move(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    *,
    beta: float,
    n_particles: int,
    rng: np.random.Generator,
) -> IntArray:
    """Update a uniformly selected pair for each selected SMC row."""

    if model.n_slots < 2:
        raise ValueError("pair moves require at least two slots")
    proposed = current[rows].copy()
    for local_index, row in enumerate(rows):
        pair = rng.choice(model.n_slots, size=2, replace=False)
        conditional = ConditionalPairPath(model, current[row], (int(pair[0]), int(pair[1])))
        proposed[local_index] = conditional.sample(
            beta=beta, n_particles=n_particles, rng=rng
        )
    return proposed
