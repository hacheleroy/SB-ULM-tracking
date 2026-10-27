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

from .paired_moves import PairFrameSupport
from .slot_existence import ABSENT, SlotExistenceMaternModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ConditionalPairPath:
    """Fully adapted pair proposal, conditional on the unselected slots."""

    def __init__(
        self, model: SlotExistenceMaternModel, current: IntArray, slots: tuple[int, int]
    ) -> None:
        self.base = PairFrameSupport(model, current, slots)
        self.model = model
        self.slots = slots
        self.initial = self.base.initial
        self.transitions = self.base.transitions
        self._predictive_cache: dict[tuple[int, int, tuple[int, ...]], FloatArray] = {}
        self._options_cache: dict[
            tuple[int, int | None, float, tuple[int, ...]],
            tuple[IntArray, IntArray, FloatArray],
        ] = {}

    def _predictive_log_density(
        self, prefix: IntArray, slot: int, time_index: int, available: IntArray
    ) -> FloatArray:
        """Exact GP predictive log density for each available new localization."""

        if available.size == 0:
            return np.empty(0, dtype=float)
        key = (slot, time_index, tuple(int(v) for v in prefix[:time_index, slot]))
        cached = self._predictive_cache.get(key)
        if cached is not None:
            return cached
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
        self._predictive_cache[key] = scores
        return scores

    def options(
        self, prefix: IntArray, previous_state: int | None, time_index: int, beta: float
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Enumerate one-frame proposal support and exact incremental log weights."""

        if not 0 <= time_index < self.model.sequence.n_frames or not 0 <= beta <= 1:
            raise ValueError("invalid frame or tempering parameter")
        if (time_index == 0) != (previous_state is None):
            raise ValueError("previous_state is required exactly after the first frame")
        key = (
            time_index, previous_state, float(beta),
            tuple(int(v) for v in prefix[:time_index, list(self.slots)].ravel()),
        )
        cached = self._options_cache.get(key)
        if cached is not None:
            return cached
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
        result = (
            np.asarray(states, dtype=np.int64),
            np.stack(assignments),
            np.asarray(log_weights, dtype=float),
        )
        self._options_cache[key] = result
        return result

    def _previous_state(self, prefix: IntArray, time_index: int) -> int:
        """Recover the pair's unborn/alive/dead state at the end of a prefix."""

        states = []
        for slot in self.slots:
            if prefix[time_index - 1, slot] != ABSENT:
                states.append(1)
            elif np.any(prefix[:time_index, slot] != ABSENT):
                states.append(2)
            else:
                states.append(0)
        return 3 * states[0] + states[1]

    def _reference_suffix_score(
        self, prefix: IntArray, reference: IntArray, time_index: int, beta: float
    ) -> float:
        """Exact log gamma_T(spliced)/gamma_(t-1)(prefix), up to fixed terms."""

        spliced = reference.copy()
        spliced[:time_index] = prefix[:time_index]
        previous = self._previous_state(spliced, time_index)
        score = 0.0
        for frame in range(time_index, self.model.sequence.n_frames):
            states, options, log_scores = self.options(spliced, previous, frame, beta)
            matching = np.flatnonzero(np.all(
                options == reference[frame, list(self.slots)], axis=1
            ))
            if matching.size != 1:
                return -np.inf
            selected = int(matching[0])
            score += float(log_scores[selected])
            previous = int(states[selected])
        return score

    def _ancestor_mh(
        self, particles: IntArray, reference: IntArray, time_index: int,
        beta: float, log_weights: FloatArray, rng: np.random.Generator,
    ) -> int:
        """One forced-move MH step targeting the exact PGAS ancestor law."""

        fixed = particles.shape[0] - 1
        proposed = int(rng.integers(fixed))
        old_score = self._reference_suffix_score(
            particles[fixed], reference, time_index, beta
        )
        if not np.isfinite(old_score):
            raise ValueError("the retained reference suffix must be supported")
        new_score = self._reference_suffix_score(
            particles[proposed], reference, time_index, beta
        )
        log_ratio = (log_weights[proposed] + new_score
                     - log_weights[fixed] - old_score)
        return proposed if np.log(rng.random()) < log_ratio else fixed

    def sample(
        self,
        *,
        beta: float,
        n_particles: int,
        rng: np.random.Generator,
        ancestor_sampling: bool = False,
        start_frame: int = 0,
    ) -> IntArray:
        """One exact-target conditional SMC/particle-Gibbs transition.

        Resampling is multinomial; the reference particle retains its own
        ancestor.  Proposal weights are local normalizers because proposals
        are fully adapted to the incremental collapsed likelihood. A positive
        start_frame conditions on the reference prefix and resamples only the
        suffix. This remains a Gibbs kernel for the tempered full target.
        """

        if n_particles < 2:
            raise ValueError("conditional SMC requires at least two particles")
        if not 0 <= start_frame < self.model.sequence.n_frames:
            raise ValueError("start_frame must index an observed frame")
        reference = self.base.current
        particles = np.broadcast_to(reference, (n_particles, *reference.shape)).copy()
        state_indices = np.empty(n_particles, dtype=np.int64)
        if start_frame:
            state_indices.fill(self._previous_state(reference, start_frame))
        weights = np.full(n_particles, 1.0 / n_particles)
        log_weights = np.full(n_particles, -np.log(n_particles))
        fixed = n_particles - 1
        for time_index in range(start_frame, self.model.sequence.n_frames):
            if time_index > start_frame:
                ancestors = rng.choice(n_particles, size=fixed, p=weights)
                retained_parent = (
                    self._ancestor_mh(
                        particles, reference, time_index, beta, log_weights, rng
                    ) if ancestor_sampling else fixed
                )
                parents = np.concatenate((ancestors, np.asarray([retained_parent])))
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
            log_weights = incremental - logsumexp(incremental)
            weights = np.exp(log_weights)
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
    ancestor_sampling: bool = False,
    start_frame: int = 0,
) -> IntArray:
    """Update a uniformly selected pair, optionally conditioning on its prefix."""

    if model.n_slots < 2:
        raise ValueError("pair moves require at least two slots")
    proposed = current[rows].copy()
    for local_index, row in enumerate(rows):
        pair = rng.choice(model.n_slots, size=2, replace=False)
        conditional = ConditionalPairPath(model, current[row], (int(pair[0]), int(pair[1])))
        proposed[local_index] = conditional.sample(
            beta=beta, n_particles=n_particles, rng=rng,
            ancestor_sampling=ancestor_sampling,
            start_frame=start_frame,
        )
    return proposed

