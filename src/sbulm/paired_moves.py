"""Exact conditional-base pair proposals with likelihood-informed multiple tries."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .slot_existence import (
    ABSENT,
    BirthDeathPrior,
    SlotExistenceMaternModel,
    enumerate_frame_slot_associations,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _pair_markov_model(prior: BirthDeathPrior) -> tuple[IntArray, FloatArray, FloatArray]:
    """Unborn/alive/dead initial law and transitions for two selected slots."""

    states = np.asarray(list(product(range(3), repeat=2)), dtype=np.int64)
    initial = np.asarray([1.0 - prior.initial_probability, prior.initial_probability, 0.0])
    transition = np.asarray(
        [
            [1.0 - prior.birth_probability, prior.birth_probability, 0.0],
            [0.0, prior.survival_probability, 1.0 - prior.survival_probability],
            [0.0, 0.0, 1.0],
        ]
    )
    probabilities = np.ones((states.shape[0], states.shape[0]), dtype=float)
    for slot in range(2):
        probabilities *= transition[states[:, slot, None], states[None, :, slot]]
    with np.errstate(divide="ignore"):
        return (
            states,
            np.sum(np.log(initial[states]), axis=1),
            np.log(probabilities),
        )


class PairConditionalBase:
    """Normalized two-slot base law conditional on all unselected slots.

    The two selected slots are drawn jointly, including both existence
    intervals and all their frame assignments.  The remaining slots and their
    claimed detections are fixed.  A nine-state HMM gives an exact draw without
    enumerating interval products or possible full association histories.
    """

    def __init__(
        self,
        model: SlotExistenceMaternModel,
        current: IntArray,
        slots: tuple[int, int],
    ) -> None:
        if current.shape != (model.sequence.n_frames, model.n_slots):
            raise ValueError("current must have shape (T, K)")
        if len(set(slots)) != 2 or any(slot < 0 or slot >= model.n_slots for slot in slots):
            raise ValueError("slots must identify two distinct valid slots")
        self.model = model
        self.current = current
        self.slots = slots
        other = np.ones(model.n_slots, dtype=bool)
        other[list(slots)] = False
        self.available_by_frame: list[IntArray] = []
        self._frame_cache: dict[tuple[bool, bool, int], tuple[IntArray, FloatArray]] = {}
        states, initial, transitions = _pair_markov_model(model.birth_death_prior)
        self.states = states
        self.transitions = transitions
        emissions = np.empty((model.sequence.n_frames, states.shape[0]), dtype=float)
        for time_index, cardinality in enumerate(model.sequence.cardinalities):
            claimed = current[time_index, other]
            claimed = claimed[claimed >= 0]
            available = np.setdiff1d(
                np.arange(cardinality, dtype=np.int64), claimed, assume_unique=True
            )
            self.available_by_frame.append(available)
            for index, state in enumerate(states):
                _, weights = self._frame_options(tuple(state == 1), available.size)
                emissions[time_index, index] = float(logsumexp(weights))
        forward = np.empty_like(emissions)
        forward[0] = initial + emissions[0]
        for time_index in range(1, model.sequence.n_frames):
            forward[time_index] = emissions[time_index] + logsumexp(
                forward[time_index - 1, :, None] + transitions, axis=0
            )
        self.forward = forward
        self.log_normalizer = float(logsumexp(forward[-1]))

    def _frame_options(
        self, mask: tuple[bool, bool], cardinality: int
    ) -> tuple[IntArray, FloatArray]:
        key = (*mask, cardinality)
        cached = self._frame_cache.get(key)
        if cached is None:
            states = enumerate_frame_slot_associations(
                np.asarray(mask, dtype=bool), cardinality
            )
            assigned = np.count_nonzero(states >= 0, axis=1)
            weights = (
                assigned * np.log(self.model.detection_probability)
                + (sum(mask) - assigned) * np.log1p(-self.model.detection_probability)
                + (cardinality - assigned) * np.log(self.model.clutter_intensity)
            )
            cached = states, np.asarray(weights, dtype=float)
            self._frame_cache[key] = cached
        return cached

    def sample(self, n_samples: int, rng: np.random.Generator) -> IntArray:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        n_frames = self.model.sequence.n_frames
        indices = np.empty((n_samples, n_frames), dtype=np.int64)
        indices[:, -1] = rng.choice(
            self.states.shape[0],
            size=n_samples,
            p=np.exp(self.forward[-1] - self.log_normalizer),
        )
        for time_index in range(n_frames - 2, -1, -1):
            for next_state in np.unique(indices[:, time_index + 1]):
                rows = np.flatnonzero(indices[:, time_index + 1] == next_state)
                weights = self.forward[time_index] + self.transitions[:, next_state]
                indices[rows, time_index] = rng.choice(
                    self.states.shape[0],
                    size=rows.size,
                    p=np.exp(weights - logsumexp(weights)),
                )

        samples = np.broadcast_to(self.current, (n_samples, *self.current.shape)).copy()
        for time_index, available in enumerate(self.available_by_frame):
            for index in np.unique(indices[:, time_index]):
                rows = np.flatnonzero(indices[:, time_index] == index)
                mask = tuple(self.states[index] == 1)
                options, weights = self._frame_options(mask, available.size)
                draws = rng.choice(
                    options.shape[0],
                    size=rows.size,
                    p=np.exp(weights - logsumexp(weights)),
                )
                values = options[draws].copy()
                assigned = values >= 0
                values[assigned] = available[values[assigned]]
                samples[rows, time_index, self.slots[0]] = values[:, 0]
                samples[rows, time_index, self.slots[1]] = values[:, 1]
        return samples

    def log_probability(self, candidates: IntArray) -> FloatArray:
        """Evaluate the exact conditional proposal (useful for oracle tests)."""

        if candidates.ndim != 3 or candidates.shape[1:] != self.current.shape:
            raise ValueError("candidates must have shape (S, T, K)")
        other = np.ones(self.model.n_slots, dtype=bool)
        other[list(self.slots)] = False
        if not np.array_equal(
            candidates[:, :, other],
            np.broadcast_to(self.current[:, other], candidates[:, :, other].shape),
        ):
            raise ValueError("unselected slots must remain fixed")
        selected = candidates[:, :, list(self.slots)]
        self.model._validate_associations(candidates)
        scores = np.zeros(candidates.shape[0], dtype=float)
        for slot in range(2):
            scores += self.model.birth_death_prior.log_probability(
                selected[:, :, slot] != ABSENT
            )
        assigned = np.count_nonzero(selected >= 0, axis=2)
        active = np.count_nonzero(selected != ABSENT, axis=2)
        available_counts = np.asarray([a.size for a in self.available_by_frame])
        scores += np.sum(
            assigned * np.log(self.model.detection_probability)
            + (active - assigned) * np.log1p(-self.model.detection_probability)
            + (available_counts[None, :] - assigned) * np.log(self.model.clutter_intensity),
            axis=1,
        )
        return scores - self.log_normalizer


def multiple_try_paired_move(
    current: IntArray,
    current_marginal: FloatArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    *,
    beta: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, NDArray[np.bool_]]:
    """Joint pair update with independent multiple-try Metropolis weights.

    Conditional-base candidates have importance weight proportional to
    ``exp(beta * log_marginal_density)``.  The reverse set includes the current
    state and ``n_candidates - 1`` fresh base candidates.  The ratio of the
    two weight sums is the exact multiple-try Hastings correction.
    """

    if model.n_slots < 2 or n_candidates < 2:
        raise ValueError("paired moves need two slots and at least two tries")
    if rows.size == 0:
        return current[rows].copy(), current_marginal[rows].copy(), np.zeros(0, bool)
    n_draws = 2 * n_candidates - 1
    candidates = np.empty(
        (rows.size, n_draws, model.sequence.n_frames, model.n_slots), dtype=np.int64
    )
    for local_index, row in enumerate(rows):
        first, second = rng.choice(model.n_slots, size=2, replace=False)
        conditional = PairConditionalBase(model, current[row], (int(first), int(second)))
        candidates[local_index] = conditional.sample(n_draws, rng)
    marginal = model.log_marginal_density(
        candidates.reshape(-1, model.sequence.n_frames, model.n_slots)
    ).reshape(rows.size, n_draws)
    forward = beta * marginal[:, :n_candidates]
    forward_log_sum = logsumexp(forward, axis=1)
    probabilities = np.exp(forward - forward_log_sum[:, None])
    selected = np.sum(
        np.cumsum(probabilities, axis=1) < rng.random(rows.size)[:, None], axis=1
    )
    selected = np.minimum(selected, n_candidates - 1)
    reverse_log_sum = np.logaddexp(
        beta * current_marginal[rows],
        logsumexp(beta * marginal[:, n_candidates:], axis=1),
    )
    accepted = np.log(rng.random(rows.size)) < forward_log_sum - reverse_log_sum
    indices = np.arange(rows.size)
    return candidates[indices, selected], marginal[indices, selected], accepted


def _local_pair_candidate_set(
    model: SlotExistenceMaternModel,
    current: IntArray,
    slots: tuple[int, int],
    start: int,
    length: int,
) -> IntArray:
    """Enumerate a one- or two-frame conditional support for two slots."""

    if length not in (1, 2) or start < 0 or start + length > model.sequence.n_frames:
        raise ValueError("invalid pair window")
    other = np.ones(model.n_slots, dtype=bool)
    other[list(slots)] = False
    frame_options: list[IntArray] = []
    for time_index in range(start, start + length):
        claimed = current[time_index, other]
        claimed = claimed[claimed >= 0]
        cardinality = model.sequence.cardinalities[time_index]
        available = np.setdiff1d(
            np.arange(cardinality, dtype=np.int64), claimed, assume_unique=True
        )
        options = []
        for first_alive, second_alive in product((False, True), repeat=2):
            states = enumerate_frame_slot_associations(
                np.asarray([first_alive, second_alive]), available.size
            )
            states = states.copy()
            assigned = states >= 0
            states[assigned] = available[states[assigned]]
            options.append(states)
        frame_options.append(np.concatenate(options))

    candidates = []
    for frames in product(*frame_options):
        proposal = current.copy()
        for offset, frame in enumerate(frames):
            proposal[start + offset, slots[0]] = frame[0]
            proposal[start + offset, slots[1]] = frame[1]
        if all(
            _is_contiguous(proposal[:, slot] != ABSENT) for slot in slots
        ):
            candidates.append(proposal)
    if not candidates:
        raise RuntimeError("the unchanged state must be in its conditional support")
    return np.stack(candidates)


def _is_contiguous(mask: NDArray[np.bool_]) -> bool:
    alive = np.flatnonzero(mask)
    return bool(alive.size == 0 or alive[-1] - alive[0] + 1 == alive.size)


def local_pair_multiple_try_move(
    current: IntArray,
    current_marginal: FloatArray,
    current_count: FloatArray,
    current_existence: FloatArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    *,
    beta: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """Reweight reversible local two-slot candidates with the exact likelihood.

    Given the other slots and all frames outside a selected window, the finite
    candidate set is identical in the forward and reverse direction.  Uniform
    draws from that set are therefore an independent proposal; multiple-try
    weights are the complete tempered target density.  Both selected slots can
    change existence, misses, clutter, or assignments within the window.
    """

    if model.n_slots < 2 or n_candidates < 2:
        raise ValueError("paired moves need two slots and at least two tries")
    if rows.size == 0:
        empty = np.empty(0, dtype=float)
        return current[rows].copy(), empty, empty, empty, np.zeros(0, bool)

    n_draws = 2 * n_candidates - 1
    candidates = np.empty(
        (rows.size, n_draws, model.sequence.n_frames, model.n_slots), dtype=np.int64
    )
    for local_index, row in enumerate(rows):
        first, second = rng.choice(model.n_slots, size=2, replace=False)
        start = int(rng.integers(model.sequence.n_frames))
        pair = (int(first), int(second))
        # The raw option count depends on fixed other slots only.  In
        # particular, the length decision must not use pair states inside the
        # window, or its selection probability would change on reversal.
        length = 1
        if start + 1 < model.sequence.n_frames and rng.random() < 0.5:
            other = np.ones(model.n_slots, dtype=bool)
            other[list(pair)] = False
            available_counts = [
                model.sequence.cardinalities[time_index]
                - np.count_nonzero(current[row, time_index, other] >= 0)
                for time_index in (start, start + 1)
            ]
            raw_counts = [4 + 3 * size + size * size for size in available_counts]
            if raw_counts[0] * raw_counts[1] <= 128:
                length = 2
        options = _local_pair_candidate_set(model, current[row], pair, start, length)
        candidates[local_index] = options[
            rng.integers(options.shape[0], size=n_draws)
        ]

    flat = candidates.reshape(-1, model.sequence.n_frames, model.n_slots)
    marginal = model.log_marginal_density(flat).reshape(rows.size, n_draws)
    count = model.log_count_density(
        flat, include_poisson_void=False
    ).reshape(rows.size, n_draws)
    existence = model.log_existence_prior(flat).reshape(rows.size, n_draws)
    forward = existence[:, :n_candidates] + count[:, :n_candidates] + (
        beta * marginal[:, :n_candidates]
    )
    forward_log_sum = logsumexp(forward, axis=1)
    probabilities = np.exp(forward - forward_log_sum[:, None])
    selected = np.sum(
        np.cumsum(probabilities, axis=1) < rng.random(rows.size)[:, None], axis=1
    )
    selected = np.minimum(selected, n_candidates - 1)
    reverse = existence[:, n_candidates:] + count[:, n_candidates:] + (
        beta * marginal[:, n_candidates:]
    )
    reverse_log_sum = np.logaddexp(
        current_existence[rows] + current_count[rows] + beta * current_marginal[rows],
        logsumexp(reverse, axis=1),
    )
    accepted = np.log(rng.random(rows.size)) < forward_log_sum - reverse_log_sum
    indices = np.arange(rows.size)
    return (
        candidates[indices, selected],
        marginal[indices, selected],
        count[indices, selected],
        existence[indices, selected],
        accepted,
    )
