"""Reversible, temporally guided two-slot association/clutter proposals.

The auxiliary Markov-chain proposal is not the ultrasound posterior. Its
normalized law depends only on the selected slots' fixed existence masks,
the other slots, and frames outside the selected window. Those quantities
are invariant under the update, so the Hastings ratio is simply the old
minus proposed auxiliary path score. The exact collapsed target is still
used for acceptance by the SMC rejuvenation step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .slot_existence import SlotExistenceMaternModel, enumerate_frame_slot_associations

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _categorical(log_weights: FloatArray, rng: np.random.Generator) -> int:
    probabilities = np.exp(log_weights - logsumexp(log_weights))
    return int(rng.choice(probabilities.size, p=probabilities))


class GuidedPairWindow:
    """Finite-state guided law conditional on the unchanged surroundings.

    Each frame's options include missed observations and unused detections.
    A positive Gaussian displacement surrogate links the same assigned slot
    in adjacent frames. This is an instrumental proposal, not an extra term
    in the scientific target. All compatible options have positive mass.
    """

    def __init__(
        self,
        model: SlotExistenceMaternModel,
        current: IntArray,
        slots: tuple[int, int],
        start: int,
        length: int = 3,
    ) -> None:
        if current.shape != (model.sequence.n_frames, model.n_slots):
            raise ValueError("current must have shape (T, K)")
        if len(set(slots)) != 2 or any(s < 0 or s >= model.n_slots for s in slots):
            raise ValueError("slots must contain two distinct valid indices")
        if length < 2 or start < 0 or start + length > model.sequence.n_frames:
            raise ValueError("invalid multi-frame window")
        model._validate_associations(current[None])
        self.model, self.current, self.slots = model, current, slots
        self.start, self.length = start, length
        other = np.ones(model.n_slots, dtype=bool)
        other[list(slots)] = False
        self.options: list[IntArray] = []
        self.emissions: list[FloatArray] = []
        self.transitions: list[FloatArray] = []
        for time_index in range(start, start + length):
            cardinality = model.sequence.cardinalities[time_index]
            claimed = current[time_index, other]
            claimed = claimed[claimed >= 0]
            available = np.setdiff1d(
                np.arange(cardinality, dtype=np.int64), claimed, assume_unique=True
            )
            mask = current[time_index, list(slots)] != -2
            states = enumerate_frame_slot_associations(mask, available.size)
            assigned = states >= 0
            states[assigned] = available[states[assigned]]
            self.options.append(states)
            # The constant missed/clutter factors cancel within each frame.
            self.emissions.append(
                np.count_nonzero(assigned, axis=1)
                * np.log(model.detection_probability / model.clutter_intensity)
            )

        # A fixed, strictly positive scale keeps the instrumental law broad.
        # The exact Matérn marginal is evaluated separately in MH acceptance.
        self.displacement_variance = np.asarray(
            2.0 * model.localization_std**2 + np.asarray(model.coordinate_std) ** 2,
            dtype=float,
        )
        for offset in range(length - 1):
            self.transitions.append(
                self._edge(
                    self.options[offset],
                    self.options[offset + 1],
                    start + offset,
                    start + offset + 1,
                )
            )
        if start > 0:
            fixed = current[start - 1, list(slots)][None, :]
            self.emissions[0] += self._edge(fixed, self.options[0], start - 1, start)[0]
        if start + length < model.sequence.n_frames:
            fixed = current[start + length, list(slots)][None, :]
            self.emissions[-1] += self._edge(
                self.options[-1], fixed, start + length - 1, start + length
            )[:, 0]

        self.forward = [self.emissions[0]]
        for offset in range(1, length):
            self.forward.append(
                self.emissions[offset]
                + logsumexp(
                    self.forward[-1][:, None] + self.transitions[offset - 1], axis=0
                )
            )
        self.log_normalizer = float(logsumexp(self.forward[-1]))

    def _edge(
        self, left: IntArray, right: IntArray, first: int, second: int
    ) -> FloatArray:
        scores = np.zeros((left.shape[0], right.shape[0]), dtype=float)
        for position in range(2):
            left_index = left[:, position]
            right_index = right[:, position]
            left_valid = left_index >= 0
            right_valid = right_index >= 0
            if not np.any(left_valid) or not np.any(right_valid):
                continue
            left_point = self.model.sequence.detections[first][left_index[left_valid]]
            right_point = self.model.sequence.detections[second][right_index[right_valid]]
            differences = left_point[:, None, :] - right_point[None, :, :]
            scores[np.ix_(left_valid, right_valid)] -= 0.5 * np.sum(
                differences**2 / self.displacement_variance, axis=-1
            )
        return scores

    def log_probability(self, candidate: IntArray) -> float:
        """Log normalized conditional q, checking that context did not change."""

        if candidate.shape != self.current.shape:
            raise ValueError("candidate must have the same shape as current")
        unchanged = np.ones_like(candidate, dtype=bool)
        unchanged[self.start : self.start + self.length, list(self.slots)] = False
        if np.any(candidate[unchanged] != self.current[unchanged]):
            raise ValueError("candidate changed the fixed surroundings")
        indices = []
        for offset, states in enumerate(self.options):
            values = candidate[self.start + offset, list(self.slots)]
            matches = np.flatnonzero(np.all(states == values, axis=1))
            if matches.size != 1:
                raise ValueError("candidate is outside the conditional support")
            indices.append(int(matches[0]))
        score = sum(self.emissions[t][index] for t, index in enumerate(indices))
        score += sum(
            self.transitions[t][indices[t], indices[t + 1]]
            for t in range(self.length - 1)
        )
        return float(score - self.log_normalizer)

    def sample(self, rng: np.random.Generator) -> IntArray:
        indices = np.empty(self.length, dtype=np.int64)
        indices[-1] = _categorical(self.forward[-1], rng)
        for offset in range(self.length - 2, -1, -1):
            indices[offset] = _categorical(
                self.forward[offset] + self.transitions[offset][:, indices[offset + 1]],
                rng,
            )
        proposal = self.current.copy()
        for offset, index in enumerate(indices):
            proposal[self.start + offset, self.slots[0]] = self.options[offset][index, 0]
            proposal[self.start + offset, self.slots[1]] = self.options[offset][index, 1]
        return proposal


def propose_guided_tracklets(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
    *,
    length: int = 3,
) -> tuple[IntArray, FloatArray]:
    """Return proposals and log q(old)/q(new) for state-independent windows."""

    if model.n_slots < 2 or model.sequence.n_frames < length or length < 2:
        raise ValueError("guided tracklets need two slots and a valid window")
    proposed = current[rows].copy()
    hastings = np.zeros(rows.size, dtype=float)
    for local_index, row in enumerate(rows):
        pair = tuple(int(s) for s in rng.choice(model.n_slots, size=2, replace=False))
        start = int(rng.integers(model.sequence.n_frames - length + 1))
        conditional = GuidedPairWindow(model, current[row], pair, start, length)
        proposed[local_index] = conditional.sample(rng)
        hastings[local_index] = conditional.log_probability(current[row]) - (
            conditional.log_probability(proposed[local_index])
        )
    return proposed, hastings
