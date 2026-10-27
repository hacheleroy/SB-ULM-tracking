"""Exact finite-support Gibbs redraw of one complete birth/death slot.

All one-slot trajectories are scored once under the collapsed Matérn model.
At each update, states that use a detection claimed by the fixed other
slots are removed; the remaining normalized law is the exact conditional
tempered posterior. The update may change the entire existence interval,
assignments, misses and clutter without a Metropolis correction.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .slot_existence import ABSENT, MISSED, SlotExistenceMaternModel

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class WholeSlotGibbs:
    """Reusable one-slot score table for an observation with <=62 detections."""

    def __init__(self, model: SlotExistenceMaternModel) -> None:
        self.model = model
        cards = tuple(model.sequence.cardinalities)
        if sum(cards) > 62:
            raise ValueError("whole-slot bit masks require at most 62 detections")
        self.offsets = np.cumsum((0,) + cards[:-1])
        paths: list[tuple[int, ...]] = []
        prior: list[float] = []
        bits: list[int] = []
        assigned: list[int] = []
        alive: list[int] = []
        for mask, log_prior in zip(model.interval_masks(),
                                   model.interval_log_probabilities(), strict=True):
            choices = [tuple(range(MISSED, card)) if exists else (ABSENT,)
                       for exists, card in zip(mask, cards, strict=True)]
            for path in product(*choices):
                used = 0
                n_assigned = 0
                for frame, value in enumerate(path):
                    if value >= 0:
                        used |= 1 << (int(self.offsets[frame]) + value)
                        n_assigned += 1
                paths.append(path)
                prior.append(float(log_prior))
                bits.append(used)
                assigned.append(n_assigned)
                alive.append(int(mask.sum()))
        self.paths = np.asarray(paths, dtype=np.int64)
        self.bits = np.asarray(bits, dtype=np.int64)
        n_assigned = np.asarray(assigned)
        self.base = (np.asarray(prior)
                     + n_assigned * np.log(model.detection_probability
                                           / model.clutter_intensity)
                     + (np.asarray(alive) - n_assigned)
                     * np.log1p(-model.detection_probability))
        self.gp = np.empty(len(paths), dtype=float)
        for start in range(0, len(paths), 2_048):
            end = min(start + 2_048, len(paths))
            states = np.full((end - start, model.sequence.n_frames, model.n_slots),
                             ABSENT, dtype=np.int64)
            states[:, :, 0] = self.paths[start:end]
            self.gp[start:end] = model.log_marginal_density(states)
        self._beta: float | None = None
        self._scores: FloatArray | None = None

    def log_conditional_scores(self, current: IntArray, slot: int, beta: float) -> FloatArray:
        """Return scores on the fixed-other-slot support; invalid paths are -inf."""

        if current.shape != (self.model.sequence.n_frames, self.model.n_slots):
            raise ValueError("current path has an unexpected shape")
        if not 0 <= slot < self.model.n_slots or not 0 <= beta <= 1:
            raise ValueError("slot or beta is outside its support")
        if self._beta != beta:
            self._scores = self.base + beta * self.gp
            self._beta = beta
        assert self._scores is not None
        other = np.delete(current, slot, axis=1)
        occupied = 0
        for frame, values in enumerate(other):
            for value in values:
                if value >= 0:
                    occupied |= 1 << (int(self.offsets[frame]) + int(value))
        scores = self._scores.copy()
        scores[(self.bits & occupied) != 0] = -np.inf
        return scores

    def sample(self, current: IntArray, rows: IntArray, beta: float,
               rng: np.random.Generator) -> IntArray:
        """Sample one uniformly chosen slot per row from its exact conditional."""

        proposals = current[rows].copy()
        for index, row in enumerate(rows):
            slot = int(rng.integers(self.model.n_slots))
            scores = self.log_conditional_scores(current[row], slot, beta)
            finite = np.flatnonzero(np.isfinite(scores))
            if not finite.size:
                raise ValueError("no supported whole-slot conditional state")
            probabilities = np.exp(scores[finite] - logsumexp(scores[finite]))
            chosen = int(rng.choice(finite, p=probabilities))
            proposals[index, :, slot] = self.paths[chosen]
        return proposals
