"""Non-self, GP-predictive association moves with exact Hastings correction.

The instrumental finite-state HMM uses predictive localization densities
conditioned only on observations outside the selected three-frame window.
Its inputs are invariant under a move. Forward filtering with a second
"already different" flag samples exactly from this law excluding the current
path. The Metropolis step still evaluates the full collapsed target.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .slot_existence import ABSENT, SlotExistenceMaternModel
from .tracklet_moves import GuidedPairWindow, _categorical

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@lru_cache(maxsize=8_192)
def _invariant_windows(
    mask_bytes: bytes,
    n_frames: int,
    n_slots: int,
    cardinalities: tuple[int, ...],
    length: int,
) -> tuple[tuple[tuple[tuple[int, int], int], ...], tuple[float, ...]]:
    """Selection probabilities depend only on masks and fixed observation counts."""

    masks = np.frombuffer(mask_bytes, dtype=np.bool_).reshape(n_frames, n_slots)
    counts = np.asarray(cardinalities)
    choices: list[tuple[tuple[int, int], int]] = []
    weights: list[int] = []
    for pair in combinations(range(n_slots), 2):
        for start in range(n_frames - length + 1):
            active = masks[start : start + length, list(pair)]
            observable = counts[start : start + length] > 0
            if not np.any(active & observable[:, None]):
                continue
            choices.append((pair, start))
            weights.append(
                1 + int(np.count_nonzero(active))
                + 2 * int(np.count_nonzero(np.all(active, axis=1)))
            )
    if not choices:
        return (), ()
    total = sum(weights)
    return tuple(choices), tuple(value / total for value in weights)


class ConditionalGPWindow(GuidedPairWindow):
    """Positive, non-self local proposal guided by fixed-context GP prediction.

    The assigned-detection emission is the Gaussian predictive log density
    under the target's Matérn covariance conditioned only on fixed detections
    outside the window. Within-window GP dependence is approximated by
    half-weight displacement edges. The exact target enters MH acceptance.
    """

    def __init__(
        self,
        model: SlotExistenceMaternModel,
        current: IntArray,
        slots: tuple[int, int],
        start: int,
        length: int = 3,
    ) -> None:
        super().__init__(model, current, slots, start, length)
        self.transitions = [0.5 * edge for edge in self.transitions]
        self._add_predictive_emissions()
        self.forward = [self.emissions[0]]
        for offset in range(1, length):
            self.forward.append(
                self.emissions[offset]
                + logsumexp(
                    self.forward[-1][:, None] + self.transitions[offset - 1], axis=0
                )
            )
        self.log_normalizer = float(logsumexp(self.forward[-1]))

    def _add_predictive_emissions(self) -> None:
        model = self.model
        frame_indices = np.arange(self.start, self.start + self.length)
        outside = np.ones(model.sequence.n_frames, dtype=bool)
        outside[frame_indices] = False
        for position, slot in enumerate(self.slots):
            observed = np.flatnonzero(outside & (self.current[:, slot] >= 0))
            means = np.zeros((self.length, model.sequence.dimension), dtype=float)
            variances = np.empty_like(means)
            for dimension, covariance in enumerate(model._gp_model._prior_covariances):
                variances[:, dimension] = np.diag(covariance)[frame_indices] + (
                    model.localization_std**2
                )
                if observed.size:
                    observed_covariance = covariance[np.ix_(observed, observed)].copy()
                    observed_covariance.flat[:: observed.size + 1] += (
                        model.localization_std**2
                    )
                    factor = cho_factor(observed_covariance, lower=True, check_finite=False)
                    cross = covariance[np.ix_(frame_indices, observed)]
                    values = np.asarray(
                        [
                            model.sequence.detections[t][self.current[t, slot], dimension]
                            for t in observed
                        ]
                    )
                    means[:, dimension] = cross @ cho_solve(
                        factor, values, check_finite=False
                    )
                    variances[:, dimension] -= np.sum(
                        cross * cho_solve(factor, cross.T, check_finite=False).T,
                        axis=1,
                    )
            if np.any(variances <= 0):
                raise ArithmeticError("GP predictive variance must be positive")
            for offset, states in enumerate(self.options):
                assigned = states[:, position] >= 0
                if not np.any(assigned):
                    continue
                points = model.sequence.detections[self.start + offset][
                    states[assigned, position]
                ]
                normalized = (points - means[offset]) ** 2 / variances[offset]
                self.emissions[offset][assigned] += -0.5 * np.sum(
                    normalized + np.log(2.0 * np.pi * variances[offset]), axis=1
                )

    def _indices(self, candidate: IntArray) -> list[int]:
        self.log_probability(candidate)
        return [
            int(np.flatnonzero(np.all(
                options == candidate[self.start + offset, list(self.slots)], axis=1
            ))[0])
            for offset, options in enumerate(self.options)
        ]

    def _different_forward(self, forbidden: list[int]) -> tuple[list[FloatArray], list[FloatArray]]:
        same = np.full(self.options[0].shape[0], -np.inf)
        same[forbidden[0]] = self.emissions[0][forbidden[0]]
        different = self.emissions[0].copy()
        different[forbidden[0]] = -np.inf
        same_history, different_history = [same], [different]
        for offset in range(1, self.length):
            edge = self.transitions[offset - 1]
            new_same = np.full(self.options[offset].shape[0], -np.inf)
            new_same[forbidden[offset]] = (
                same[forbidden[offset - 1]]
                + edge[forbidden[offset - 1], forbidden[offset]]
                + self.emissions[offset][forbidden[offset]]
            )
            from_same = same[forbidden[offset - 1]] + edge[forbidden[offset - 1]].copy()
            from_same[forbidden[offset]] = -np.inf
            new_different = self.emissions[offset] + np.logaddexp(
                logsumexp(different[:, None] + edge, axis=0), from_same
            )
            same, different = new_same, new_different
            same_history.append(same)
            different_history.append(different)
        return same_history, different_history

    def propose_nonself(self, rng: np.random.Generator) -> tuple[IntArray, float]:
        """Draw y != x and return log q(x|y) - log q(y|x)."""

        old_indices = self._indices(self.current)
        same, different = self._different_forward(old_indices)
        old_different_partition = float(logsumexp(different[-1]))
        if not np.isfinite(old_different_partition):
            return self.current.copy(), 0.0
        indices = np.empty(self.length, dtype=np.int64)
        indices[-1] = _categorical(different[-1], rng)
        differing = True
        for offset in range(self.length - 1, 0, -1):
            if not differing:
                indices[offset - 1] = old_indices[offset - 1]
                continue
            next_index = indices[offset]
            edge = self.transitions[offset - 1][:, next_index]
            diff_weights = different[offset - 1] + edge
            same_weights = same[offset - 1] + edge
            if next_index == old_indices[offset]:
                same_weights = np.full_like(same_weights, -np.inf)
            choice = _categorical(np.concatenate((same_weights, diff_weights)), rng)
            if choice < same_weights.size:
                indices[offset - 1] = choice
                differing = False
            else:
                indices[offset - 1] = choice - same_weights.size

        proposed = self.current.copy()
        for offset, index in enumerate(indices):
            proposed[self.start + offset, self.slots[0]] = self.options[offset][index, 0]
            proposed[self.start + offset, self.slots[1]] = self.options[offset][index, 1]
        if np.array_equal(proposed, self.current):
            raise RuntimeError("the non-self proposal returned the forbidden path")
        new_indices = [int(index) for index in indices]
        _, reverse_different = self._different_forward(new_indices)
        new_different_partition = float(logsumexp(reverse_different[-1]))
        log_hastings = (
            self.log_probability(self.current)
            - self.log_probability(proposed)
            + old_different_partition
            - new_different_partition
        )
        return proposed, float(log_hastings)


def propose_conditional_gp_tracklets(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
    *,
    length: int = 3,
) -> tuple[IntArray, FloatArray]:
    """Update two slots over three frames with exact non-self MH correction."""

    if model.n_slots < 2 or length < 2 or model.sequence.n_frames < length:
        raise ValueError("conditional-GP tracklets require two slots and a valid window")
    proposed = current[rows].copy()
    hastings = np.zeros(rows.size, dtype=float)
    for local_index, row in enumerate(rows):
        choices, probabilities = _invariant_windows(
            (current[row] != ABSENT).tobytes(),
            model.sequence.n_frames,
            model.n_slots,
            model.sequence.cardinalities,
            length,
        )
        if not choices:
            continue
        pair, start = choices[int(rng.choice(len(choices), p=probabilities))]
        window = ConditionalGPWindow(model, current[row], pair, start, length)
        proposed[local_index], hastings[local_index] = window.propose_nonself(rng)
    return proposed, hastings
