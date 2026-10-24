"""Enumerable ULM posterior with explicit birth/death slots.

Each latent slot has a contiguous existence mask.  Association values use
three distinct states: ``-2`` means that the slot does not exist, ``-1`` means
that it exists but was missed, and a non-negative value is an assigned
detection index.  Unassigned detections are uniform Poisson clutter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log, prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .data import DetectionSequence
from .null_association import (
    NullAssociationMaternModel,
    enumerate_frame_null_associations,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

ABSENT = -2
MISSED = -1


@dataclass(frozen=True)
class BirthDeathPrior:
    """Three-state prior: unborn, alive, then permanently dead.

    A slot is alive in frame zero with probability ``initial_probability``.
    While unborn, it is born before the next frame with probability
    ``birth_probability``.  While alive, it survives to the next frame with
    probability ``survival_probability``.  Death is absorbing.
    """

    initial_probability: float = 0.50
    birth_probability: float = 0.25
    survival_probability: float = 0.80

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_probability", self.initial_probability),
            ("birth_probability", self.birth_probability),
            ("survival_probability", self.survival_probability),
        ):
            if not 0 < value < 1 or not np.isfinite(value):
                raise ValueError(f"{name} must lie strictly between zero and one")

    def log_probability(self, masks: ArrayLike) -> FloatArray:
        """Return independent-slot log probabilities for contiguous masks."""

        raw = np.asarray(masks)
        if raw.ndim < 1:
            raise ValueError("masks must have at least one dimension")
        if raw.shape[-1] < 1:
            raise ValueError("masks must contain at least one frame")
        boolean = raw.astype(bool, copy=False)
        if not np.array_equal(raw, boolean):
            raise ValueError("masks must be boolean or contain only zero and one")

        flat = boolean.reshape(-1, boolean.shape[-1])
        scores = np.empty(flat.shape[0], dtype=float)
        n_frames = flat.shape[1]
        for row_index, row in enumerate(flat):
            alive = np.flatnonzero(row)
            if alive.size == 0:
                scores[row_index] = log(1.0 - self.initial_probability) + (
                    (n_frames - 1) * log(1.0 - self.birth_probability)
                )
                continue
            birth = int(alive[0])
            death = int(alive[-1])
            if not np.all(row[birth : death + 1]) or np.any(row[:birth]) or np.any(
                row[death + 1 :]
            ):
                raise ValueError("each slot must have one contiguous existence interval")
            if birth == 0:
                score = log(self.initial_probability)
            else:
                score = (
                    log(1.0 - self.initial_probability)
                    + (birth - 1) * log(1.0 - self.birth_probability)
                    + log(self.birth_probability)
                )
            score += (death - birth) * log(self.survival_probability)
            if death < n_frames - 1:
                score += log(1.0 - self.survival_probability)
            scores[row_index] = score
        return scores.reshape(boolean.shape[:-1])


def enumerate_contiguous_existence_masks(n_frames: int) -> BoolArray:
    """Enumerate the never-alive mask and every non-empty interval mask."""

    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    masks = [np.zeros(n_frames, dtype=bool)]
    for birth in range(n_frames):
        for death in range(birth, n_frames):
            mask = np.zeros(n_frames, dtype=bool)
            mask[birth : death + 1] = True
            masks.append(mask)
    return np.stack(masks)


def enumerate_frame_slot_associations(
    existence_mask: ArrayLike, n_detections: int
) -> IntArray:
    """Enumerate frame associations compatible with one slot-existence mask."""

    raw = np.asarray(existence_mask)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError("existence_mask must be a non-empty vector")
    boolean = raw.astype(bool, copy=False)
    if not np.array_equal(raw, boolean):
        raise ValueError("existence_mask must be boolean or contain only zero and one")
    active = np.flatnonzero(boolean)
    if active.size == 0:
        return np.full((1, boolean.size), ABSENT, dtype=np.int64)
    active_states = enumerate_frame_null_associations(active.size, n_detections)
    states = np.full((active_states.shape[0], boolean.size), ABSENT, dtype=np.int64)
    states[:, active] = active_states
    return states


@dataclass(frozen=True)
class SlotExistencePosterior:
    """Normalized exact posterior over masks and three-state associations."""

    associations: IntArray
    existence_masks: BoolArray
    log_probabilities: FloatArray
    probabilities: FloatArray

    def __post_init__(self) -> None:
        if self.associations.ndim != 3:
            raise ValueError("associations must have shape (H, T, K)")
        if self.existence_masks.shape != self.associations.shape:
            raise ValueError("existence_masks and associations must have matching shapes")
        if not np.array_equal(self.existence_masks, self.associations != ABSENT):
            raise ValueError("existence_masks must agree with the absent sentinel")
        n_hypotheses = self.associations.shape[0]
        if self.log_probabilities.shape != (n_hypotheses,):
            raise ValueError("log_probabilities has the wrong shape")
        if self.probabilities.shape != (n_hypotheses,):
            raise ValueError("probabilities has the wrong shape")
        if np.any(self.probabilities < 0) or not np.isclose(
            np.sum(self.probabilities), 1.0
        ):
            raise ValueError("probabilities must be non-negative and sum to one")

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def normalized_entropy(self) -> float:
        positive = self.probabilities[self.probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        return 0.0 if self.probabilities.size <= 1 else entropy / log(
            self.probabilities.size
        )

    def credible_set_size(self, mass: float = 0.95) -> int:
        if not 0 < mass <= 1:
            raise ValueError("mass must lie in (0, 1]")
        ordered = np.sort(self.probabilities)[::-1]
        return int(np.searchsorted(np.cumsum(ordered), mass, side="left") + 1)


@dataclass(frozen=True)
class SlotEventMarginals:
    """Posterior summaries for slot existence, misses, clutter, birth and death."""

    existence_probabilities: FloatArray
    missed_probabilities: FloatArray
    expected_clutter: FloatArray
    birth_time_probabilities: FloatArray
    death_time_probabilities: FloatArray


class SlotExistenceMaternModel:
    r"""Collapsed Matérn model with explicit births, deaths and missed detections.

    Conditional on the existence masks :math:`z`, the finite-set factor is

    .. math::

       e^{-T\lambda_c} p_D^{n_A}(1-p_D)^{n_M}
       (\lambda_c/|R|)^{n_C}.

    The Matérn path is analytically integrated using assigned detections only.
    Slot labels are exchangeable because every slot shares the same priors.
    """

    def __init__(
        self,
        sequence: DetectionSequence,
        *,
        n_slots: int,
        localization_std: float,
        coordinate_std: ArrayLike,
        length_scale: float,
        detection_probability: float,
        expected_clutter_per_frame: float,
        roi_lower: ArrayLike,
        roi_upper: ArrayLike,
        birth_death_prior: BirthDeathPrior | None = None,
        nu: float = 1.5,
        relative_jitter: float = 1e-10,
    ) -> None:
        if n_slots < 1:
            raise ValueError("n_slots must be positive")
        self.sequence = sequence
        self.n_slots = int(n_slots)
        self.birth_death_prior = (
            BirthDeathPrior() if birth_death_prior is None else birth_death_prior
        )
        self._gp_model = NullAssociationMaternModel(
            sequence,
            n_particles=n_slots,
            localization_std=localization_std,
            coordinate_std=coordinate_std,
            length_scale=length_scale,
            detection_probability=detection_probability,
            expected_clutter_per_frame=expected_clutter_per_frame,
            roi_lower=roi_lower,
            roi_upper=roi_upper,
            nu=nu,
            relative_jitter=relative_jitter,
            anchor_first_frame=False,
        )
        self.localization_std = self._gp_model.localization_std
        self.coordinate_std = self._gp_model.coordinate_std
        self.length_scale = self._gp_model.length_scale
        self.detection_probability = self._gp_model.detection_probability
        self.expected_clutter_per_frame = self._gp_model.expected_clutter_per_frame
        self.roi_lower = self._gp_model.roi_lower
        self.roi_upper = self._gp_model.roi_upper
        self.roi_volume = self._gp_model.roi_volume
        self.clutter_intensity = self._gp_model.clutter_intensity
        self.nu = self._gp_model.nu
        self.relative_jitter = self._gp_model.relative_jitter

        self._interval_masks = enumerate_contiguous_existence_masks(sequence.n_frames)
        self._interval_log_probabilities = np.asarray(
            self.birth_death_prior.log_probability(self._interval_masks), dtype=float
        )
        self._mask_combinations: BoolArray | None = None
        self._frame_state_cache: dict[tuple[tuple[bool, ...], int], IntArray] = {}
        self._frame_log_weight_cache: dict[
            tuple[tuple[bool, ...], int], FloatArray
        ] = {}
        (
            self._joint_existence_states,
            self._joint_transition_log_probabilities,
            self._base_forward_log_weights,
            self._log_base_partition_without_void,
        ) = self._build_sequential_base()

    @property
    def n_interval_masks(self) -> int:
        return int(self._interval_masks.shape[0])

    @property
    def base_log_normalizer(self) -> float:
        return self._log_base_partition_without_void - (
            self.sequence.n_frames * self.expected_clutter_per_frame
        )

    def interval_masks(self) -> BoolArray:
        return self._interval_masks.copy()

    def interval_log_probabilities(self) -> FloatArray:
        """Return prior log probabilities in ``interval_masks`` order."""

        return self._interval_log_probabilities.copy()

    def mask_combinations(self) -> BoolArray:
        if self._mask_combinations is None:
            interval_indices = np.asarray(
                list(
                    product(
                        range(self._interval_masks.shape[0]), repeat=self.n_slots
                    )
                ),
                dtype=np.int64,
            )
            self._mask_combinations = np.transpose(
                self._interval_masks[interval_indices], (0, 2, 1)
            )
        return self._mask_combinations.copy()

    def _build_sequential_base(
        self,
    ) -> tuple[IntArray, FloatArray, FloatArray, float]:
        """Build an exact three-state HMM for the base mask distribution."""

        # Per-slot states are unborn, alive, and permanently dead.
        joint_states = np.asarray(
            list(product(range(3), repeat=self.n_slots)), dtype=np.int64
        )
        prior = self.birth_death_prior
        initial = np.asarray(
            [1.0 - prior.initial_probability, prior.initial_probability, 0.0]
        )
        transition = np.asarray(
            [
                [1.0 - prior.birth_probability, prior.birth_probability, 0.0],
                [0.0, prior.survival_probability, 1.0 - prior.survival_probability],
                [0.0, 0.0, 1.0],
            ]
        )
        with np.errstate(divide="ignore"):
            initial_log = np.sum(np.log(initial[joint_states]), axis=1)
        transition_probabilities = np.ones(
            (joint_states.shape[0], joint_states.shape[0]), dtype=float
        )
        for slot in range(self.n_slots):
            transition_probabilities *= transition[
                joint_states[:, slot, None], joint_states[None, :, slot]
            ]
        with np.errstate(divide="ignore"):
            transition_log = np.log(transition_probabilities)

        emissions = np.empty(
            (self.sequence.n_frames, joint_states.shape[0]), dtype=float
        )
        active_masks = joint_states == 1
        for time_index, cardinality in enumerate(self.sequence.cardinalities):
            for state_index, active_mask in enumerate(active_masks):
                emissions[time_index, state_index] = float(
                    logsumexp(
                        self.frame_log_count_weights(active_mask, cardinality)
                    )
                )

        forward = np.empty_like(emissions)
        forward[0] = initial_log + emissions[0]
        for time_index in range(1, self.sequence.n_frames):
            forward[time_index] = emissions[time_index] + logsumexp(
                forward[time_index - 1, :, None] + transition_log,
                axis=0,
            )
        return (
            joint_states,
            transition_log,
            forward,
            float(logsumexp(forward[-1])),
        )

    def frame_states(self, existence_mask: ArrayLike, n_detections: int) -> IntArray:
        raw = np.asarray(existence_mask)
        boolean = raw.astype(bool, copy=False)
        if boolean.shape != (self.n_slots,):
            raise ValueError("existence_mask must have one value per slot")
        if not np.array_equal(raw, boolean):
            raise ValueError("existence_mask must be boolean or contain only zero and one")
        key = (tuple(bool(value) for value in boolean), int(n_detections))
        cached = self._frame_state_cache.get(key)
        if cached is None:
            cached = enumerate_frame_slot_associations(boolean, n_detections)
            self._frame_state_cache[key] = cached
        return cached.copy()

    def frame_log_count_weights(
        self, existence_mask: ArrayLike, n_detections: int
    ) -> FloatArray:
        raw = np.asarray(existence_mask)
        boolean = raw.astype(bool, copy=False)
        if boolean.shape != (self.n_slots,):
            raise ValueError("existence_mask must have one value per slot")
        if not np.array_equal(raw, boolean):
            raise ValueError("existence_mask must be boolean or contain only zero and one")
        key = (tuple(bool(value) for value in boolean), int(n_detections))
        cached = self._frame_log_weight_cache.get(key)
        if cached is None:
            states = self.frame_states(boolean, n_detections)
            assigned = np.count_nonzero(states >= 0, axis=1)
            active = int(np.count_nonzero(boolean))
            clutter = n_detections - assigned
            cached = np.asarray(
                assigned * log(self.detection_probability)
                + (active - assigned) * log(1.0 - self.detection_probability)
                + clutter * log(self.clutter_intensity),
                dtype=float,
            )
            self._frame_log_weight_cache[key] = cached
        return cached.copy()

    def _validate_associations(self, associations: ArrayLike) -> tuple[IntArray, BoolArray]:
        raw = np.asarray(associations)
        expected = (self.sequence.n_frames, self.n_slots)
        if raw.ndim != 3 or raw.shape[1:] != expected:
            raise ValueError(f"associations must have shape (S, {expected[0]}, {expected[1]})")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("associations must contain integer detection indices")
        integer = raw.astype(np.int64, copy=False)
        masks = integer != ABSENT
        for slot in range(self.n_slots):
            self.birth_death_prior.log_probability(masks[:, :, slot])
        for time_index, cardinality in enumerate(self.sequence.cardinalities):
            frame = integer[:, time_index]
            if np.any(frame < ABSENT) or np.any(frame >= cardinality):
                raise ValueError("an association contains an invalid detection index")
            for first in range(self.n_slots):
                for second in range(first + 1, self.n_slots):
                    duplicate = (
                        (frame[:, first] >= 0)
                        & (frame[:, second] >= 0)
                        & (frame[:, first] == frame[:, second])
                    )
                    if np.any(duplicate):
                        raise ValueError("non-null associations must be injective")
        return integer, masks

    def log_existence_prior(self, associations: ArrayLike) -> FloatArray:
        integer, masks = self._validate_associations(associations)
        scores = np.zeros(integer.shape[0], dtype=float)
        for slot in range(self.n_slots):
            scores += self.birth_death_prior.log_probability(masks[:, :, slot])
        return scores

    def log_count_density(
        self, associations: ArrayLike, *, include_poisson_void: bool = True
    ) -> FloatArray:
        integer, masks = self._validate_associations(associations)
        assigned = np.count_nonzero(integer >= 0, axis=(1, 2))
        missed = np.count_nonzero(masks, axis=(1, 2)) - assigned
        clutter = sum(self.sequence.cardinalities) - assigned
        scores = (
            assigned * log(self.detection_probability)
            + missed * log(1.0 - self.detection_probability)
            + clutter * log(self.clutter_intensity)
        )
        if include_poisson_void:
            scores = scores - (
                self.sequence.n_frames * self.expected_clutter_per_frame
            )
        return np.asarray(scores, dtype=float)

    def _as_gp_associations(self, associations: IntArray) -> IntArray:
        converted = associations.copy()
        converted[converted == ABSENT] = MISSED
        return converted

    def log_marginal_density(self, associations: ArrayLike) -> FloatArray:
        integer, _ = self._validate_associations(associations)
        return self._gp_model.log_marginal_density(self._as_gp_associations(integer))

    def log_joint_density(self, associations: ArrayLike) -> FloatArray:
        return (
            self.log_existence_prior(associations)
            + self.log_count_density(associations)
            + self.log_marginal_density(associations)
        )

    def log_base_probability(self, associations: ArrayLike) -> FloatArray:
        return (
            self.log_existence_prior(associations)
            + self.log_count_density(associations, include_poisson_void=False)
            - self._log_base_partition_without_void
        )

    def sample_base_associations(
        self, *, n_samples: int, rng: np.random.Generator
    ) -> IntArray:
        """Draw exactly from the normalized existence-and-count base density.

        Existence masks are drawn by forward-filtering/backward-sampling the
        three-state birth--alive--death HMM.  This is linear in the number of
        frames and avoids enumerating every combination of slot intervals.
        """

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        n_frames = self.sequence.n_frames
        state_indices = np.empty((n_samples, n_frames), dtype=np.int64)
        final_probabilities = np.exp(
            self._base_forward_log_weights[-1]
            - self._log_base_partition_without_void
        )
        state_indices[:, -1] = rng.choice(
            self._joint_existence_states.shape[0],
            size=n_samples,
            p=final_probabilities,
        )
        for time_index in range(n_frames - 2, -1, -1):
            for next_state in np.unique(state_indices[:, time_index + 1]):
                rows = np.flatnonzero(
                    state_indices[:, time_index + 1] == next_state
                )
                log_probabilities = (
                    self._base_forward_log_weights[time_index]
                    + self._joint_transition_log_probabilities[:, next_state]
                )
                probabilities = np.exp(
                    log_probabilities - logsumexp(log_probabilities)
                )
                state_indices[rows, time_index] = rng.choice(
                    self._joint_existence_states.shape[0],
                    size=rows.size,
                    p=probabilities,
                )
        masks = self._joint_existence_states[state_indices] == 1
        associations = np.empty(masks.shape, dtype=np.int64)
        for time_index, cardinality in enumerate(self.sequence.cardinalities):
            frame_masks, inverse = np.unique(
                masks[:, time_index], axis=0, return_inverse=True
            )
            for mask_index, frame_mask in enumerate(frame_masks):
                rows = np.flatnonzero(inverse == mask_index)
                states = self.frame_states(frame_mask, cardinality)
                weights = self.frame_log_count_weights(frame_mask, cardinality)
                state_probabilities = np.exp(weights - logsumexp(weights))
                selected = rng.choice(
                    states.shape[0], size=rows.size, p=state_probabilities
                )
                associations[rows, time_index] = states[selected]
        return associations

    def conditional_path_means(self, associations: ArrayLike) -> FloatArray:
        integer, masks = self._validate_associations(associations)
        means = self._gp_model.conditional_path_means(
            self._as_gp_associations(integer)
        )
        return np.where(masks[..., None], means, np.nan)

    def sample_conditional_paths(
        self, associations: ArrayLike, *, rng: np.random.Generator
    ) -> FloatArray:
        integer, masks = self._validate_associations(associations)
        paths = self._gp_model.sample_conditional_paths(
            self._as_gp_associations(integer), rng=rng
        )
        return np.where(masks[..., None], paths, np.nan)


def slot_existence_support_size(model: SlotExistenceMaternModel) -> int:
    """Return the exact number of mask-and-association hypotheses."""

    total = 0
    for masks in model.mask_combinations():
        total += prod(
            model.frame_states(masks[time_index], cardinality).shape[0]
            for time_index, cardinality in enumerate(model.sequence.cardinalities)
        )
    return int(total)


def enumerate_slot_existence_support(
    model: SlotExistenceMaternModel,
    *,
    max_hypotheses: int = 1_000_000,
) -> tuple[IntArray, BoolArray]:
    """Enumerate all labelled slot masks and compatible associations."""

    n_hypotheses = slot_existence_support_size(model)
    if n_hypotheses > max_hypotheses:
        raise ValueError(
            f"enumeration would create {n_hypotheses:,} hypotheses; "
            "increase max_hypotheses deliberately or use the SMC sampler"
        )
    associations = np.empty(
        (n_hypotheses, model.sequence.n_frames, model.n_slots), dtype=np.int64
    )
    masks_out = np.empty_like(associations, dtype=bool)
    cursor = 0
    for masks in model.mask_combinations():
        frame_states = tuple(
            model.frame_states(masks[time_index], cardinality)
            for time_index, cardinality in enumerate(model.sequence.cardinalities)
        )
        for state_indices in product(
            *(range(states.shape[0]) for states in frame_states)
        ):
            for time_index, state_index in enumerate(state_indices):
                associations[cursor, time_index] = frame_states[time_index][state_index]
            masks_out[cursor] = masks
            cursor += 1
    if cursor != n_hypotheses:
        raise RuntimeError("internal support-size mismatch")
    return associations, masks_out


def enumerate_slot_existence_posterior(
    sequence: DetectionSequence,
    *,
    n_slots: int,
    localization_std: float,
    coordinate_std: ArrayLike,
    length_scale: float,
    detection_probability: float,
    expected_clutter_per_frame: float,
    roi_lower: ArrayLike,
    roi_upper: ArrayLike,
    birth_death_prior: BirthDeathPrior | None = None,
    nu: float = 1.5,
    relative_jitter: float = 1e-10,
    max_hypotheses: int = 1_000_000,
) -> SlotExistencePosterior:
    """Exactly enumerate the posterior over birth/death slots."""

    model = SlotExistenceMaternModel(
        sequence,
        n_slots=n_slots,
        localization_std=localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=expected_clutter_per_frame,
        roi_lower=roi_lower,
        roi_upper=roi_upper,
        birth_death_prior=birth_death_prior,
        nu=nu,
        relative_jitter=relative_jitter,
    )
    associations, masks = enumerate_slot_existence_support(
        model, max_hypotheses=max_hypotheses
    )
    log_scores = model.log_joint_density(associations)
    normalized = log_scores - logsumexp(log_scores)
    return SlotExistencePosterior(
        associations=associations,
        existence_masks=masks,
        log_probabilities=normalized,
        probabilities=np.exp(normalized),
    )


def reference_slot_association(
    sequence: DetectionSequence, existence_mask: ArrayLike
) -> IntArray:
    """Recover simulated truth using ``-2`` for absent and ``-1`` for missed."""

    if sequence.reference_ids is None:
        raise ValueError("reference_ids are required")
    raw = np.asarray(existence_mask)
    if raw.ndim != 2 or raw.shape[0] != sequence.n_frames:
        raise ValueError("existence_mask must have shape (T, K)")
    masks = raw.astype(bool, copy=False)
    if not np.array_equal(raw, masks):
        raise ValueError("existence_mask must be boolean or contain only zero and one")
    n_slots = masks.shape[1]
    prior = BirthDeathPrior()
    for slot in range(n_slots):
        prior.log_probability(masks[None, :, slot])
    association = np.where(masks, MISSED, ABSENT).astype(np.int64)
    for time_index, frame_ids in enumerate(sequence.reference_ids):
        for detection_index, identity in enumerate(frame_ids):
            if identity < 0:
                continue
            if identity >= n_slots:
                raise ValueError("a reference ID exceeds the number of slots")
            if not masks[time_index, identity]:
                raise ValueError("a detection is labelled as an absent slot")
            if association[time_index, identity] >= 0:
                raise ValueError("a frame contains a duplicate reference ID")
            association[time_index, identity] = detection_index
    return association


def slot_event_marginals(
    associations: ArrayLike,
    sequence: DetectionSequence,
    *,
    probabilities: ArrayLike | None = None,
) -> SlotEventMarginals:
    """Summarize existence, miss, clutter, birth and last-alive probabilities.

    Birth distributions have columns ``0..T-1`` plus ``never``.  Death-time
    distributions use the last alive frame ``0..T-1``; column ``T-1`` thus
    includes slots surviving through the observation horizon, and the final
    column again denotes ``never existed``.
    """

    integer = np.asarray(associations)
    if integer.ndim != 3 or integer.shape[1] != sequence.n_frames:
        raise ValueError("associations must have shape (S, T, K)")
    if integer.shape[0] == 0:
        raise ValueError("at least one association sample is required")
    if probabilities is None:
        weights = np.full(integer.shape[0], 1.0 / integer.shape[0])
    else:
        weights = np.asarray(probabilities, dtype=float)
        if weights.shape != (integer.shape[0],):
            raise ValueError("probabilities has the wrong shape")
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")

    masks = integer != ABSENT
    existence = np.tensordot(weights, masks, axes=(0, 0))
    missed = np.tensordot(weights, integer == MISSED, axes=(0, 0))
    assigned_counts = np.count_nonzero(integer >= 0, axis=2)
    cardinalities = np.asarray(sequence.cardinalities, dtype=float)
    expected_clutter = cardinalities - np.tensordot(
        weights, assigned_counts, axes=(0, 0)
    )

    n_frames = sequence.n_frames
    n_slots = integer.shape[2]
    birth = np.zeros((n_slots, n_frames + 1), dtype=float)
    death = np.zeros_like(birth)
    for sample_index, sample_masks in enumerate(masks):
        for slot in range(n_slots):
            alive = np.flatnonzero(sample_masks[:, slot])
            if alive.size == 0:
                birth[slot, n_frames] += weights[sample_index]
                death[slot, n_frames] += weights[sample_index]
            else:
                birth[slot, int(alive[0])] += weights[sample_index]
                death[slot, int(alive[-1])] += weights[sample_index]
    return SlotEventMarginals(
        existence_probabilities=np.asarray(existence, dtype=float),
        missed_probabilities=np.asarray(missed, dtype=float),
        expected_clutter=np.asarray(expected_clutter, dtype=float),
        birth_time_probabilities=birth,
        death_time_probabilities=death,
    )
