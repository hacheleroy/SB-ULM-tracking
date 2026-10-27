"""Annealed SMC for explicit ULM birth/death slots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .conditional_pair_pg import conditional_pair_pg_move
from .conditional_tracklet_moves import propose_conditional_gp_tracklets
from .data import DetectionSequence
from .paired_moves import local_pair_multiple_try_move, multiple_try_paired_move
from .slot_existence import (
    ABSENT,
    MISSED,
    BirthDeathPrior,
    SlotExistenceMaternModel,
)
from .smc import _next_beta, _normalized_weights, systematic_resample
from .tracklet_moves import propose_guided_tracklets

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class BirthDeathSMCObserver(Protocol):
    """Optional read-only diagnostics; callbacks must not consume sampler RNG."""

    def begin_stage(self, step: int, beta: float) -> None: ...

    def record_resampling(
        self, associations: IntArray, weights: FloatArray, ancestors: IntArray
    ) -> None: ...

    def record_pair(
        self, before: IntArray, proposed: IntArray, accepted: BoolArray, pg_selected: BoolArray
    ) -> None: ...

    def end_stage(self, associations: IntArray) -> None: ...


@dataclass(frozen=True)
class BirthDeathSMCConfig:
    """Numerical controls and reversible-move mixture."""

    n_particles: int = 8_192
    target_ess_fraction: float = 0.80
    mh_sweeps: int = 10
    final_mh_sweeps: int = 10
    association_move_probability: float = 0.45
    interval_move_probability: float = 0.45
    slot_swap_probability: float = 0.10
    blocked_slot_move_probability: float = 0.0
    paired_slot_move_probability: float = 0.0
    paired_slot_candidates: int = 2
    paired_slot_scope: str = "local"
    paired_slot_path_particles: int = 8
    paired_slot_pg_fraction: float = 1.0
    paired_slot_pg_beta_switch: float | None = None
    paired_slot_pg_early_fraction: float | None = None
    paired_slot_pg_ancestor_sampling: bool = False
    paired_slot_suffix_pg_final_fraction: float = 0.0
    paired_slot_suffix_pg_start_frame: int = 5
    tracklet_move_probability: float = 0.0
    tracklet_proposal_kind: str = "displacement"
    max_tempering_steps: int = 128
    bisection_iterations: int = 48
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least two")
        if not 0 < self.target_ess_fraction < 1:
            raise ValueError("target_ess_fraction must lie in (0, 1)")
        if self.mh_sweeps < 0 or self.final_mh_sweeps < 0:
            raise ValueError("mh_sweeps and final_mh_sweeps must be non-negative")
        probabilities = np.asarray(
            [
                self.association_move_probability,
                self.interval_move_probability,
                self.slot_swap_probability,
                self.blocked_slot_move_probability,
                self.paired_slot_move_probability,
                self.tracklet_move_probability,
            ],
            dtype=float,
        )
        if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
            raise ValueError("move probabilities must be finite and non-negative")
        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("move probabilities must sum to one")
        if self.max_tempering_steps < 1 or self.bisection_iterations < 1:
            raise ValueError("tempering and bisection iteration counts must be positive")
        if self.paired_slot_candidates < 2:
            raise ValueError("paired_slot_candidates must be at least two")
        if self.paired_slot_path_particles < 2:
            raise ValueError("paired_slot_path_particles must be at least two")
        if self.paired_slot_scope not in ("local", "full_sequence", "conditional_smc"):
            raise ValueError("unknown paired_slot_scope")
        if not np.isfinite(self.paired_slot_pg_fraction) or not 0 <= self.paired_slot_pg_fraction <= 1:
            raise ValueError("paired_slot_pg_fraction must lie in [0, 1]")
        if self.paired_slot_scope != "conditional_smc" and self.paired_slot_pg_fraction != 1:
            raise ValueError("paired_slot_pg_fraction requires conditional_smc scope")
        if (self.paired_slot_pg_beta_switch is None) != (self.paired_slot_pg_early_fraction is None):
            raise ValueError("PG beta switch and early fraction must be set together")
        if self.paired_slot_pg_beta_switch is not None:
            if self.paired_slot_scope != "conditional_smc":
                raise ValueError("PG beta schedule requires conditional_smc scope")
            if not np.isfinite(self.paired_slot_pg_beta_switch) or not (
                0 < self.paired_slot_pg_beta_switch <= 1
            ):
                raise ValueError("PG beta switch must lie in (0, 1]")
            if not np.isfinite(self.paired_slot_pg_early_fraction) or not (
                0 <= self.paired_slot_pg_early_fraction <= 1
            ):
                raise ValueError("PG early fraction must lie in [0, 1]")
        if self.paired_slot_pg_ancestor_sampling and self.paired_slot_scope != "conditional_smc":
            raise ValueError("PG ancestor sampling requires conditional_smc scope")
        if not np.isfinite(self.paired_slot_suffix_pg_final_fraction) or not (
            0 <= self.paired_slot_suffix_pg_final_fraction <= 1
        ):
            raise ValueError("suffix PG fraction must lie in [0, 1]")
        if self.paired_slot_suffix_pg_final_fraction:
            if self.paired_slot_scope != "conditional_smc":
                raise ValueError("suffix PG requires conditional_smc scope")
            if self.paired_slot_pg_fraction + self.paired_slot_suffix_pg_final_fraction > 1:
                raise ValueError("full and suffix PG fractions must sum to at most one")
        if self.paired_slot_suffix_pg_start_frame < 1:
            raise ValueError("suffix PG must condition on at least one frame")
        if self.tracklet_proposal_kind not in ("displacement", "conditional_gp_nonself"):
            raise ValueError("unknown tracklet_proposal_kind")


@dataclass(frozen=True)
class BirthDeathSMCStageDiagnostics:
    """Diagnostics for one tempering/resample/rejuvenation stage."""

    step: int
    beta: float
    beta_increment: float
    effective_sample_size: float
    unique_parent_count: int
    unique_state_count: int
    mh_sweeps: int
    mh_acceptance_rate: float
    association_acceptance_rate: float
    interval_acceptance_rate: float
    slot_swap_acceptance_rate: float
    blocked_slot_acceptance_rate: float
    paired_slot_acceptance_rate: float
    paired_slot_state_change_rate: float
    tracklet_acceptance_rate: float
    tracklet_state_change_rate: float
    paired_slot_count_signature_change_rate: float = 0.0
    paired_slot_pg_selected_fraction: float = 0.0
    paired_slot_pg_count_signature_change_rate: float = 0.0
    paired_slot_suffix_pg_selected_fraction: float = 0.0
    paired_slot_suffix_pg_count_signature_change_rate: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BirthDeathSMCResult:
    """Unweighted posterior draws with explicit slot masks."""

    associations: IntArray
    existence_masks: BoolArray
    latent_paths: FloatArray
    log_marginal_densities: FloatArray
    log_count_densities: FloatArray
    log_existence_priors: FloatArray
    log_evidence_estimate: float
    stages: tuple[BirthDeathSMCStageDiagnostics, ...]
    config: BirthDeathSMCConfig

    def __post_init__(self) -> None:
        n_samples = self.config.n_particles
        if self.associations.ndim != 3 or self.associations.shape[0] != n_samples:
            raise ValueError("associations must contain one state per SMC particle")
        if self.existence_masks.shape != self.associations.shape:
            raise ValueError("existence_masks and associations have incompatible shapes")
        if not np.array_equal(self.existence_masks, self.associations != ABSENT):
            raise ValueError("existence_masks must agree with the absent sentinel")
        if self.latent_paths.shape[:3] != self.associations.shape:
            raise ValueError("latent_paths and associations have incompatible shapes")
        for values in (
            self.log_marginal_densities,
            self.log_count_densities,
            self.log_existence_priors,
        ):
            if values.shape != (n_samples,):
                raise ValueError("a log-density vector has the wrong shape")
        if not self.stages or not np.isclose(self.stages[-1].beta, 1.0):
            raise ValueError("the SMC result must terminate at beta=1")

    @property
    def n_tempering_steps(self) -> int:
        return len(self.stages)


def _frame_lookup(states: IntArray) -> dict[bytes, int]:
    return {state.tobytes(): index for index, state in enumerate(states)}


def _propose_association_moves(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray]:
    proposed = current.copy()
    log_hastings = np.zeros(current.shape[0], dtype=float)
    if rows.size == 0:
        return proposed, log_hastings
    selected_frames = rng.integers(model.sequence.n_frames, size=rows.size)
    for time_index in np.unique(selected_frames):
        frame_rows = rows[selected_frames == time_index]
        cardinality = model.sequence.cardinalities[int(time_index)]
        frame_masks, inverse = np.unique(
            current[frame_rows, time_index] != ABSENT, axis=0, return_inverse=True
        )
        for mask_index, mask in enumerate(frame_masks):
            local_rows = frame_rows[inverse == mask_index]
            states = model.frame_states(mask, cardinality)
            if states.shape[0] <= 1:
                continue
            lookup = _frame_lookup(states)
            current_indices = np.asarray(
                [lookup[current[row, time_index].tobytes()] for row in local_rows],
                dtype=np.int64,
            )
            offsets = rng.integers(1, states.shape[0], size=local_rows.size)
            proposed_indices = (current_indices + offsets) % states.shape[0]
            proposed[local_rows, time_index] = states[proposed_indices]
    return proposed, log_hastings


def _propose_interval_moves(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray]:
    """Replace one birth/death interval and redraw changed frame states.

    The interval proposal is symmetric.  When active cardinality changes, the
    whole frame association is redrawn uniformly.  The returned Hastings term
    exactly accounts for the different numbers of compatible frame states.
    """

    proposed = current.copy()
    log_hastings = np.zeros(current.shape[0], dtype=float)
    if rows.size == 0:
        return proposed, log_hastings
    interval_masks = model.interval_masks()
    interval_lookup = {mask.tobytes(): index for index, mask in enumerate(interval_masks)}
    selected_slots = rng.integers(model.n_slots, size=rows.size)
    offsets = rng.integers(1, interval_masks.shape[0], size=rows.size)

    for local_index, row in enumerate(rows):
        slot = int(selected_slots[local_index])
        old_masks = current[row] != ABSENT
        current_interval = interval_lookup[old_masks[:, slot].tobytes()]
        proposed_interval = (current_interval + int(offsets[local_index])) % interval_masks.shape[0]
        new_masks = old_masks.copy()
        new_masks[:, slot] = interval_masks[proposed_interval]
        changed_frames = np.flatnonzero(old_masks[:, slot] != new_masks[:, slot])
        for time_index in changed_frames:
            cardinality = model.sequence.cardinalities[int(time_index)]
            old_states = model.frame_states(old_masks[time_index], cardinality)
            new_states = model.frame_states(new_masks[time_index], cardinality)
            selected = int(rng.integers(new_states.shape[0]))
            proposed[row, time_index] = new_states[selected]
            log_hastings[row] += np.log(new_states.shape[0]) - np.log(
                old_states.shape[0]
            )
    return proposed, log_hastings


def _propose_slot_swaps(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray]:
    proposed = current.copy()
    log_hastings = np.zeros(current.shape[0], dtype=float)
    if rows.size == 0:
        return proposed, log_hastings
    if model.n_slots < 2:
        raise ValueError("slot-swap moves require at least two slots")
    first_slots = rng.integers(model.n_slots, size=rows.size)
    offsets = rng.integers(1, model.n_slots, size=rows.size)
    second_slots = (first_slots + offsets) % model.n_slots
    for local_index, row in enumerate(rows):
        first = int(first_slots[local_index])
        second = int(second_slots[local_index])
        temporary = proposed[row, :, first].copy()
        proposed[row, :, first] = proposed[row, :, second]
        proposed[row, :, second] = temporary
    return proposed, log_hastings


def _propose_blocked_slot_moves(
    current: IntArray,
    rows: IntArray,
    model: SlotExistenceMaternModel,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray]:
    r"""Redraw one slot from its exact conditional base distribution.

    Conditional on every other slot, this proposal jointly samples the chosen
    slot's birth/death interval and all of its frame associations under the
    existence-and-count base density.  The returned Hastings term cancels
    those two target factors exactly, so at :math:`\beta=0` every proposal is
    accepted and at later temperatures only the collapsed path density enters
    the Metropolis decision.
    """

    proposed = current.copy()
    log_hastings = np.zeros(current.shape[0], dtype=float)
    if rows.size == 0:
        return proposed, log_hastings

    interval_masks = model.interval_masks()
    interval_log_probabilities = model.interval_log_probabilities()
    interval_lookup = {
        mask.tobytes(): index for index, mask in enumerate(interval_masks)
    }
    selected_slots = rng.integers(model.n_slots, size=rows.size)
    log_missed = np.log1p(-model.detection_probability)
    log_assigned = np.log(model.detection_probability / model.clutter_intensity)

    for local_index, row in enumerate(rows):
        slot = int(selected_slots[local_index])
        other_slots = np.arange(model.n_slots) != slot
        available_by_frame: list[IntArray] = []
        active_log_partitions = np.empty(model.sequence.n_frames, dtype=float)
        for time_index, cardinality in enumerate(model.sequence.cardinalities):
            claimed = current[row, time_index, other_slots]
            claimed = claimed[claimed >= 0]
            available = np.setdiff1d(
                np.arange(cardinality, dtype=np.int64),
                claimed,
                assume_unique=True,
            )
            available_by_frame.append(available)
            active_log_partitions[time_index] = np.logaddexp(
                log_missed,
                -np.inf
                if available.size == 0
                else np.log(available.size) + log_assigned,
            )

        interval_log_weights = (
            interval_log_probabilities
            + interval_masks.astype(float) @ active_log_partitions
        )
        interval_probabilities = np.exp(
            interval_log_weights - logsumexp(interval_log_weights)
        )
        proposed_interval = int(
            rng.choice(interval_masks.shape[0], p=interval_probabilities)
        )
        new_mask = interval_masks[proposed_interval]
        proposed[row, :, slot] = ABSENT
        new_local_log_weight = float(
            interval_log_probabilities[proposed_interval]
        )
        for time_index in np.flatnonzero(new_mask):
            available = available_by_frame[int(time_index)]
            option_log_weights = np.concatenate(
                [
                    np.asarray([log_missed]),
                    np.full(available.size, log_assigned),
                ]
            )
            option_probabilities = np.exp(
                option_log_weights - logsumexp(option_log_weights)
            )
            selected = int(rng.choice(option_probabilities.size, p=option_probabilities))
            if selected == 0:
                proposed[row, time_index, slot] = MISSED
                new_local_log_weight += log_missed
            else:
                proposed[row, time_index, slot] = available[selected - 1]
                new_local_log_weight += log_assigned

        old_mask = current[row, :, slot] != ABSENT
        old_interval = interval_lookup[old_mask.tobytes()]
        old_local_log_weight = float(interval_log_probabilities[old_interval])
        for time_index in np.flatnonzero(old_mask):
            old_local_log_weight += (
                log_missed
                if current[row, time_index, slot] < 0
                else log_assigned
            )
        log_hastings[row] = old_local_log_weight - new_local_log_weight

    return proposed, log_hastings


def _propose_birth_death_moves(
    current: IntArray,
    model: SlotExistenceMaternModel,
    config: BirthDeathSMCConfig,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, IntArray]:
    move_probabilities = np.asarray(
        [
            config.association_move_probability,
            config.interval_move_probability,
            config.slot_swap_probability,
            config.blocked_slot_move_probability,
            config.paired_slot_move_probability,
            config.tracklet_move_probability,
        ]
    )
    if model.n_slots < 2 and move_probabilities[2] > 0:
        raise ValueError("slot_swap_probability must be zero for a one-slot model")
    if model.n_slots < 2 and move_probabilities[4] > 0:
        raise ValueError("paired_slot_move_probability must be zero for a one-slot model")
    if model.n_slots < 2 and move_probabilities[5] > 0:
        raise ValueError("tracklet_move_probability must be zero for a one-slot model")
    if model.sequence.n_frames < 3 and move_probabilities[5] > 0:
        raise ValueError("tracklet moves require at least three frames")
    move_types = rng.choice(6, size=current.shape[0], p=move_probabilities).astype(np.int64)
    proposed = current.copy()
    log_hastings = np.zeros(current.shape[0], dtype=float)
    proposal_functions = (
        _propose_association_moves,
        _propose_interval_moves,
        _propose_slot_swaps,
        _propose_blocked_slot_moves,
    )
    for move_type, proposal_function in enumerate(proposal_functions):
        rows = np.flatnonzero(move_types == move_type).astype(np.int64)
        if rows.size == 0:
            continue
        local_proposed, local_hastings = proposal_function(
            current, rows, model, rng
        )
        proposed[rows] = local_proposed[rows]
        log_hastings[rows] = local_hastings[rows]
    return proposed, log_hastings, move_types


def _paired_pg_fraction(config: BirthDeathSMCConfig, beta: float) -> float:
    """Fixed-in-advance, state-independent PG selection at this temperature."""

    if (config.paired_slot_pg_beta_switch is not None
            and beta < config.paired_slot_pg_beta_switch):
        assert config.paired_slot_pg_early_fraction is not None
        return config.paired_slot_pg_early_fraction
    return config.paired_slot_pg_fraction


def _metropolis_birth_death_rejuvenation(
    associations: IntArray,
    log_marginal_densities: FloatArray,
    log_count_densities: FloatArray,
    log_existence_priors: FloatArray,
    model: SlotExistenceMaternModel,
    *,
    beta: float,
    n_sweeps: int,
    config: BirthDeathSMCConfig,
    rng: np.random.Generator,
    observer: BirthDeathSMCObserver | None = None,
) -> tuple[
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    tuple[float, ...],
]:
    if n_sweeps == 0:
        return (
            associations,
            log_marginal_densities,
            log_count_densities,
            log_existence_priors,
            (0.0,) * 14,
        )

    accepted_total = 0
    proposed_total = 0
    accepted_by_move = np.zeros(6, dtype=np.int64)
    proposed_by_move = np.zeros(6, dtype=np.int64)
    paired_changes = 0
    paired_count_changes = 0
    paired_pg_selected = 0
    paired_pg_count_changes = 0
    suffix_pg_selected = 0
    suffix_pg_count_changes = 0
    tracklet_changes = 0
    current = associations
    current_marginal = log_marginal_densities
    current_count = log_count_densities
    current_existence = log_existence_priors
    for _ in range(n_sweeps):
        proposed, log_hastings, move_types = _propose_birth_death_moves(
            current, model, config, rng
        )
        proposed_marginal = model.log_marginal_density(proposed)
        proposed_count = model.log_count_density(proposed, include_poisson_void=False)
        proposed_existence = model.log_existence_prior(proposed)
        log_ratio = (
            proposed_existence
            - current_existence
            + proposed_count
            - current_count
            + beta * (proposed_marginal - current_marginal)
            + log_hastings
        )
        accepted = (np.log(rng.random(current.shape[0])) < log_ratio) & (
            (move_types != 4) & (move_types != 5)
        )
        current[accepted] = proposed[accepted]
        current_marginal[accepted] = proposed_marginal[accepted]
        current_count[accepted] = proposed_count[accepted]
        current_existence[accepted] = proposed_existence[accepted]
        accepted_total += int(np.count_nonzero(accepted))
        proposed_total += accepted.size
        for move_type in range(4):
            selected = move_types == move_type
            proposed_by_move[move_type] += int(np.count_nonzero(selected))
            accepted_by_move[move_type] += int(np.count_nonzero(accepted & selected))

        pair_rows = np.flatnonzero(move_types == 4).astype(np.int64)
        if pair_rows.size:
            pg_mask = np.zeros(pair_rows.size, dtype=bool)
            suffix_mask = np.zeros(pair_rows.size, dtype=bool)
            if config.paired_slot_scope == "local":
                (
                    pair_proposed,
                    pair_marginal,
                    pair_count,
                    pair_existence,
                    pair_accepted,
                ) = local_pair_multiple_try_move(
                    current,
                    current_marginal,
                    current_count,
                    current_existence,
                    pair_rows,
                    model,
                    beta=beta,
                    n_candidates=config.paired_slot_candidates,
                    rng=rng,
                )
            elif config.paired_slot_scope == "full_sequence":
                pair_proposed, pair_marginal, pair_accepted = multiple_try_paired_move(
                    current,
                    current_marginal,
                    pair_rows,
                    model,
                    beta=beta,
                    n_candidates=config.paired_slot_candidates,
                    rng=rng,
                )
                pair_count = model.log_count_density(
                    pair_proposed, include_poisson_void=False
                )
                pair_existence = model.log_existence_prior(pair_proposed)
            else:
                fraction = _paired_pg_fraction(config, beta)
                suffix_fraction = (
                    config.paired_slot_suffix_pg_final_fraction if beta == 1.0 else 0.0
                )
                if suffix_fraction:
                    selection = rng.random(pair_rows.size)
                    pg_mask = selection < fraction
                    suffix_mask = (selection >= fraction) & (
                        selection < fraction + suffix_fraction
                    )
                elif fraction == 1:
                    pg_mask[:] = True
                elif fraction > 0:
                    pg_mask = rng.random(pair_rows.size) < fraction
                pair_proposed = current[pair_rows].copy()
                pair_marginal = current_marginal[pair_rows].copy()
                pair_count = current_count[pair_rows].copy()
                pair_existence = current_existence[pair_rows].copy()
                pair_accepted = np.zeros(pair_rows.size, dtype=bool)
                local_positions = np.flatnonzero(~(pg_mask | suffix_mask))
                if local_positions.size:
                    local_rows = pair_rows[local_positions]
                    local_state, local_marginal, local_count, local_existence, local_accept = (
                        local_pair_multiple_try_move(
                            current, current_marginal, current_count, current_existence,
                            local_rows, model, beta=beta,
                            n_candidates=config.paired_slot_candidates, rng=rng,
                        )
                    )
                    pair_proposed[local_positions] = local_state
                    pair_marginal[local_positions] = local_marginal
                    pair_count[local_positions] = local_count
                    pair_existence[local_positions] = local_existence
                    pair_accepted[local_positions] = local_accept
                pg_positions = np.flatnonzero(pg_mask)
                if pg_positions.size:
                    pg_state = conditional_pair_pg_move(
                        current, pair_rows[pg_positions], model,
                        beta=beta, n_particles=config.paired_slot_path_particles,
                        rng=rng, ancestor_sampling=config.paired_slot_pg_ancestor_sampling,
                    )
                    pair_proposed[pg_positions] = pg_state
                    pair_marginal[pg_positions] = model.log_marginal_density(pg_state)
                    pair_count[pg_positions] = model.log_count_density(
                        pg_state, include_poisson_void=False
                    )
                    pair_existence[pg_positions] = model.log_existence_prior(pg_state)
                    # Particle Gibbs always leaves the target invariant; only
                    # its actual state/count changes measure mixing.
                    pair_accepted[pg_positions] = True
                suffix_positions = np.flatnonzero(suffix_mask)
                if suffix_positions.size:
                    suffix_state = conditional_pair_pg_move(
                        current, pair_rows[suffix_positions], model,
                        beta=beta, n_particles=config.paired_slot_path_particles,
                        rng=rng, start_frame=config.paired_slot_suffix_pg_start_frame,
                    )
                    pair_proposed[suffix_positions] = suffix_state
                    pair_marginal[suffix_positions] = model.log_marginal_density(suffix_state)
                    pair_count[suffix_positions] = model.log_count_density(
                        suffix_state, include_poisson_void=False
                    )
                    pair_existence[suffix_positions] = model.log_existence_prior(suffix_state)
                    pair_accepted[suffix_positions] = True
                paired_pg_selected += pg_positions.size + suffix_positions.size
                suffix_pg_selected += suffix_positions.size
            if observer is not None:
                observer.record_pair(
                    current[pair_rows], pair_proposed, pair_accepted, pg_mask | suffix_mask
                )
            selected_rows = pair_rows[pair_accepted]
            selected_states = pair_proposed[pair_accepted]
            paired_changes += int(
                np.count_nonzero(
                    np.any(selected_states != current[selected_rows], axis=(1, 2))
                )
            )
            old_alive = np.count_nonzero(current[selected_rows] != ABSENT, axis=2)
            new_alive = np.count_nonzero(selected_states != ABSENT, axis=2)
            old_assigned = np.count_nonzero(current[selected_rows] >= 0, axis=2)
            new_assigned = np.count_nonzero(selected_states >= 0, axis=2)
            paired_count_changes += int(np.count_nonzero(np.any(
                (old_alive != new_alive) | (old_assigned != new_assigned), axis=1
            )))
            if np.any(pg_mask | suffix_mask):
                selected_pg = pg_mask | suffix_mask
                pg_rows = pair_rows[selected_pg]
                pg_states = pair_proposed[selected_pg]
                pg_old_alive = np.count_nonzero(current[pg_rows] != ABSENT, axis=2)
                pg_new_alive = np.count_nonzero(pg_states != ABSENT, axis=2)
                pg_old_assigned = np.count_nonzero(current[pg_rows] >= 0, axis=2)
                pg_new_assigned = np.count_nonzero(pg_states >= 0, axis=2)
                changed = np.any(
                    (pg_old_alive != pg_new_alive)
                    | (pg_old_assigned != pg_new_assigned), axis=1
                )
                paired_pg_count_changes += int(np.count_nonzero(changed))
                suffix_pg_count_changes += int(np.count_nonzero(changed[suffix_mask[selected_pg]]))
            current[selected_rows] = selected_states
            current_marginal[selected_rows] = pair_marginal[pair_accepted]
            current_count[selected_rows] = pair_count[pair_accepted]
            current_existence[selected_rows] = pair_existence[pair_accepted]
            proposed_by_move[4] += pair_rows.size
            accepted_by_move[4] += int(np.count_nonzero(pair_accepted))
            accepted_total += int(np.count_nonzero(pair_accepted))

        tracklet_rows = np.flatnonzero(move_types == 5).astype(np.int64)
        if tracklet_rows.size:
            move = (
                propose_guided_tracklets
                if config.tracklet_proposal_kind == "displacement"
                else propose_conditional_gp_tracklets
            )
            tracklet_proposed, tracklet_hastings = move(
                current, tracklet_rows, model, rng
            )
            tracklet_marginal = model.log_marginal_density(tracklet_proposed)
            tracklet_count = model.log_count_density(
                tracklet_proposed, include_poisson_void=False
            )
            tracklet_existence = model.log_existence_prior(tracklet_proposed)
            tracklet_ratio = (
                tracklet_existence - current_existence[tracklet_rows]
                + tracklet_count - current_count[tracklet_rows]
                + beta * (tracklet_marginal - current_marginal[tracklet_rows])
                + tracklet_hastings
            )
            tracklet_accepted = np.log(rng.random(tracklet_rows.size)) < tracklet_ratio
            accepted_rows = tracklet_rows[tracklet_accepted]
            tracklet_changes += int(np.count_nonzero(np.any(
                tracklet_proposed[tracklet_accepted] != current[accepted_rows], axis=(1, 2)
            )))
            current[accepted_rows] = tracklet_proposed[tracklet_accepted]
            current_marginal[accepted_rows] = tracklet_marginal[tracklet_accepted]
            current_count[accepted_rows] = tracklet_count[tracklet_accepted]
            current_existence[accepted_rows] = tracklet_existence[tracklet_accepted]
            proposed_by_move[5] += tracklet_rows.size
            accepted_by_move[5] += int(np.count_nonzero(tracklet_accepted))
            accepted_total += int(np.count_nonzero(tracklet_accepted))

    rates = np.divide(
        accepted_by_move,
        proposed_by_move,
        out=np.zeros(6, dtype=float),
        where=proposed_by_move > 0,
    )
    return (
        current,
        current_marginal,
        current_count,
        current_existence,
        (
            accepted_total / proposed_total,
            float(rates[0]),
            float(rates[1]),
            float(rates[2]),
            float(rates[3]),
            float(rates[4]),
            paired_changes / proposed_by_move[4] if proposed_by_move[4] else 0.0,
            float(rates[5]),
            tracklet_changes / proposed_by_move[5] if proposed_by_move[5] else 0.0,
            paired_count_changes / proposed_by_move[4] if proposed_by_move[4] else 0.0,
            paired_pg_selected / proposed_by_move[4] if proposed_by_move[4] else 0.0,
            paired_pg_count_changes / paired_pg_selected if paired_pg_selected else 0.0,
            suffix_pg_selected / proposed_by_move[4] if proposed_by_move[4] else 0.0,
            suffix_pg_count_changes / suffix_pg_selected if suffix_pg_selected else 0.0,
        ),
    )


def sample_slot_existence_posterior(
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
    config: BirthDeathSMCConfig | None = None,
    observer: BirthDeathSMCObserver | None = None,
) -> BirthDeathSMCResult:
    r"""Sample the posterior over explicit birth/death slots.

    The SMC base includes the exact birth--survival--death prior and the
    detection/miss/clutter count factors.  The collapsed Matérn localization
    density is introduced by adaptive tempering.  Rejuvenation combines full
    frame reassignment, birth/death interval replacement, global slot
    permutation, and optional conditional-base blocked slot moves.
    """

    controls = BirthDeathSMCConfig() if config is None else config
    if (controls.paired_slot_suffix_pg_final_fraction
            and controls.paired_slot_suffix_pg_start_frame >= sequence.n_frames):
        raise ValueError("suffix PG start must precede the last frame")
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
    if n_slots < 2 and (
        controls.slot_swap_probability > 0
        or controls.paired_slot_move_probability > 0
        or controls.tracklet_move_probability > 0
    ):
        raise ValueError("slot-swap, paired-slot and tracklet moves require at least two slots")
    rng = np.random.default_rng(controls.seed)
    associations = model.sample_base_associations(
        n_samples=controls.n_particles, rng=rng
    )
    log_marginal = model.log_marginal_density(associations)
    log_count = model.log_count_density(associations, include_poisson_void=False)
    log_existence = model.log_existence_prior(associations)
    beta = 0.0
    log_evidence = model.base_log_normalizer
    stages: list[BirthDeathSMCStageDiagnostics] = []
    target_ess = controls.target_ess_fraction * controls.n_particles

    for step in range(1, controls.max_tempering_steps + 1):
        next_beta = _next_beta(
            log_marginal,
            beta,
            target_ess=target_ess,
            bisection_iterations=controls.bisection_iterations,
        )
        increment = next_beta - beta
        incremental_log_weights = increment * log_marginal
        weights, log_weight_sum = _normalized_weights(incremental_log_weights)
        ess = float(1.0 / np.sum(weights**2))
        log_evidence += log_weight_sum - np.log(controls.n_particles)

        ancestors = systematic_resample(weights, rng)
        if observer is not None:
            observer.begin_stage(step, next_beta)
            observer.record_resampling(associations, weights, ancestors)
        associations = associations[ancestors].copy()
        log_marginal = log_marginal[ancestors].copy()
        log_count = log_count[ancestors].copy()
        log_existence = log_existence[ancestors].copy()
        stage_sweeps = controls.mh_sweeps + (
            controls.final_mh_sweeps if next_beta == 1.0 else 0
        )
        (
            associations,
            log_marginal,
            log_count,
            log_existence,
            acceptance,
        ) = _metropolis_birth_death_rejuvenation(
            associations,
            log_marginal,
            log_count,
            log_existence,
            model,
            beta=next_beta,
            n_sweeps=stage_sweeps,
            config=controls,
            rng=rng,
            observer=observer,
        )
        unique_states = np.unique(
            associations.reshape(controls.n_particles, -1), axis=0
        ).shape[0]
        stages.append(
            BirthDeathSMCStageDiagnostics(
                step=step,
                beta=float(next_beta),
                beta_increment=float(increment),
                effective_sample_size=ess,
                unique_parent_count=int(np.unique(ancestors).size),
                unique_state_count=int(unique_states),
                mh_sweeps=stage_sweeps,
                mh_acceptance_rate=acceptance[0],
                association_acceptance_rate=acceptance[1],
                interval_acceptance_rate=acceptance[2],
                slot_swap_acceptance_rate=acceptance[3],
                blocked_slot_acceptance_rate=acceptance[4],
                paired_slot_acceptance_rate=acceptance[5],
                paired_slot_state_change_rate=acceptance[6],
                tracklet_acceptance_rate=acceptance[7],
                tracklet_state_change_rate=acceptance[8],
                paired_slot_count_signature_change_rate=acceptance[9],
                paired_slot_pg_selected_fraction=acceptance[10],
                paired_slot_pg_count_signature_change_rate=acceptance[11],
                paired_slot_suffix_pg_selected_fraction=acceptance[12],
                paired_slot_suffix_pg_count_signature_change_rate=acceptance[13],
            )
        )
        if observer is not None:
            observer.end_stage(associations)
        beta = next_beta
        if beta == 1.0:
            break
    else:
        raise RuntimeError(
            "SMC did not reach beta=1 within max_tempering_steps; "
            "increase the limit or inspect target degeneracy"
        )

    latent_paths = model.sample_conditional_paths(associations, rng=rng)
    return BirthDeathSMCResult(
        associations=associations,
        existence_masks=associations != ABSENT,
        latent_paths=latent_paths,
        log_marginal_densities=log_marginal,
        log_count_densities=log_count,
        log_existence_priors=log_existence,
        log_evidence_estimate=float(log_evidence),
        stages=tuple(stages),
        config=controls,
    )

