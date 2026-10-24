"""Annealed SMC with explicit null-association and clutter moves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import DetectionSequence
from .null_association import NullAssociationMaternModel
from .smc import (
    AnnealedSMCConfig,
    SMCStageDiagnostics,
    _next_beta,
    _normalized_weights,
    systematic_resample,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NullAssociationSMCResult:
    """Unweighted posterior draws for the missed-detection/clutter model."""

    associations: IntArray
    latent_paths: FloatArray
    log_marginal_densities: FloatArray
    log_count_densities: FloatArray
    log_evidence_estimate: float
    stages: tuple[SMCStageDiagnostics, ...]
    config: AnnealedSMCConfig

    def __post_init__(self) -> None:
        n_samples = self.config.n_particles
        if self.associations.ndim != 3 or self.associations.shape[0] != n_samples:
            raise ValueError("associations must contain one path per SMC particle")
        if self.latent_paths.shape[:3] != self.associations.shape:
            raise ValueError("latent_paths and associations have incompatible shapes")
        if self.log_marginal_densities.shape != (n_samples,):
            raise ValueError("log_marginal_densities has the wrong shape")
        if self.log_count_densities.shape != (n_samples,):
            raise ValueError("log_count_densities has the wrong shape")
        if not self.stages or not np.isclose(self.stages[-1].beta, 1.0):
            raise ValueError("the SMC result must terminate at beta=1")

    @property
    def n_tempering_steps(self) -> int:
        return len(self.stages)


def _state_lookups(
    model: NullAssociationMaternModel,
) -> tuple[tuple[IntArray, ...], tuple[dict[bytes, int], ...], IntArray]:
    states = tuple(model.frame_states(time_index) for time_index in range(model.sequence.n_frames))
    lookups = tuple(
        {state.tobytes(): index for index, state in enumerate(frame_states)}
        for frame_states in states
    )
    mutable = np.asarray(
        [
            time_index
            for time_index, frame_states in enumerate(states)
            if time_index > 0 and frame_states.shape[0] > 1
        ],
        dtype=np.int64,
    )
    return states, lookups, mutable


def _propose_null_moves(
    associations: IntArray,
    *,
    frame_states: tuple[IntArray, ...],
    state_lookups: tuple[dict[bytes, int], ...],
    mutable_frames: IntArray,
    frame_move_probability: float,
    rng: np.random.Generator,
) -> tuple[IntArray, NDArray[np.bool_]]:
    """Propose reversible frame replacements or contiguous label swaps."""

    n_samples, n_frames, n_particles = associations.shape
    proposed = associations.copy()
    frame_moves = rng.random(n_samples) < frame_move_probability
    if n_particles == 1:
        frame_moves[:] = True
    if mutable_frames.size == 0:
        frame_moves[:] = False

    frame_rows = np.flatnonzero(frame_moves)
    if frame_rows.size:
        selected_frames = mutable_frames[rng.integers(mutable_frames.size, size=frame_rows.size)]
        for time_index in np.unique(selected_frames):
            rows = frame_rows[selected_frames == time_index]
            states = frame_states[int(time_index)]
            lookup = state_lookups[int(time_index)]
            current_indices = np.asarray(
                [lookup[associations[row, time_index].tobytes()] for row in rows],
                dtype=np.int64,
            )
            offsets = rng.integers(1, states.shape[0], size=rows.size)
            proposed_indices = (current_indices + offsets) % states.shape[0]
            proposed[rows, time_index] = states[proposed_indices]

    segment_rows = np.flatnonzero(~frame_moves)
    if segment_rows.size and n_particles > 1:
        starts = rng.integers(1, n_frames, size=segment_rows.size)
        available_lengths = n_frames - starts
        ends = starts + np.floor(rng.random(segment_rows.size) * available_lengths).astype(np.int64)
        first_labels = rng.integers(n_particles, size=segment_rows.size)
        offsets = rng.integers(1, n_particles, size=segment_rows.size)
        second_labels = (first_labels + offsets) % n_particles
        for local_index, row in enumerate(segment_rows):
            interval = slice(starts[local_index], ends[local_index] + 1)
            first = first_labels[local_index]
            second = second_labels[local_index]
            temporary = proposed[row, interval, first].copy()
            proposed[row, interval, first] = proposed[row, interval, second]
            proposed[row, interval, second] = temporary
    return proposed, frame_moves


def _metropolis_null_rejuvenation(
    associations: IntArray,
    log_marginal_densities: FloatArray,
    log_count_densities: FloatArray,
    model: NullAssociationMaternModel,
    *,
    beta: float,
    n_sweeps: int,
    frame_move_probability: float,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, FloatArray, tuple[float, float, float]]:
    if n_sweeps == 0:
        return (
            associations,
            log_marginal_densities,
            log_count_densities,
            (0.0, 0.0, 0.0),
        )

    frame_states, state_lookups, mutable_frames = _state_lookups(model)
    if mutable_frames.size == 0 and model.n_particles == 1:
        return (
            associations,
            log_marginal_densities,
            log_count_densities,
            (0.0, 0.0, 0.0),
        )

    accepted_total = 0
    proposed_total = 0
    accepted_by_move = np.zeros(2, dtype=np.int64)
    proposed_by_move = np.zeros(2, dtype=np.int64)
    current = associations
    current_marginal = log_marginal_densities
    current_count = log_count_densities
    for _ in range(n_sweeps):
        proposed, frame_moves = _propose_null_moves(
            current,
            frame_states=frame_states,
            state_lookups=state_lookups,
            mutable_frames=mutable_frames,
            frame_move_probability=frame_move_probability,
            rng=rng,
        )
        proposed_marginal = model.log_marginal_density(proposed)
        proposed_count = model.log_count_density(proposed, include_poisson_void=False)
        log_ratio = proposed_count - current_count + beta * (proposed_marginal - current_marginal)
        accepted = np.log(rng.random(current.shape[0])) < log_ratio
        current[accepted] = proposed[accepted]
        current_marginal[accepted] = proposed_marginal[accepted]
        current_count[accepted] = proposed_count[accepted]
        accepted_total += int(np.count_nonzero(accepted))
        proposed_total += accepted.size
        for move_index, selected in enumerate((frame_moves, ~frame_moves)):
            proposed_by_move[move_index] += int(np.count_nonzero(selected))
            accepted_by_move[move_index] += int(np.count_nonzero(accepted & selected))

    rates = np.divide(
        accepted_by_move,
        proposed_by_move,
        out=np.zeros(2, dtype=float),
        where=proposed_by_move > 0,
    )
    return (
        current,
        current_marginal,
        current_count,
        (
            accepted_total / proposed_total,
            float(rates[0]),
            float(rates[1]),
        ),
    )


def sample_null_association_posterior(
    sequence: DetectionSequence,
    *,
    n_particles: int,
    localization_std: float,
    coordinate_std: ArrayLike,
    length_scale: float,
    detection_probability: float,
    expected_clutter_per_frame: float,
    roi_lower: ArrayLike,
    roi_upper: ArrayLike,
    nu: float = 1.5,
    relative_jitter: float = 1e-10,
    anchor_first_frame: bool = True,
    config: AnnealedSMCConfig | None = None,
) -> NullAssociationSMCResult:
    r"""Sample the fixed-latent-count posterior with explicit null moves.

    The SMC base density is the normalized product of the detection,
    missed-detection, and clutter-intensity factors. The Matérn-marginalized
    localization density is introduced through adaptive tempering:

    .. math::

       \pi_\beta(a)\propto q_0(a)\,p(y_a\mid a)^\beta.

    Single-frame proposals replace a complete partial injection and therefore
    include assignment-to-null, null-to-assignment, reassignment, and clutter
    reclamation moves. Segment label swaps preserve coherent crossing modes.
    """

    controls = AnnealedSMCConfig() if config is None else config
    model = NullAssociationMaternModel(
        sequence,
        n_particles=n_particles,
        localization_std=localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=detection_probability,
        expected_clutter_per_frame=expected_clutter_per_frame,
        roi_lower=roi_lower,
        roi_upper=roi_upper,
        nu=nu,
        relative_jitter=relative_jitter,
        anchor_first_frame=anchor_first_frame,
    )
    rng = np.random.default_rng(controls.seed)
    associations = model.sample_base_associations(n_samples=controls.n_particles, rng=rng)
    log_marginal = model.log_marginal_density(associations)
    log_count = model.log_count_density(associations, include_poisson_void=False)
    beta = 0.0
    log_evidence = model.base_log_normalizer
    stages: list[SMCStageDiagnostics] = []
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
        associations = associations[ancestors].copy()
        log_marginal = log_marginal[ancestors].copy()
        log_count = log_count[ancestors].copy()
        stage_mh_sweeps = controls.mh_sweeps + (controls.final_mh_sweeps if next_beta == 1.0 else 0)
        associations, log_marginal, log_count, acceptance = _metropolis_null_rejuvenation(
            associations,
            log_marginal,
            log_count,
            model,
            beta=next_beta,
            n_sweeps=stage_mh_sweeps,
            frame_move_probability=controls.frame_move_probability,
            rng=rng,
        )
        unique_states = np.unique(associations.reshape(controls.n_particles, -1), axis=0).shape[0]
        stages.append(
            SMCStageDiagnostics(
                step=step,
                beta=float(next_beta),
                beta_increment=float(increment),
                effective_sample_size=ess,
                unique_parent_count=int(np.unique(ancestors).size),
                unique_state_count=int(unique_states),
                mh_sweeps=stage_mh_sweeps,
                mh_acceptance_rate=acceptance[0],
                frame_acceptance_rate=acceptance[1],
                segment_acceptance_rate=acceptance[2],
            )
        )
        beta = next_beta
        if beta == 1.0:
            break
    else:
        raise RuntimeError(
            "SMC did not reach beta=1 within max_tempering_steps; "
            "increase the limit or inspect target degeneracy"
        )

    latent_paths = model.sample_conditional_paths(associations, rng=rng)
    return NullAssociationSMCResult(
        associations=associations,
        latent_paths=latent_paths,
        log_marginal_densities=log_marginal,
        log_count_densities=log_count,
        log_evidence_estimate=float(log_evidence),
        stages=tuple(stages),
        config=controls,
    )
