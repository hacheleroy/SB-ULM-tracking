"""Finite-oracle tests of the likelihood-guided full-history PG kernel."""

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.conditional_pair_pg import ConditionalPairPath
from sbulm.data import DetectionSequence
from sbulm.existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from sbulm.paired_moves import PairConditionalBase
from sbulm.slot_existence import SlotExistenceMaternModel, enumerate_slot_existence_support
from sbulm.wp11c_full_history_pilot import _pilot_model


def _model(n_frames: int = 2, n_slots: int = 2) -> SlotExistenceMaternModel:
    frames = [np.asarray([[-0.008, 0.002]])]
    frames += [np.empty((0, 2)) for _ in range(n_frames - 2)]
    frames += [np.asarray([[0.008, -0.002]])]
    return SlotExistenceMaternModel(
        DetectionSequence(
            times=np.arange(n_frames, dtype=float) * 0.001, detections=tuple(frames)
        ),
        n_slots=n_slots,
        localization_std=0.005,
        coordinate_std=(0.02, 0.01),
        length_scale=0.003,
        detection_probability=0.7,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_incremental_gp_and_count_weights_equal_exact_target(beta: float) -> None:
    model = _model(n_frames=3)
    support, _ = enumerate_slot_existence_support(model)
    for state in support[::7]:
        kernel = ConditionalPairPath(model, state, (0, 1))
        prefix = state.copy()
        previous = None
        total = 0.0
        for time_index in range(model.sequence.n_frames):
            states, options, weights = kernel.options(prefix, previous, time_index, beta)
            matches = np.flatnonzero(np.all(options == state[time_index], axis=1))
            assert len(matches) == 1
            index = int(matches[0])
            total += weights[index]
            previous = int(states[index])
        expected = (
            model.log_existence_prior(state[None])[0]
            + model.log_count_density(state[None], include_poisson_void=False)[0]
            + beta * model.log_marginal_density(state[None])[0]
        )
        assert total == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_two_particle_two_frame_pg_satisfies_exact_detailed_balance(beta: float) -> None:
    """Sum every proposal, ancestor and final-selection event (not MC)."""

    model = _model()
    support, _ = enumerate_slot_existence_support(model)
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    conditional = PairConditionalBase(model, support[0], (0, 1))
    log_target = conditional.log_probability(support) + beta * model.log_marginal_density(support)
    target = np.exp(log_target - logsumexp(log_target))
    transition = np.zeros((len(support), len(support)))
    for ref_index, reference in enumerate(support):
        kernel = ConditionalPairPath(model, reference, (0, 1))
        initial_states, initial_options, first_log = kernel.options(
            reference, None, 0, beta
        )
        first_prob = np.exp(first_log - logsumexp(first_log))
        ref_first = np.flatnonzero(np.all(initial_options == reference[0], axis=1))
        assert len(ref_first) == 1
        ref_state_index = int(initial_states[ref_first[0]])
        ref_states, _, ref_log = kernel.options(reference, ref_state_index, 1, beta)
        assert ref_states.size
        ref_normalizer = float(logsumexp(ref_log))
        for first_index, (state_index, first_frame, probability) in enumerate(
            zip(initial_states, initial_options, first_prob, strict=True)
        ):
            first_path = reference.copy()
            first_path[0] = first_frame
            # At the second frame, a non-reference particle resamples its
            # ancestor uniformly because the first-stage adapted weights match.
            for ancestor_path, ancestor_state in (
                (first_path, int(state_index)), (reference, ref_state_index)
            ):
                _, second_options, second_log = kernel.options(
                    ancestor_path, ancestor_state, 1, beta
                )
                second_normalizer = float(logsumexp(second_log))
                select_nonreference = np.exp(second_normalizer - np.logaddexp(
                    second_normalizer, ref_normalizer
                ))
                for next_frame, conditional_probability in zip(
                    second_options, np.exp(second_log - second_normalizer), strict=True
                ):
                    output = ancestor_path.copy()
                    output[1] = next_frame
                    transition[ref_index, lookup[output.tobytes()]] += (
                        probability * 0.5 * conditional_probability * select_nonreference
                    )
        transition[ref_index, ref_index] += 1 - np.sum(transition[ref_index])
    assert np.allclose(np.sum(transition, axis=1), 1.0, atol=1e-12)
    assert np.allclose(target[:, None] * transition, target[None, :] * transition.T,
                       rtol=1e-10, atol=1e-13)


def test_pg_rejuvenation_keeps_valid_reference_and_changes_modes() -> None:
    model = _model(n_frames=3, n_slots=3)
    rng = np.random.default_rng(92)
    states = model.sample_base_associations(n_samples=32, rng=rng)
    controls = BirthDeathSMCConfig(
        n_particles=len(states), mh_sweeps=1, final_mh_sweeps=0,
        association_move_probability=0, interval_move_probability=0,
        slot_swap_probability=0, blocked_slot_move_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_path_particles=4,
    )
    result = _metropolis_birth_death_rejuvenation(
        states.copy(), model.log_marginal_density(states),
        model.log_count_density(states, include_poisson_void=False),
        model.log_existence_prior(states), model, beta=1.0,
        n_sweeps=1, config=controls, rng=rng,
    )
    model._validate_associations(result[0])
    assert result[-1][5] == 1.0
    assert result[-1][6] > 0.0


def test_pg_configuration_is_opt_in() -> None:
    assert BirthDeathSMCConfig().paired_slot_scope == "local"
    with pytest.raises(ValueError, match="paired_slot_path_particles"):
        BirthDeathSMCConfig(paired_slot_path_particles=1)
    with pytest.raises(ValueError, match="paired_slot_pg_fraction"):
        BirthDeathSMCConfig(paired_slot_scope="conditional_smc", paired_slot_pg_fraction=1.1)
    with pytest.raises(ValueError, match="requires conditional_smc"):
        BirthDeathSMCConfig(paired_slot_pg_fraction=0.1)


def test_zero_pg_fraction_is_identical_to_default_local_move() -> None:
    model = _pilot_model()
    states = model.sample_base_associations(n_samples=64, rng=np.random.default_rng(18))
    common = {
        "n_particles": len(states), "mh_sweeps": 2, "final_mh_sweeps": 0,
        "association_move_probability": 0, "interval_move_probability": 0,
        "slot_swap_probability": 0, "blocked_slot_move_probability": 0,
        "paired_slot_move_probability": 1,
    }
    local = _metropolis_birth_death_rejuvenation(
        states.copy(), model.log_marginal_density(states),
        model.log_count_density(states, include_poisson_void=False),
        model.log_existence_prior(states), model, beta=1, n_sweeps=2,
        config=BirthDeathSMCConfig(**common), rng=np.random.default_rng(43),
    )
    mixed = _metropolis_birth_death_rejuvenation(
        states.copy(), model.log_marginal_density(states),
        model.log_count_density(states, include_poisson_void=False),
        model.log_existence_prior(states), model, beta=1, n_sweeps=2,
        config=BirthDeathSMCConfig(
            **common, paired_slot_scope="conditional_smc", paired_slot_pg_fraction=0
        ), rng=np.random.default_rng(43),
    )
    for expected, observed in zip(local[:4], mixed[:4], strict=True):
        assert np.array_equal(expected, observed)
    assert local[-1] == mixed[-1]


def test_sparse_pg_mixture_from_exact_posterior_retains_count_modes() -> None:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    log_target = model.log_joint_density(support)
    exact = np.exp(log_target - logsumexp(log_target))
    rng = np.random.default_rng(77)
    current = support[rng.choice(len(support), size=512, p=exact)]
    controls = BirthDeathSMCConfig(
        n_particles=len(current), mh_sweeps=2, final_mh_sweeps=0,
        association_move_probability=0, interval_move_probability=0,
        slot_swap_probability=0, blocked_slot_move_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=0.1, paired_slot_path_particles=8,
    )
    result = _metropolis_birth_death_rejuvenation(
        current.copy(), model.log_marginal_density(current),
        model.log_count_density(current, include_poisson_void=False),
        model.log_existence_prior(current), model, beta=1, n_sweeps=2,
        config=controls, rng=np.random.default_rng(177),
    )
    assert 0.06 < result[-1][10] < 0.14
    assert result[-1][11] > 0.2
    mode_labels, inverse = np.unique(
        np.concatenate((
            np.count_nonzero(support != -2, axis=2),
            np.count_nonzero(support >= 0, axis=2),
        ), axis=1), axis=0, return_inverse=True,
    )
    assert len(mode_labels) > 1
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    indices = np.asarray([lookup[state.tobytes()] for state in result[0]])
    exact_modes = np.bincount(inverse, weights=exact)
    sampled_modes = np.bincount(inverse[indices], minlength=len(exact_modes)) / len(indices)
    assert 0.5 * np.sum(np.abs(exact_modes - sampled_modes)) < 0.12


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_two_particle_pgas_exact_transition_is_stationary(beta: float) -> None:
    """Enumerate both ancestors, all proposals, MH decisions and final draws."""

    model = _model()
    support, _ = enumerate_slot_existence_support(model)
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    log_target = (model.log_existence_prior(support)
                  + model.log_count_density(support, include_poisson_void=False)
                  + beta * model.log_marginal_density(support))
    target = np.exp(log_target - logsumexp(log_target))
    transition = np.zeros((len(support), len(support)))
    for ref_index, reference in enumerate(support):
        kernel = ConditionalPairPath(model, reference, (0, 1))
        first_states, first_options, first_log = kernel.options(reference, None, 0, beta)
        first_prob = np.exp(first_log - logsumexp(first_log))
        ref_first = np.flatnonzero(np.all(first_options == reference[0], axis=1))
        ref_state = int(first_states[ref_first[0]])
        for first_state, first_frame, probability in zip(
            first_states, first_options, first_prob, strict=True
        ):
            candidate = reference.copy()
            candidate[0] = first_frame
            old_score = kernel._reference_suffix_score(reference, reference, 1, beta)
            new_score = kernel._reference_suffix_score(candidate, reference, 1, beta)
            accept = float(min(1.0, np.exp(new_score - old_score)))
            # The fully adapted first-frame normalizer gives both particles
            # the same importance weight; a forced proposal is deterministic.
            for fixed_prefix, fixed_state, fixed_probability in (
                (reference, ref_state, 1 - accept),
                (candidate, int(first_state), accept),
            ):
                if fixed_probability == 0:
                    continue
                _, _, fixed_log = kernel.options(fixed_prefix, fixed_state, 1, beta)
                fixed_normalizer = float(logsumexp(fixed_log))
                fixed_output = fixed_prefix.copy()
                fixed_output[1] = reference[1]
                for nonref_prefix, nonref_state in (
                    (candidate, int(first_state)), (reference, ref_state)
                ):
                    _, options, log_scores = kernel.options(
                        nonref_prefix, nonref_state, 1, beta
                    )
                    nonref_normalizer = float(logsumexp(log_scores))
                    nonref_probability = np.exp(
                        nonref_normalizer
                        - np.logaddexp(nonref_normalizer, fixed_normalizer)
                    )
                    base = probability * fixed_probability * 0.5
                    transition[ref_index, lookup[fixed_output.tobytes()]] += (
                        base * (1 - nonref_probability)
                    )
                    for frame, frame_probability in zip(
                        options, np.exp(log_scores - nonref_normalizer), strict=True
                    ):
                        output = nonref_prefix.copy()
                        output[1] = frame
                        transition[ref_index, lookup[output.tobytes()]] += (
                            base * nonref_probability * frame_probability
                        )
    assert np.allclose(transition.sum(axis=1), 1, atol=1e-12)
    assert np.allclose(target @ transition, target, rtol=1e-11, atol=1e-12)
