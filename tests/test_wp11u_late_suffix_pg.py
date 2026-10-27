"""Exact finite-target validation of conditional suffix particle Gibbs."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.conditional_pair_pg import ConditionalPairPath
from sbulm.data import DetectionSequence
from sbulm.existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from sbulm.slot_existence import SlotExistenceMaternModel, enumerate_slot_existence_support
from sbulm.wp11u_late_suffix_pg import (
    MAX_MEDIAN_TV,
    SEEDS,
    _controls,
    _errors,
    _protocol,
    _reference,
)

ROOT = Path(__file__).resolve().parents[1]


def _model():
    return SlotExistenceMaternModel(
        DetectionSequence(
            times=np.arange(3) * 0.001,
            detections=(np.array([[-0.008, 0.002]]), np.empty((0, 2)),
                        np.array([[0.008, -0.002]])),
        ),
        n_slots=2, localization_std=0.005, coordinate_std=(0.02, 0.01),
        length_scale=0.003, detection_probability=0.7,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04), roi_upper=(0.04, 0.04),
    )


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_two_particle_suffix_transition_is_exactly_stationary(beta):
    """Enumerate every suffix proposal, ancestor choice and final selection."""

    model = _model()
    support, _ = enumerate_slot_existence_support(model)
    lookup = {state.tobytes(): i for i, state in enumerate(support)}
    score = (model.log_existence_prior(support)
             + model.log_count_density(support, include_poisson_void=False)
             + beta * model.log_marginal_density(support))
    target = np.exp(score - logsumexp(score))
    transition = np.zeros((len(support), len(support)))
    for ref_index, reference in enumerate(support):
        kernel = ConditionalPairPath(model, reference, (0, 1))
        previous = kernel._previous_state(reference, 1)
        _, options, first_log = kernel.options(reference, previous, 1, beta)
        first_prob = np.exp(first_log - logsumexp(first_log))
        _, _, ref_last = kernel.options(reference, kernel._previous_state(reference, 2), 2,
                                        beta)
        ref_normalizer = float(logsumexp(ref_last))
        for frame, probability in zip(options, first_prob, strict=True):
            candidate = reference.copy()
            candidate[1] = frame
            next_state = kernel._previous_state(candidate, 2)
            for ancestor in (candidate, reference):
                _, last_options, last_log = kernel.options(ancestor, next_state if
                    ancestor is candidate else kernel._previous_state(reference, 2), 2,
                    beta)
                normalizer = float(logsumexp(last_log))
                selected = np.exp(normalizer - np.logaddexp(normalizer, ref_normalizer))
                for final_frame, final_probability in zip(
                    last_options, np.exp(last_log - normalizer), strict=True
                ):
                    output = ancestor.copy()
                    output[2] = final_frame
                    transition[ref_index, lookup[output.tobytes()]] += (
                        probability * 0.5 * selected * final_probability
                    )
        transition[ref_index, ref_index] += 1 - transition[ref_index].sum()
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(target @ transition, target, atol=1e-12, rtol=1e-11)


def test_suffix_rejuvenation_preserves_prefix_and_reports_count_movement():
    model = _model()
    states = model.sample_base_associations(n_samples=48, rng=np.random.default_rng(19))
    controls = BirthDeathSMCConfig(
        n_particles=len(states), association_move_probability=0,
        interval_move_probability=0, slot_swap_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=0, paired_slot_suffix_pg_final_fraction=1,
        paired_slot_suffix_pg_start_frame=1, paired_slot_path_particles=4,
    )
    result = _metropolis_birth_death_rejuvenation(
        states.copy(), model.log_marginal_density(states),
        model.log_count_density(states, include_poisson_void=False),
        model.log_existence_prior(states), model, beta=1.0,
        n_sweeps=2, config=controls, rng=np.random.default_rng(49),
    )
    model._validate_associations(result[0])
    np.testing.assert_array_equal(result[0][:, 0], states[:, 0])
    assert result[-1][12] == 1
    assert result[-1][13] > 0


def test_suffix_option_rejects_invalid_fraction_and_frame():
    with pytest.raises(ValueError, match="sum to at most one"):
        BirthDeathSMCConfig(paired_slot_scope="conditional_smc", paired_slot_pg_fraction=.9,
                            paired_slot_suffix_pg_final_fraction=.2)
    with pytest.raises(ValueError, match="condition on at least one"):
        BirthDeathSMCConfig(paired_slot_suffix_pg_start_frame=0)


def test_frozen_primary_reference_and_opt_in_controls():
    case, exact = _reference(ROOT / "results/wp11t_subset_exact_primary.json")
    assert case.model.n_slots == 4
    assert len(exact["joint_alive_by_sixth_by_last_clutter"]) == 33
    assert _errors(np.asarray(exact["joint_alive_by_sixth_by_last_clutter"]), exact)[
        "full_3d_joint_tv"
    ] == 0
    assert _controls(SEEDS[0], "control").paired_slot_suffix_pg_final_fraction == 0
    candidate = _controls(SEEDS[0], "candidate")
    assert candidate.paired_slot_suffix_pg_start_frame == 5
    assert candidate.paired_slot_suffix_pg_final_fraction == .4
    assert candidate.final_mh_sweeps == 24
    assert _protocol()["selection_gate"]["maximum_candidate_median_full_joint_tv"] == (
        MAX_MEDIAN_TV
    )
