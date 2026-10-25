"""Finite-state and SMC checks for the opt-in full-history pair move."""

import numpy as np
import pytest

from sbulm.data import DetectionSequence
from sbulm.existence_smc import BirthDeathSMCConfig, _metropolis_birth_death_rejuvenation
from sbulm.paired_moves import PairConditionalBase
from sbulm.slot_existence import (
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
)


def _small_model() -> SlotExistenceMaternModel:
    sequence = DetectionSequence(
        times=np.arange(3, dtype=float) * 0.001,
        detections=(
            np.asarray([[-0.008, 0.0]]),
            np.empty((0, 2)),
            np.asarray([[0.008, 0.0], [-0.007, 0.002]]),
        ),
    )
    return SlotExistenceMaternModel(
        sequence,
        n_slots=2,
        localization_std=0.005,
        coordinate_std=(0.02, 0.01),
        length_scale=0.003,
        detection_probability=0.7,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )


def _off_diagonal_transition(
    old: int, new: int, base: np.ndarray, weights: np.ndarray
) -> float:
    """Enumerate the two-try forward and one-draw reverse MTM law."""

    chosen_weight = weights[new]
    forward_sum = chosen_weight + weights[:, None]
    reverse_sum = weights[old] + weights[None, :]
    acceptance = np.minimum(1.0, forward_sum / reverse_sum)
    selected = chosen_weight / forward_sum
    return float(2.0 * base[new] * np.sum(
        base[:, None] * base[None, :] * selected * acceptance
    ))


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_full_history_pair_satisfies_exact_detailed_balance(beta: float) -> None:
    model = _small_model()
    support, _ = enumerate_slot_existence_support(model)
    conditional = PairConditionalBase(model, support[0], (0, 1))
    base = np.exp(conditional.log_probability(support))
    log_likelihood = model.log_marginal_density(support)
    weights = np.exp(beta * log_likelihood - np.max(beta * log_likelihood))
    target = base * weights
    target /= np.sum(target)
    for old, new in ((0, 1), (0, len(support) // 2), (10, len(support) - 1)):
        forward = _off_diagonal_transition(old, new, base, weights)
        reverse = _off_diagonal_transition(new, old, base, weights)
        assert target[old] * forward == pytest.approx(
            target[new] * reverse, rel=2e-12, abs=1e-15
        )


def test_full_history_pair_rejuvenation_changes_count_signatures_at_base() -> None:
    model = _small_model()
    rng = np.random.default_rng(74)
    current = model.sample_base_associations(n_samples=512, rng=rng)
    controls = BirthDeathSMCConfig(
        n_particles=len(current),
        mh_sweeps=2,
        final_mh_sweeps=0,
        association_move_probability=0.0,
        interval_move_probability=0.0,
        slot_swap_probability=0.0,
        blocked_slot_move_probability=0.0,
        paired_slot_move_probability=1.0,
        paired_slot_scope="full_sequence",
    )
    result = _metropolis_birth_death_rejuvenation(
        current,
        model.log_marginal_density(current),
        model.log_count_density(current, include_poisson_void=False),
        model.log_existence_prior(current),
        model,
        beta=0.0,
        n_sweeps=2,
        config=controls,
        rng=rng,
    )
    model._validate_associations(result[0])
    assert result[-1][5] == pytest.approx(1.0)
    assert result[-1][9] > 0.1


def test_full_history_pair_configuration_is_opt_in() -> None:
    assert BirthDeathSMCConfig().paired_slot_scope == "local"
    with pytest.raises(ValueError, match="paired_slot_scope"):
        BirthDeathSMCConfig(paired_slot_scope="unknown")
