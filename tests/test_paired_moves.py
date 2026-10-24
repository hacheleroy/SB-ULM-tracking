import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.data import DetectionSequence
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    _metropolis_birth_death_rejuvenation,
)
from sbulm.paired_moves import (
    PairConditionalBase,
    _local_pair_candidate_set,
    multiple_try_paired_move,
)
from sbulm.slot_existence import (
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
)


def _small_model(n_slots: int = 2) -> SlotExistenceMaternModel:
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
        n_slots=n_slots,
        localization_std=0.005,
        coordinate_std=(0.02, 0.01),
        length_scale=0.003,
        detection_probability=0.7,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )


def test_pair_conditional_base_matches_complete_oracle() -> None:
    model = _small_model()
    support, _ = enumerate_slot_existence_support(model)
    conditional = PairConditionalBase(model, support[0], (0, 1))
    np.testing.assert_allclose(
        conditional.log_probability(support),
        model.log_base_probability(support),
        rtol=0,
        atol=1e-12,
    )
    assert logsumexp(conditional.log_probability(support)) == pytest.approx(0, abs=1e-12)

    draws = conditional.sample(16_000, np.random.default_rng(13))
    model._validate_associations(draws)
    expected_alive = np.sum(
        np.exp(conditional.log_probability(support))
        * np.count_nonzero(support != -2, axis=(1, 2))
    )
    assert np.mean(np.count_nonzero(draws != -2, axis=(1, 2))) == pytest.approx(
        expected_alive, abs=0.04
    )


def test_pair_conditional_base_keeps_other_slots_fixed() -> None:
    model = _small_model(n_slots=3)
    support, _ = enumerate_slot_existence_support(model)
    current = support[len(support) // 2]
    selected = support[np.all(support[:, :, 2] == current[:, 2], axis=1)]
    conditional = PairConditionalBase(model, current, (0, 1))
    expected = model.log_base_probability(selected)
    expected -= logsumexp(expected)
    np.testing.assert_allclose(conditional.log_probability(selected), expected, atol=1e-12)
    drawn = conditional.sample(64, np.random.default_rng(7))
    np.testing.assert_array_equal(
        drawn[:, :, 2], np.broadcast_to(current[None, :, 2], (64, 3))
    )


def test_multiple_try_paired_move_accepts_at_beta_zero() -> None:
    model = _small_model(n_slots=3)
    rng = np.random.default_rng(23)
    current = model.sample_base_associations(n_samples=128, rng=rng)
    rows = np.arange(current.shape[0], dtype=np.int64)
    marginal = model.log_marginal_density(current)
    proposed, proposed_marginal, accepted = multiple_try_paired_move(
        current, marginal, rows, model, beta=0.0, n_candidates=2, rng=rng
    )
    assert np.all(accepted)
    assert np.all(np.isfinite(model.log_joint_density(proposed)))
    np.testing.assert_allclose(proposed_marginal, model.log_marginal_density(proposed))


def test_local_pair_window_has_identical_forward_and_reverse_support() -> None:
    model = _small_model()
    current = model.sample_base_associations(
        n_samples=1, rng=np.random.default_rng(6)
    )[0]
    options = _local_pair_candidate_set(model, current, (0, 1), 1, 2)
    assert options.shape[0] > 1
    assert any(np.array_equal(option, current) for option in options)
    assert np.unique(options.reshape(options.shape[0], -1), axis=0).shape[0] == len(options)
    for alternate in options[:: max(1, len(options) // 10)]:
        reverse = _local_pair_candidate_set(model, alternate, (0, 1), 1, 2)
        assert {state.tobytes() for state in reverse} == {
            state.tobytes() for state in options
        }
    assert np.all(np.isfinite(model.log_joint_density(options)))


def test_paired_only_rejuvenation_is_calibrated_on_enumerable_case() -> None:
    model = _small_model()
    support, _ = enumerate_slot_existence_support(model)
    log_exact = model.log_joint_density(support)
    exact = np.exp(log_exact - logsumexp(log_exact))
    expected_alive = float(
        np.sum(exact * np.count_nonzero(support != -2, axis=(1, 2)))
    )
    rng = np.random.default_rng(31)
    current = support[rng.choice(support.shape[0], size=1_024, p=exact)].copy()
    controls = BirthDeathSMCConfig(
        n_particles=current.shape[0],
        mh_sweeps=4,
        final_mh_sweeps=0,
        association_move_probability=0.0,
        interval_move_probability=0.0,
        slot_swap_probability=0.0,
        blocked_slot_move_probability=0.0,
        paired_slot_move_probability=1.0,
        paired_slot_candidates=2,
    )
    updated = _metropolis_birth_death_rejuvenation(
        current,
        model.log_marginal_density(current),
        model.log_count_density(current, include_poisson_void=False),
        model.log_existence_prior(current),
        model,
        beta=1.0,
        n_sweeps=4,
        config=controls,
        rng=rng,
    )
    assert updated[-1][5] > 0.05
    actual_alive = float(np.mean(np.count_nonzero(updated[0] != -2, axis=(1, 2))))
    assert actual_alive == pytest.approx(expected_alive, abs=0.15)
