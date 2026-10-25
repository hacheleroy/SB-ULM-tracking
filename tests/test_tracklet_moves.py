"""Finite-state detailed balance and exact-posterior checks for WP11."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.data import DetectionSequence
from sbulm.existence_smc import BirthDeathSMCConfig, sample_slot_existence_posterior
from sbulm.slot_existence import (
    SlotExistenceMaternModel,
    enumerate_slot_existence_posterior,
    slot_event_marginals,
    slot_existence_support_size,
)
from sbulm.tracklet_moves import GuidedPairWindow, propose_guided_tracklets
from sbulm.wp7_convergence_audit import _make_case


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


def _conditional_support(window: GuidedPairWindow) -> np.ndarray:
    states = []
    for frames in product(*window.options):
        candidate = window.current.copy()
        for offset, frame in enumerate(frames):
            candidate[window.start + offset, window.slots[0]] = frame[0]
            candidate[window.start + offset, window.slots[1]] = frame[1]
        states.append(candidate)
    return np.stack(states)


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_exact_conditional_kernel_obeys_detailed_balance(beta: float) -> None:
    model = _small_model()
    current = np.asarray([[0, -1], [-1, -1], [0, -1]], dtype=np.int64)
    window = GuidedPairWindow(model, current, (0, 1), 0)
    support = _conditional_support(window)
    model._validate_associations(support)
    log_q = np.asarray([window.log_probability(state) for state in support])
    assert logsumexp(log_q) == pytest.approx(0.0, abs=1e-12)
    assert np.any(support < 0) and np.any(support >= 0)

    # Every pair of distinct states in this exact finite conditional oracle
    # must carry identical stationary flow in both directions.
    log_target = (
        model.log_existence_prior(support)
        + model.log_count_density(support, include_poisson_void=False)
        + beta * model.log_marginal_density(support)
    )
    log_pi = log_target - logsumexp(log_target)
    log_acceptance = np.minimum(
        0.0,
        log_target[None, :] - log_target[:, None] + log_q[:, None] - log_q[None, :],
    )
    log_flow = log_pi[:, None] + log_q[None, :] + log_acceptance
    np.testing.assert_allclose(log_flow, log_flow.T, rtol=0, atol=1e-12)

    alternate = GuidedPairWindow(model, support[-1], (0, 1), 0)
    np.testing.assert_allclose(
        [alternate.log_probability(state) for state in support], log_q, atol=1e-12
    )


def test_four_slot_bridge_preserves_masks_and_reverses_tracklets() -> None:
    case = _make_case(bridge=True)
    model = case.model
    assert slot_existence_support_size(model) == 282_469
    current = model.sample_base_associations(n_samples=1, rng=np.random.default_rng(19))[0]
    window = GuidedPairWindow(model, current, (0, 2), 1)
    draws = np.asarray([window.sample(np.random.default_rng(seed)) for seed in range(30)])
    model._validate_associations(draws)
    np.testing.assert_array_equal(draws != -2, np.broadcast_to(current != -2, draws.shape))
    np.testing.assert_array_equal(
        draws[:, :, [1, 3]], np.broadcast_to(current[:, [1, 3]], draws[:, :, [1, 3]].shape)
    )
    for draw in draws:
        reverse = GuidedPairWindow(model, draw, (0, 2), 1)
        assert reverse.log_probability(current) == pytest.approx(
            window.log_probability(current), abs=1e-12
        )
        assert reverse.log_probability(draw) == pytest.approx(
            window.log_probability(draw), abs=1e-12
        )


def test_proposals_change_clutter_and_do_not_change_existence() -> None:
    model = _small_model()
    initial = np.asarray([[0, -1], [-1, -1], [0, -1]], dtype=np.int64)
    states = np.repeat(initial[None], 200, axis=0)
    proposals, corrections = propose_guided_tracklets(
        states, np.arange(len(states)), model, np.random.default_rng(8)
    )
    model._validate_associations(proposals)
    np.testing.assert_array_equal(proposals != -2, states != -2)
    assert np.all(np.isfinite(corrections))
    assert np.any(np.count_nonzero(proposals >= 0, axis=(1, 2)) != 2)
    for old, new, correction in zip(states[:10], proposals[:10], corrections[:10]):
        window = GuidedPairWindow(model, old, (0, 1), 0)
        assert correction == pytest.approx(
            window.log_probability(old) - window.log_probability(new), abs=1e-12
        )


def test_tracklet_mixture_recovers_small_exact_event_weights() -> None:
    model = _small_model()
    kwargs = {
        "n_slots": 2,
        "localization_std": model.localization_std,
        "coordinate_std": model.coordinate_std,
        "length_scale": model.length_scale,
        "detection_probability": model.detection_probability,
        "expected_clutter_per_frame": model.expected_clutter_per_frame,
        "roi_lower": model.roi_lower,
        "roi_upper": model.roi_upper,
    }
    exact = enumerate_slot_existence_posterior(model.sequence, **kwargs)
    sampled = sample_slot_existence_posterior(
        model.sequence,
        **kwargs,
        config=BirthDeathSMCConfig(
            n_particles=2048,
            mh_sweeps=6,
            final_mh_sweeps=8,
            association_move_probability=0.15,
            interval_move_probability=0.30,
            slot_swap_probability=0.05,
            blocked_slot_move_probability=0.15,
            paired_slot_move_probability=0.20,
            tracklet_move_probability=0.15,
            seed=17,
        ),
    )
    expected = slot_event_marginals(
        exact.associations, model.sequence, probabilities=exact.probabilities
    )
    observed = slot_event_marginals(sampled.associations, model.sequence)
    assert np.max(np.abs(observed.expected_clutter - expected.expected_clutter)) < 0.07
    assert np.max(
        np.abs(observed.existence_probabilities - expected.existence_probabilities)
    ) < 0.07
    assert np.mean([stage.tracklet_acceptance_rate for stage in sampled.stages]) > 0.05
