"""Exact finite-state checks for conditional-GP, non-self tracklet updates."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from test_tracklet_moves import _small_model

from sbulm.conditional_tracklet_moves import (
    ConditionalGPWindow,
    _invariant_windows,
    propose_conditional_gp_tracklets,
)
from sbulm.slot_existence import slot_existence_support_size
from sbulm.wp7_convergence_audit import _make_case


def _support(window: ConditionalGPWindow) -> np.ndarray:
    candidates = []
    for frames in product(*window.options):
        candidate = window.current.copy()
        for offset, frame in enumerate(frames):
            candidate[window.start + offset, window.slots[0]] = frame[0]
            candidate[window.start + offset, window.slots[1]] = frame[1]
        candidates.append(candidate)
    return np.stack(candidates)


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_nonself_kernel_satisfies_pairwise_detailed_balance(beta: float) -> None:
    model = _small_model()
    current = np.asarray([[0, -1], [-1, -1], [0, -1]], dtype=np.int64)
    window = ConditionalGPWindow(model, current, (0, 1), 0)
    support = _support(window)
    log_q = np.asarray([window.log_probability(x) for x in support])
    assert logsumexp(log_q) == pytest.approx(0.0, abs=1e-12)
    assert np.max(log_q) < -1e-5
    log_except = np.log(-np.expm1(log_q))
    target = (
        model.log_existence_prior(support)
        + model.log_count_density(support, include_poisson_void=False)
        + beta * model.log_marginal_density(support)
    )
    log_pi = target - logsumexp(target)
    log_ratio = (
        target[None, :] - target[:, None]
        + log_q[:, None] - log_q[None, :]
        + log_except[:, None] - log_except[None, :]
    )
    log_flow = (
        log_pi[:, None] + log_q[None, :] - log_except[:, None]
        + np.minimum(0.0, log_ratio)
    )
    off_diagonal = ~np.eye(len(support), dtype=bool)
    np.testing.assert_allclose(
        log_flow[off_diagonal], log_flow.T[off_diagonal], atol=1e-12, rtol=0
    )
    for old in support[::4]:
        conditional = ConditionalGPWindow(model, old, (0, 1), 0)
        np.testing.assert_allclose(
            [conditional.log_probability(x) for x in support], log_q, atol=1e-12
        )
        proposed, correction = conditional.propose_nonself(np.random.default_rng(81))
        assert not np.array_equal(proposed, old)
        old_q = conditional.log_probability(old)
        new_q = conditional.log_probability(proposed)
        assert correction == pytest.approx(
            old_q - new_q + np.log(-np.expm1(old_q))
            - np.log(-np.expm1(new_q)), abs=1e-12
        )


def test_four_slot_bridge_proposals_preserve_masks_and_reverse_context() -> None:
    case = _make_case(bridge=True)
    model = case.model
    assert slot_existence_support_size(model) == 282_469
    current = model.sample_base_associations(n_samples=24, rng=np.random.default_rng(42))
    rows = np.arange(len(current))
    proposed, correction = propose_conditional_gp_tracklets(
        current, rows, model, np.random.default_rng(19)
    )
    model._validate_associations(proposed)
    np.testing.assert_array_equal(proposed != -2, current != -2)
    assert np.all(np.isfinite(correction))
    assert np.any(np.any(proposed != current, axis=(1, 2)))
    for before, after in zip(current, proposed, strict=True):
        assert _invariant_windows(
            (before != -2).tobytes(), model.sequence.n_frames, model.n_slots,
            model.sequence.cardinalities, 3
        ) == _invariant_windows(
            (after != -2).tobytes(), model.sequence.n_frames, model.n_slots,
            model.sequence.cardinalities, 3
        )


def test_nonself_changes_active_associations_in_cluttered_case() -> None:
    from sbulm.wp8_generalization import REGIMES, make_case

    model = make_case(REGIMES[1]).model
    rng = np.random.default_rng(44)
    current = model.sample_base_associations(n_samples=64, rng=rng)
    proposed, correction = propose_conditional_gp_tracklets(
        current, np.arange(len(current)), model, rng
    )
    model._validate_associations(proposed)
    np.testing.assert_array_equal(proposed != -2, current != -2)
    assert np.all(np.isfinite(correction))
    assert np.count_nonzero(np.any(proposed != current, axis=(1, 2))) > 5
