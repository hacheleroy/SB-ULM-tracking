"""Exact conditional and small-posterior checks for the full-slot redraw."""

from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sbulm.data import DetectionSequence
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    _metropolis_birth_death_rejuvenation,
    _whole_slot_fraction,
)
from sbulm.slot_existence import ABSENT, SlotExistenceMaternModel, enumerate_slot_existence_support
from sbulm.whole_slot_gibbs import WholeSlotGibbs
from sbulm.wp11af_geometry_transfer_reference import _selected_case
from sbulm.wp11ag_global_slot_comparison import WholeSlotSMCConfig, _candidate

ROOT = Path(__file__).resolve().parents[1]


def _model() -> SlotExistenceMaternModel:
    return SlotExistenceMaternModel(
        DetectionSequence(
            times=np.arange(3) * .001,
            detections=(np.array([[-.008, .002]]), np.empty((0, 2)),
                        np.array([[.008, -.002]])),
        ),
        n_slots=2, localization_std=.005, coordinate_std=(.02, .01),
        length_scale=.003, detection_probability=.7,
        expected_clutter_per_frame=.15,
        roi_lower=(-.04, -.04), roi_upper=(.04, .04),
    )


def test_whole_slot_scores_equal_exact_tempered_conditional() -> None:
    model = _model()
    kernel = WholeSlotGibbs(model)
    support, _ = enumerate_slot_existence_support(model)
    beta = .65
    current = support[len(support) // 3]
    for slot in (0, 1):
        scores = kernel.log_conditional_scores(current, slot, beta)
        valid = np.flatnonzero(np.isfinite(scores))
        proposals = np.broadcast_to(current, (len(valid), *current.shape)).copy()
        proposals[:, :, slot] = kernel.paths[valid]
        exact = (model.log_existence_prior(proposals)
                 + model.log_count_density(proposals, include_poisson_void=False)
                 + beta * model.log_marginal_density(proposals))
        np.testing.assert_allclose(scores[valid] - logsumexp(scores[valid]),
                                   exact - logsumexp(exact), atol=1e-12)
        assert np.any(np.count_nonzero(proposals != ABSENT, axis=(1, 2))
                      != np.count_nonzero(current != ABSENT))


def test_whole_slot_redraw_keeps_exact_tempered_distribution() -> None:
    model = _model()
    kernel = WholeSlotGibbs(model)
    support, _ = enumerate_slot_existence_support(model)
    beta = .65
    target = (model.log_existence_prior(support)
              + model.log_count_density(support, include_poisson_void=False)
              + beta * model.log_marginal_density(support))
    probabilities = np.exp(target - logsumexp(target))
    rng = np.random.default_rng(734)
    current = support[rng.choice(len(support), size=2_048, p=probabilities)]
    rows = np.arange(len(current), dtype=np.int64)
    for _ in range(3):
        current = kernel.sample(current, rows, beta, rng)
    labels = np.count_nonzero(support != ABSENT, axis=(1, 2))
    expected = np.bincount(labels, weights=probabilities, minlength=7)
    actual = np.bincount(np.count_nonzero(current != ABSENT, axis=(1, 2)),
                         minlength=7) / len(current)
    assert np.abs(actual - expected).sum() / 2 < .06


def test_opt_in_schedule_keeps_early_control_bitwise_identical() -> None:
    model = _model()
    control = BirthDeathSMCConfig(
        n_particles=48, association_move_probability=0,
        interval_move_probability=0, slot_swap_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=.1, paired_slot_path_particles=4,
    )
    candidate = WholeSlotSMCConfig(**vars(control))
    assert _candidate(1300).paired_slot_whole_slot_fraction == .20
    assert _whole_slot_fraction(candidate, .64) == 0
    assert _whole_slot_fraction(candidate, .65) == .20
    states = model.sample_base_associations(n_samples=48, rng=np.random.default_rng(34))

    def rejuvenate(config: BirthDeathSMCConfig):
        return _metropolis_birth_death_rejuvenation(
            states.copy(), model.log_marginal_density(states),
            model.log_count_density(states, include_poisson_void=False),
            model.log_existence_prior(states), model, beta=.64, n_sweeps=2,
            config=config, rng=np.random.default_rng(701),
        )

    baseline, early = rejuvenate(control), rejuvenate(candidate)
    for left, right in zip(baseline[:4], early[:4], strict=True):
        np.testing.assert_array_equal(left, right)
    assert baseline[-1] == early[-1]


def test_independent_eight_frame_conditional_log_ratios() -> None:
    case, _ = _selected_case(ROOT / "results/wp11af_geometry_transfer_ledger.json")
    model = case.model
    kernel = WholeSlotGibbs(model)
    current = model.sample_base_associations(
        n_samples=1, rng=np.random.default_rng(734)
    )[0]
    scores = kernel.log_conditional_scores(current, 0, .65)
    valid = np.flatnonzero(np.isfinite(scores))
    indices = valid[np.linspace(0, len(valid) - 1, 12, dtype=int)]
    proposals = np.broadcast_to(current, (len(indices), *current.shape)).copy()
    proposals[:, :, 0] = kernel.paths[indices]
    exact = (model.log_existence_prior(proposals)
             + model.log_count_density(proposals, include_poisson_void=False)
             + .65 * model.log_marginal_density(proposals))
    np.testing.assert_allclose(scores[indices] - scores[indices[0]],
                               exact - exact[0], atol=1e-12)
