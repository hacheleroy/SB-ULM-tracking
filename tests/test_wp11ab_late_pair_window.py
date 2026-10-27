"""Target and protocol checks for the late local pair window proposal."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.data import DetectionSequence
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    _late_pair_window_weights,
    _metropolis_birth_death_rejuvenation,
)
from sbulm.paired_moves import _local_pair_candidate_set
from sbulm.slot_existence import SlotExistenceMaternModel, enumerate_slot_existence_support
from sbulm.wp11ab_late_pair_window import (
    LateWindowSMCConfig,
    _candidate,
    _inputs,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def _toy_model() -> SlotExistenceMaternModel:
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


def test_protocol_and_pinned_result_are_unchanged() -> None:
    assert "paired_slot_late_window_beta_switch" not in asdict(
        BirthDeathSMCConfig()
    )
    candidate = _candidate(1300)
    assert candidate.paired_slot_late_window_start_frames == (5, 7)
    assert candidate.paired_slot_pg_fraction == .1
    assert candidate.paired_slot_path_particles == 4
    previous, prior = _inputs(
        ROOT / "results/wp11u_late_suffix_pg.json",
        ROOT / "results/wp11y_independent_absolute_sampler.json",
        ROOT / "results/wp11z_balanced_stage_summary.json",
        ROOT / "results/wp11aa_tempered_suffix_pg.json",
    )
    assert len(previous["primary_19"]["runs"]) == 6
    assert len(prior["cases"]["decoy_17"]["runs"]) == 4
    assert _protocol()["cost"]["each_candidate_over_archived_control_at_most"] == 1.25


def test_late_window_weights_and_early_bitwise_control_equivalence() -> None:
    base = BirthDeathSMCConfig(
        n_particles=48, association_move_probability=0,
        interval_move_probability=0, slot_swap_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=.1, paired_slot_path_particles=4,
    )
    candidate = LateWindowSMCConfig(
        **asdict(base), paired_slot_late_window_start_frames=(0, 2)
    )
    assert _late_pair_window_weights(candidate, .74, 3) is None
    np.testing.assert_allclose(_late_pair_window_weights(candidate, .75, 3),
                               [.5 / 3 + .5 * .75, .5 / 3,
                                .5 / 3 + .5 * .25], atol=1e-15)
    np.testing.assert_allclose(_late_pair_window_weights(_candidate(1300), .75, 8),
                               [.0625] * 5 + [.4375, .0625, .1875])
    model = _toy_model()
    states = model.sample_base_associations(n_samples=48, rng=np.random.default_rng(19))

    def move(config: BirthDeathSMCConfig, beta: float):
        return _metropolis_birth_death_rejuvenation(
            states.copy(), model.log_marginal_density(states),
            model.log_count_density(states, include_poisson_void=False),
            model.log_existence_prior(states), model, beta=beta,
            n_sweeps=2, config=config, rng=np.random.default_rng(49),
        )

    control, early = move(base, .74), move(candidate, .74)
    for left, right in zip(control[:4], early[:4], strict=True):
        np.testing.assert_array_equal(left, right)
    assert control[-1] == early[-1]


def test_weighted_pair_windows_preserve_tempered_small_posterior() -> None:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    beta = .75
    log_target = (model.log_existence_prior(support)
                  + model.log_count_density(support, include_poisson_void=False)
                  + beta * model.log_marginal_density(support))
    exact = np.exp(log_target - logsumexp(log_target))
    rng = np.random.default_rng(277)
    current = support[rng.choice(len(support), size=1_536, p=exact)].copy()
    config = LateWindowSMCConfig(
        n_particles=len(current), association_move_probability=0,
        interval_move_probability=0, slot_swap_probability=0,
        paired_slot_move_probability=1, paired_slot_scope="conditional_smc",
        paired_slot_pg_fraction=0, paired_slot_late_window_start_frames=(0, 2),
    )
    updated = _metropolis_birth_death_rejuvenation(
        current, model.log_marginal_density(current),
        model.log_count_density(current, include_poisson_void=False),
        model.log_existence_prior(current), model, beta=beta, n_sweeps=2,
        config=config, rng=np.random.default_rng(279),
    )[0]
    for value_fn in (
        lambda states: np.count_nonzero(states != -2, axis=(1, 2)),
        lambda states: 1 - np.count_nonzero(states[:, 2] >= 0, axis=1),
    ):
        labels = value_fn(support)
        expected = np.bincount(labels, weights=exact, minlength=7)
        actual = np.bincount(value_fn(updated), minlength=7) / len(updated)
        assert np.abs(actual - expected).sum() / 2 < .075
    # For each fixed window, the reverse candidate set is unchanged.
    example = support[len(support) // 2]
    for start in (0, 1, 2):
        options = _local_pair_candidate_set(model, example, (0, 1), start, 1)
        for alternate in options:
            reverse = _local_pair_candidate_set(model, alternate, (0, 1), start, 1)
            assert {x.tobytes() for x in options} == {x.tobytes() for x in reverse}


def test_invalid_late_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="beta switch"):
        LateWindowSMCConfig(paired_slot_scope="conditional_smc",
                            paired_slot_late_window_beta_switch=0)
    with pytest.raises(ValueError, match="shares"):
        LateWindowSMCConfig(paired_slot_scope="conditional_smc",
                            paired_slot_late_window_focus_share=1.1)
