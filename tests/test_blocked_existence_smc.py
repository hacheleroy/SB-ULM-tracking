import numpy as np
import pytest

from sbulm.calibration import calibrate_matern_from_coordinate_std
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    _metropolis_birth_death_rejuvenation,
    _propose_blocked_slot_moves,
)
from sbulm.simulation import (
    DetectionCorruptionConfig,
    MultiSlotCrossingConfig,
    simulate_multislot_crossing,
    simulate_slot_existence_observations,
)
from sbulm.slot_existence import SlotExistenceMaternModel


def _three_slot_model() -> SlotExistenceMaternModel:
    crossing = MultiSlotCrossingConfig(
        n_slots=3,
        n_frames=5,
        frame_period=0.001,
        half_span=0.01368,
        central_separation=0.0,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=4)
    truth_mask = np.asarray(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
            [False, True, True],
        ]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=0.72,
            expected_clutter_per_frame=0.15,
            roi_lower=(-0.04, -0.04),
            roi_upper=(0.04, 0.04),
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=2,
    )
    coordinate_std = tuple(
        float(value) for value in np.std(truth.positions, axis=(0, 1))
    )
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    return SlotExistenceMaternModel(
        detections,
        n_slots=3,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=0.72,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )


def test_blocked_slot_hastings_term_exactly_cancels_base_density() -> None:
    model = _three_slot_model()
    rng = np.random.default_rng(8)
    current = model.sample_base_associations(n_samples=256, rng=rng)
    rows = np.arange(current.shape[0], dtype=np.int64)
    proposed, log_hastings = _propose_blocked_slot_moves(
        current, rows, model, rng
    )
    base_log_ratio = (
        model.log_existence_prior(proposed)
        - model.log_existence_prior(current)
        + model.log_count_density(proposed, include_poisson_void=False)
        - model.log_count_density(current, include_poisson_void=False)
        + log_hastings
    )
    np.testing.assert_allclose(base_log_ratio, 0.0, atol=2e-14)


def test_blocked_slot_move_accepts_every_proposal_at_beta_zero() -> None:
    model = _three_slot_model()
    rng = np.random.default_rng(12)
    current = model.sample_base_associations(n_samples=192, rng=rng)
    marginal = model.log_marginal_density(current)
    count = model.log_count_density(current, include_poisson_void=False)
    existence = model.log_existence_prior(current)
    controls = BirthDeathSMCConfig(
        n_particles=current.shape[0],
        mh_sweeps=3,
        final_mh_sweeps=0,
        association_move_probability=0.0,
        interval_move_probability=0.0,
        slot_swap_probability=0.0,
        blocked_slot_move_probability=1.0,
        seed=0,
    )
    updated = _metropolis_birth_death_rejuvenation(
        current,
        marginal,
        count,
        existence,
        model,
        beta=0.0,
        n_sweeps=3,
        config=controls,
        rng=rng,
    )
    assert updated[-1][0] == pytest.approx(1.0)
    assert updated[-1][4] == pytest.approx(1.0)
    assert np.all(np.isfinite(model.log_joint_density(updated[0])))


def test_multislot_crossing_has_one_labelled_detection_per_slot_and_frame() -> None:
    config = MultiSlotCrossingConfig(n_slots=4, n_frames=6, localization_std=0.0)
    truth, detections = simulate_multislot_crossing(config, seed=3)
    assert truth.positions.shape == (6, 4, 2)
    assert detections.reference_ids is not None
    for frame_ids in detections.reference_ids:
        np.testing.assert_array_equal(np.sort(frame_ids), np.arange(4))
    np.testing.assert_allclose(truth.positions[0, 0], [-1.0, 0.0], atol=1e-15)


def test_sequential_base_scales_without_materializing_interval_products() -> None:
    crossing = MultiSlotCrossingConfig(
        n_slots=4,
        n_frames=8,
        frame_period=0.001,
        half_span=0.02394,
        localization_std=0.005,
    )
    truth, clean = simulate_multislot_crossing(crossing, seed=9)
    truth_mask = np.asarray(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=0.70,
            expected_clutter_per_frame=0.15,
            roi_lower=(-0.04, -0.04),
            roi_upper=(0.04, 0.04),
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=5,
    )
    coordinate_std = tuple(
        float(value) for value in np.std(truth.positions, axis=(0, 1))
    )
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    model = SlotExistenceMaternModel(
        detections,
        n_slots=4,
        localization_std=crossing.localization_std,
        coordinate_std=calibration.coordinate_std,
        length_scale=calibration.length_scale,
        detection_probability=0.70,
        expected_clutter_per_frame=0.15,
        roi_lower=(-0.04, -0.04),
        roi_upper=(0.04, 0.04),
    )
    assert model.n_interval_masks**model.n_slots == 1_874_161
    assert model._mask_combinations is None
    samples = model.sample_base_associations(
        n_samples=128, rng=np.random.default_rng(21)
    )
    assert samples.shape == (128, 8, 4)
    assert model._mask_combinations is None
    assert np.all(np.isfinite(model.log_base_probability(samples)))
