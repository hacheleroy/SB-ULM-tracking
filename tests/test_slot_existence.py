from math import log

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.simulation import (
    DetectionCorruptionConfig,
    simulate_crossing,
    simulate_slot_existence_observations,
)
from sbulm.slot_existence import (
    ABSENT,
    MISSED,
    BirthDeathPrior,
    SlotExistenceMaternModel,
    enumerate_contiguous_existence_masks,
    enumerate_frame_slot_associations,
    enumerate_slot_existence_posterior,
    enumerate_slot_existence_support,
    reference_slot_association,
    slot_event_marginals,
    slot_existence_support_size,
)

ROI_LOWER = (-0.04, -0.04)
ROI_UPPER = (0.04, 0.04)


def _problem() -> tuple[object, object, np.ndarray, tuple[float, ...], float]:
    config = physical_crossing_config(
        n_frames=4,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=4)
    truth_mask = np.asarray(
        [[True, False], [True, True], [True, True], [False, True]]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=0.78,
            expected_clutter_per_frame=0.35,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=3,
    )
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    return (
        config,
        detections,
        truth_mask,
        calibration.coordinate_std,
        calibration.length_scale,
    )


def _model() -> tuple[SlotExistenceMaternModel, object, np.ndarray]:
    config, detections, truth_mask, coordinate_std, length_scale = _problem()
    model = SlotExistenceMaternModel(
        detections,
        n_slots=2,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        detection_probability=0.78,
        expected_clutter_per_frame=0.35,
        roi_lower=ROI_LOWER,
        roi_upper=ROI_UPPER,
    )
    return model, detections, truth_mask


def test_birth_death_prior_enumerates_a_normalized_partition() -> None:
    prior = BirthDeathPrior(
        initial_probability=0.4,
        birth_probability=0.3,
        survival_probability=0.8,
    )
    masks = enumerate_contiguous_existence_masks(4)
    assert masks.shape == (11, 4)
    probabilities = np.exp(prior.log_probability(masks))
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(0.6 * 0.7**3)
    full = np.flatnonzero(np.all(masks, axis=1))[0]
    assert probabilities[full] == pytest.approx(0.4 * 0.8**3)
    with pytest.raises(ValueError, match="contiguous"):
        prior.log_probability(np.asarray([[True, False, True, False]]))


def test_frame_states_distinguish_absent_missed_and_assigned() -> None:
    states = enumerate_frame_slot_associations([True, False, True], 2)
    assert states.shape == (7, 3)
    assert np.all(states[:, 1] == ABSENT)
    assert np.any(states[:, 0] == MISSED)
    for state in states:
        assigned = state[state >= 0]
        assert np.unique(assigned).size == assigned.size


def test_simulator_suppresses_absent_slot_detections() -> None:
    _, detections, truth_mask, _, _ = _problem()
    assert detections.reference_ids is not None
    for time_index, frame_ids in enumerate(detections.reference_ids):
        for slot in range(truth_mask.shape[1]):
            if not truth_mask[time_index, slot]:
                assert slot not in frame_ids
    truth = reference_slot_association(detections, truth_mask)
    np.testing.assert_array_equal(truth != ABSENT, truth_mask)


def test_simulator_accepts_a_frame_with_no_alive_slots() -> None:
    config = physical_crossing_config(n_frames=4)
    _, clean = simulate_crossing(config, seed=0)
    truth_mask = np.asarray(
        [[False, False], [True, False], [True, True], [False, True]]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=0.8,
            expected_clutter_per_frame=0.1,
            roi_lower=ROI_LOWER,
            roi_upper=ROI_UPPER,
            anchor_first_frame=False,
            max_clutter_per_frame=0,
        ),
        seed=1,
    )
    assert detections.cardinalities[0] == 0
    np.testing.assert_array_equal(
        reference_slot_association(detections, truth_mask)[0],
        np.asarray([ABSENT, ABSENT]),
    )


def test_exact_support_base_and_posterior_are_normalized() -> None:
    model, detections, _ = _model()
    associations, masks = enumerate_slot_existence_support(model)
    assert associations.shape[0] == slot_existence_support_size(model)
    np.testing.assert_array_equal(masks, associations != ABSENT)
    assert logsumexp(model.log_base_probability(associations)) == pytest.approx(
        0.0, abs=1e-12
    )

    posterior = enumerate_slot_existence_posterior(
        detections,
        n_slots=2,
        localization_std=model.localization_std,
        coordinate_std=model.coordinate_std,
        length_scale=model.length_scale,
        detection_probability=model.detection_probability,
        expected_clutter_per_frame=model.expected_clutter_per_frame,
        roi_lower=model.roi_lower,
        roi_upper=model.roi_upper,
    )
    assert posterior.associations.shape == associations.shape
    assert logsumexp(posterior.log_probabilities) == pytest.approx(0.0)

    # Exchangeable slot priors make a global label permutation equiprobable.
    swapped = posterior.associations[posterior.map_index, :, ::-1][None]
    assert model.log_joint_density(swapped)[0] == pytest.approx(
        model.log_joint_density(
            posterior.associations[posterior.map_index][None]
        )[0]
    )


def test_event_marginals_and_conditional_paths_respect_masks() -> None:
    model, detections, _ = _model()
    posterior = enumerate_slot_existence_posterior(
        detections,
        n_slots=2,
        localization_std=model.localization_std,
        coordinate_std=model.coordinate_std,
        length_scale=model.length_scale,
        detection_probability=model.detection_probability,
        expected_clutter_per_frame=model.expected_clutter_per_frame,
        roi_lower=model.roi_lower,
        roi_upper=model.roi_upper,
    )
    marginals = slot_event_marginals(
        posterior.associations,
        detections,
        probabilities=posterior.probabilities,
    )
    assert marginals.existence_probabilities.shape == (detections.n_frames, 2)
    assert marginals.missed_probabilities.shape == (detections.n_frames, 2)
    assert marginals.expected_clutter.shape == (detections.n_frames,)
    np.testing.assert_allclose(marginals.birth_time_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(marginals.death_time_probabilities.sum(axis=1), 1.0)
    assert np.all(
        marginals.missed_probabilities
        <= marginals.existence_probabilities + 1e-12
    )

    selected = posterior.associations[[posterior.map_index]]
    paths = model.sample_conditional_paths(selected, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(
        np.isnan(paths[..., 0]), selected == ABSENT
    )
    assert np.all(np.isfinite(paths[selected != ABSENT]))


def test_count_density_counts_only_alive_misses() -> None:
    model, _, _ = _model()
    association = np.full((1, model.sequence.n_frames, 2), ABSENT, dtype=np.int64)
    association[0, 0, 0] = MISSED
    association[0, 1, 0] = 0
    assigned = 1
    missed = 1
    clutter = sum(model.sequence.cardinalities) - assigned
    expected = (
        assigned * log(model.detection_probability)
        + missed * log(1.0 - model.detection_probability)
        + clutter * log(model.clutter_intensity)
        - model.sequence.n_frames * model.expected_clutter_per_frame
    )
    assert model.log_count_density(association)[0] == pytest.approx(expected)
