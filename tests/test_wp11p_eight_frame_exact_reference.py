"""Independent label-symmetry enumerator and frozen WP11p schedule."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sbulm.data import DetectionSequence
from sbulm.slot_existence import (
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
)
from sbulm.wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR
from sbulm.wp11p_eight_frame_exact_reference import (
    CANONICAL_SUPPORT_SIZE,
    CARDINALITIES,
    CLUTTER_RATES,
    LABELLED_SUPPORT_SIZE,
    OBSERVATION_HASH,
    SEEDS,
    _Accumulation,
    _canonical_blocks,
    _case,
    _counts,
    _from_joint,
    _mask_multiplicity,
    _metadata,
    _mode,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_eight_frame_geometry_and_schedule() -> None:
    info = _metadata()
    assert info["observation"]["detections_sha256"] == OBSERVATION_HASH
    assert tuple(info["observation"]["detection_cardinalities"]) == CARDINALITIES
    assert len(info["models"]) == len(CLUTTER_RATES) == 8
    assert _case(1.0).model.n_slots == 4
    assert _protocol()["canonical_support_size"] == CANONICAL_SUPPORT_SIZE
    assert _protocol()["labelled_support_size"] == LABELLED_SUPPORT_SIZE
    assert _protocol()["seeds"] == list(SEEDS)
    assert _mask_multiplicity((0, 1, 2, 3)) == 24
    assert _mask_multiplicity((0, 0, 1, 2)) == 12
    assert _mask_multiplicity((0, 0, 1, 1)) == 6
    assert _mask_multiplicity((0, 0, 0, 0)) == 1


def test_weighted_mask_multisets_match_complete_labelled_sum() -> None:
    sequence = DetectionSequence(
        times=np.arange(3) * .001,
        detections=(np.array([[0., 0.]]), np.empty((0, 2)),
                    np.array([[.002, 0.]])),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=2, localization_std=.005,
        coordinate_std=(.02, .01), length_scale=.003,
        detection_probability=.55, expected_clutter_per_frame=.5,
        roi_lower=(-.04, -.04), roi_upper=(.04, .04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    complete, _ = enumerate_slot_existence_support(model)
    scores = model.log_joint_density(complete)
    count, canonical_logs, canonical_alive = 0, [], []
    for block, multiplicity, _ in _canonical_blocks(model):
        count += len(block) * multiplicity
        canonical_logs.append(model.log_joint_density(block) + np.log(multiplicity))
        canonical_alive.append(np.count_nonzero(block != -2, axis=(1, 2)))
    assert count == len(complete)
    weighted = np.concatenate(canonical_logs)
    assert np.isclose(logsumexp(weighted), logsumexp(scores), atol=1e-12)
    canonical_hist = np.bincount(
        np.concatenate(canonical_alive), weights=np.exp(weighted - logsumexp(weighted)),
        minlength=7,
    )
    labelled_hist = np.bincount(
        np.count_nonzero(complete != -2, axis=(1, 2)),
        weights=np.exp(scores - logsumexp(scores)), minlength=7,
    )
    np.testing.assert_allclose(canonical_hist, labelled_hist, atol=1e-12)


def test_stable_histogram_and_label_invariant_modes() -> None:
    accumulator = _Accumulation(2)
    accumulator.update(np.array([-1000., -1001.]), np.array([0, 1]))
    accumulator.update(np.array([0.]), np.array([1]))
    assert np.isclose(accumulator.log_total(), 0.)
    np.testing.assert_allclose(accumulator.normalized(), [0., 1.], atol=1e-12)

    states = np.full((2, 8, 4), -2)
    states[:, :, 0] = -1
    states[0, 5, 0] = 0
    states[1, :, 1] = -1
    states[1, 5, :2] = (0, 1)
    first = _counts(states)
    second = _counts(states[:, :, ::-1])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    joint = np.zeros((33, 3))
    joint[16, 1], joint[17, 0] = .4, .6
    assert _from_joint(joint)["mode"] == [.4, .6, 0.]
    np.testing.assert_array_equal(_mode(np.array([16, 17]), np.array([1, 0])), [0, 1])


def test_invalid_bridge_checkpoint_stopped_before_sampler() -> None:
    result = json.loads((ROOT / "results/wp11p_eight_frame_exact_reference.json").read_text())
    ref = result["reference"]
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    assert ref["canonical_support_size"] == CANONICAL_SUPPORT_SIZE
    assert ref["labelled_support_size"] == LABELLED_SUPPORT_SIZE
    assert not ref["difficulty_valid"]
    assert ref["exact"]["mode"][1] < .10
    assert result["status"] == "frozen_eight_frame_bridge_invalid_stop"
    assert all(not runs for runs in result["runs"].values())
