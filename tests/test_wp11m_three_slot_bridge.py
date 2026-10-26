"""Frozen WP11m construction and label-invariant passive mode observer."""

import numpy as np

from sbulm.data import DetectionSequence
from sbulm.slot_existence import (
    ABSENT,
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
)
from sbulm.wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR
from sbulm.wp11m_three_slot_bridge import (
    CARDINALITIES,
    CLUTTER_RATES,
    OBSERVATION_SHA256,
    SUPPORT_SIZE,
    _counts,
    _metadata,
    _mode,
    _ModeObserver,
    _protocol,
    _run_order,
    _support_blocks,
)


def test_frozen_three_slot_case_and_schedule() -> None:
    metadata = _metadata()
    assert metadata["observation"]["detections_sha256"] == OBSERVATION_SHA256
    assert tuple(metadata["observation"]["detection_cardinalities"]) == CARDINALITIES
    assert _protocol()["support_size"] == SUPPORT_SIZE
    assert len(metadata["models"]) == len(CLUTTER_RATES) == 8
    assert [m["expected_clutter_per_frame"] for m in metadata["models"]] == list(
        CLUTTER_RATES
    )
    assert _run_order()[0] == (512, 920)
    assert _run_order()[-1] == (2048, 925)
    assert len(_run_order()) == 12


def test_label_invariant_modes_and_passive_trace() -> None:
    states = np.full((3, 8, 3), ABSENT, dtype=int)
    states[0, :8, 0] = -1
    states[0, 5, 0] = 0
    states[1, :8, 0] = -1
    states[1, 2:7, 1] = -1
    states[1, 5] = (0, 1, ABSENT)
    states[2, :8, 0] = -1
    states[2, 2:7, 1] = -1
    states[2, 5] = (0, -1, ABSENT)
    original = states.copy()
    alive, clutter = _counts(states)
    np.testing.assert_array_equal(alive, [8, 13, 13])
    np.testing.assert_array_equal(clutter, [1, 0, 1])
    np.testing.assert_array_equal(_mode(alive, clutter), [0, 1, 2])
    np.testing.assert_array_equal(_mode(*_counts(states[:, :, ::-1])), [0, 1, 2])
    observer = _ModeObserver()
    observer.begin_stage(1, 1.0)
    observer.record_resampling(states, np.array([.2, .3, .5]), np.array([2, 1, 1]))
    observer.record_pair(states, states[[1, 0, 2]], np.array([True, True, False]),
                         np.array([True, False, True]))
    observer.end_stage(states)
    record = observer.records[0]
    assert record["accepted_A_to_B"] == {"pg": 1, "other": 0}
    assert record["accepted_B_to_A"] == {"pg": 0, "other": 1}
    assert np.isclose(sum(record["post_mode"]), 1)
    np.testing.assert_array_equal(states, original)


def test_streamed_support_matches_complete_small_enumeration() -> None:
    sequence = DetectionSequence(
        times=np.arange(3) * .001,
        detections=(np.array([[0.0, 0.0]]), np.empty((0, 2)),
                    np.array([[.002, 0.0]])),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=2, localization_std=.005,
        coordinate_std=(.02, .01), length_scale=.003,
        detection_probability=.55, expected_clutter_per_frame=1.0,
        roi_lower=(-.04, -.04), roi_upper=(.04, .04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    exact, _ = enumerate_slot_existence_support(model)
    streamed = np.concatenate(list(_support_blocks(model)))
    np.testing.assert_array_equal(streamed, exact)
