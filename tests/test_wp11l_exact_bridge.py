"""Frozen eight-frame bridge construction and passive macro-mode observer."""

import numpy as np

from sbulm.slot_existence import ABSENT
from sbulm.wp11l_exact_bridge import (
    CARDINALITIES,
    CLUTTER_RATES,
    OBSERVATION_SHA256,
    SUPPORT_SIZE,
    _BridgeObserver,
    _counts,
    _metadata,
    _mode,
    _protocol,
    _run_order,
)


def test_frozen_bridge_metadata_and_schedule() -> None:
    metadata = _metadata()
    assert metadata["observation"]["detections_sha256"] == OBSERVATION_SHA256
    assert tuple(metadata["observation"]["detection_cardinalities"]) == CARDINALITIES
    assert _protocol()["support_size"] == SUPPORT_SIZE
    assert len(metadata["models"]) == len(CLUTTER_RATES) == 8
    assert [model["expected_clutter_per_frame"] for model in metadata["models"]] == list(
        CLUTTER_RATES
    )
    assert len(_run_order()) == 12
    assert _run_order()[0] == (512, 900)
    assert _run_order()[-1] == (2048, 905)


def test_macromodes_are_label_invariant_and_observer_passive() -> None:
    states = np.full((3, 8, 2), ABSENT, dtype=int)
    states[0, :8, 0] = -1
    states[0, 5, 0] = 0  # A: one long-lived slot, one clutter detection
    states[1, :8, 0] = -1
    states[1, 2:7, 1] = -1
    states[1, 5] = (0, 1)  # B: two overlapping slots, no frame-six clutter
    states[2, :8, 0] = -1
    states[2, 2:7, 1] = -1
    states[2, 5] = (0, -1)  # Other: high alive count with one clutter detection
    before = states.copy()
    np.testing.assert_array_equal(_mode(states), [0, 1, 2])
    np.testing.assert_array_equal(_mode(states[:, :, ::-1]), [0, 1, 2])
    alive, clutter = _counts(states)
    np.testing.assert_array_equal(alive, [8, 13, 13])
    np.testing.assert_array_equal(clutter, [1, 0, 1])
    observer = _BridgeObserver()
    observer.begin_stage(1, 1.0)
    observer.record_resampling(states, np.array([.2, .3, .5]), np.array([2, 1, 1]))
    observer.record_pair(states, states[[1, 0, 2]], np.array([True, True, False]),
                         np.array([True, False, True]))
    observer.end_stage(states)
    record = observer.records[0]
    assert record["accepted_A_to_B"] == {"pg": 1, "other": 0}
    assert record["accepted_B_to_A"] == {"pg": 0, "other": 1}
    assert np.isclose(sum(record["post_mode"]), 1)
    np.testing.assert_array_equal(states, before)
