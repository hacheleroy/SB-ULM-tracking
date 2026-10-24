"""Regression checks for the fixed four-slot convergence audit."""

import numpy as np

from sbulm.slot_existence import slot_existence_support_size
from sbulm.wp7_convergence_audit import (
    _case_metadata,
    _compare_large,
    _make_case,
)


def test_bridge_is_exactly_enumerable_and_large_data_are_frozen() -> None:
    bridge = _make_case(bridge=True)
    large = _make_case(bridge=False)
    assert bridge.model.sequence.cardinalities == (1, 0, 1, 1)
    assert slot_existence_support_size(bridge.model) == 282_469
    assert _case_metadata(bridge)["detections_sha256"] == (
        "62d3aeeafff58eb47404f87291829004f5bd498d07ab1494afa86ba77115b8ef"
    )
    assert large.model.sequence.cardinalities == (0, 2, 4, 3, 4, 2, 1, 1)


def test_cross_run_stability_is_order_independent_and_inclusive_at_gate() -> None:
    common = {
        "existence": [[0.1]],
        "missed": [[0.1]],
        "clutter": [0.1],
        "birth": [[0.1]],
        "death": [[0.1]],
    }
    left = {
        "seed": 10,
        "runtime_seconds": 1.0,
        "event_marginals": common,
        "alive_count_histogram": [0.5, 0.5],
        "assigned_count_histogram": [0.5, 0.5],
    }
    right = {
        "seed": 11,
        "runtime_seconds": 2.0,
        "event_marginals": {
            **common,
            "clutter": [0.2],
        },
        "alive_count_histogram": [0.4, 0.6],
        "assigned_count_histogram": [0.45, 0.55],
    }
    first = _compare_large([left, right])
    second = _compare_large([right, left])
    assert first["stability_passed"]
    assert second["stability_passed"]
    assert np.isclose(
        first["metrics"]["maximum_alive_count_tv"],
        second["metrics"]["maximum_alive_count_tv"],
    )
    assert first["metrics"]["maximum_clutter_marginal_difference"] <= 0.10
