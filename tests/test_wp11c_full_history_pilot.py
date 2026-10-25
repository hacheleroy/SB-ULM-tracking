"""Frozen protocol and early-stop artifact for WP11c."""

import json
from pathlib import Path

import pytest

from sbulm.slot_existence import enumerate_slot_existence_support
from sbulm.wp11c_full_history_pilot import (
    ARMS,
    MINIMUM_MEAN_COUNT_CHANGE_RATE,
    SEEDS,
    _pilot_model,
    _protocol,
)


def test_wp11c_small_case_and_frozen_early_stop() -> None:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    saved = json.loads(Path("results/wp11c_full_history_pilot.json").read_text())
    assert len(support) == saved["exact_support_size"] == 269
    assert saved["protocol"] == _protocol(model)
    assert saved["protocol"]["seeds"] == list(SEEDS)
    assert saved["protocol"]["arms"] == list(ARMS)
    assert saved["status"] == "stopped_before_fresh_holdout"
    assert saved["mean_full_history_count_change_rate"] < MINIMUM_MEAN_COUNT_CHANGE_RATE
    assert saved["base_posterior_overlap"]["total_variation"] == pytest.approx(0.9888814)
    assert saved["base_posterior_overlap"]["posterior_credible_set_95_size"] == 39
    assert saved["base_posterior_overlap"][
        "base_mass_on_posterior_credible_set_95"
    ] == pytest.approx(0.00128154, abs=1e-7)
    assert all(set(saved["runs"][arm]) == {str(seed) for seed in SEEDS} for arm in ARMS)
