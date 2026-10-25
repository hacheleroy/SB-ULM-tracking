"""Frozen beta-only PG schedule and exact-oracle result checks."""

import json
from pathlib import Path

import pytest

from sbulm.existence_smc import BirthDeathSMCConfig, _paired_pg_fraction
from sbulm.wp11i_late_pg_oracle import (
    ARMS,
    BETA_SWITCH,
    SEEDS,
    _controls,
    _protocol,
)


def test_schedule_is_state_independent_and_default_is_unchanged() -> None:
    constant, late = _controls("constant"), _controls("late")
    assert _protocol()["seeds"] == list(SEEDS) == [610, 611, 612]
    assert _protocol()["arms"] == list(ARMS)
    assert late.paired_slot_path_particles == constant.paired_slot_path_particles == 4
    assert _paired_pg_fraction(constant, 0.5) == _paired_pg_fraction(constant, 1) == 0.1
    assert _paired_pg_fraction(late, BETA_SWITCH - 0.001) == 0.025
    assert _paired_pg_fraction(late, BETA_SWITCH) == 0.25
    assert _paired_pg_fraction(late, 1) == 0.25
    assert _paired_pg_fraction(BirthDeathSMCConfig(), 0.5) == 1.0
    with pytest.raises(ValueError, match="set together"):
        BirthDeathSMCConfig(paired_slot_pg_beta_switch=0.75)
    with pytest.raises(ValueError, match="conditional_smc scope"):
        BirthDeathSMCConfig(paired_slot_pg_beta_switch=0.75, paired_slot_pg_early_fraction=0)


def test_saved_oracle_respects_frozen_gate() -> None:
    path = Path(__file__).resolve().parents[1] / "results/wp11i_late_pg_oracle.json"
    result = json.loads(path.read_text())
    assert result["protocol"] == _protocol()
    assert set(result["runs"]) == set(ARMS)
    assert all(set(result["runs"][arm]) == {str(seed) for seed in SEEDS} for arm in ARMS)
    passed = (
        all(run["within_exact_envelope"] for arm in result["runs"].values()
            for run in arm.values())
        and result["mean_count_change_rate"]["late"]
        > result["mean_count_change_rate"]["constant"]
    )
    assert result["status"] == (
        "exact_oracle_passed" if passed else "stopped_before_new_development_case"
    )
