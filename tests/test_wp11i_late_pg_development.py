"""Frozen fresh development observation and late-PG controls."""

import json
from pathlib import Path

from sbulm.wp8_generalization import _metadata, make_case
from sbulm.wp11i_late_pg_development import (
    ARMS,
    CARDINALITIES,
    OBSERVATION_SHA256,
    REGIME,
    SEEDS,
    _controls,
    _protocol,
)


def test_fresh_case_and_schedule_frozen_before_inference() -> None:
    case = make_case(REGIME)
    observed = _metadata(REGIME, case)["model_and_detections"]
    assert observed["detections_sha256"] == OBSERVATION_SHA256
    assert tuple(observed["detection_cardinalities"]) == CARDINALITIES
    protocol = _protocol()
    assert protocol["seeds"] == list(SEEDS) == [510, 511]
    assert protocol["arms"] == list(ARMS) == ["constant", "late"]
    assert protocol["arm_order_by_seed"] == {
        "510": ["constant", "late"], "511": ["late", "constant"]
    }
    control, late = _controls(510, "constant"), _controls(510, "late")
    assert control.n_particles == late.n_particles == 512
    assert control.paired_slot_pg_fraction == 0.1
    assert control.paired_slot_pg_beta_switch is None
    assert late.paired_slot_pg_beta_switch == 0.75
    assert late.paired_slot_pg_early_fraction == 0.025
    assert late.paired_slot_pg_fraction == 0.25
    assert late.paired_slot_path_particles == control.paired_slot_path_particles == 4


def test_saved_development_gate_reloads() -> None:
    path = Path(__file__).resolve().parents[1] / "results/wp11i_late_pg_development.json"
    result = json.loads(path.read_text())
    assert result["protocol"] == _protocol()
    assert set(result["runs"]) == {str(seed) for seed in SEEDS}
    assert all(set(result["runs"][str(seed)]) == set(ARMS) for seed in SEEDS)
    passed = (
        all(check["passed"] for check in result["seed_checks"].values())
        and result["target_metrics_improved"]
        and result["other_metrics_within_tolerance"]
    )
    assert result["status"] == (
        "development_screen_passed" if passed else "stopped_before_high_particle_test"
    )
