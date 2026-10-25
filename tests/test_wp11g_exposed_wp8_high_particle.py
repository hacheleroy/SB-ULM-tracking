"""Frozen protocol checks that do not launch the costly 8,192-particle run."""

import copy
import json
from pathlib import Path

import pytest

from sbulm.wp7_convergence_audit import _compare_large
from sbulm.wp11g_exposed_wp8_high_particle import (
    N_PARTICLES,
    SEEDS,
    _candidate_controls,
    _finish,
    _frozen_inputs,
    _protocol,
)

CONTROL = Path(__file__).resolve().parents[1] / "results/wp8_generalization.json"
RESULT = Path(__file__).resolve().parents[1] / "results/wp11g_exposed_wp8_high_particle.json"


def test_frozen_control_and_candidate_scope():
    case, record, metadata = _frozen_inputs(CONTROL)
    assert case.model.sequence.cardinalities == (1, 4, 4, 4, 3, 5, 1, 1)
    assert metadata["model_and_detections"]["detections_sha256"] == _protocol()["observation_sha256"]
    assert record["cross_run"]["stability_passed"] is False
    assert max(record["cross_run"]["metrics"].values()) > 0.20
    for seed in SEEDS:
        config = _candidate_controls(seed)
        assert config.n_particles == N_PARTICLES
        assert config.seed == seed
        assert config.paired_slot_scope == "conditional_smc"
        assert config.paired_slot_pg_fraction == 0.1
        assert config.paired_slot_path_particles == 4
        assert config.paired_slot_move_probability == 0.25
        assert config.mh_sweeps == config.final_mh_sweeps == 8
    assert _protocol()["maximum_candidate_runs"] == 2


def test_frozen_control_rejects_mutated_copy(tmp_path):
    altered = tmp_path / "control.json"
    altered.write_bytes(CONTROL.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash changed"):
        _frozen_inputs(altered)


def test_two_seed_finish_uses_inherited_seven_metrics():
    _, record, _ = _frozen_inputs(CONTROL)
    first = record["runs"][str(SEEDS[0])]
    second = record["runs"][str(SEEDS[1])]
    result = {
        "control_cross_run": record["cross_run"],
        "candidate_runs": {str(SEEDS[0]): {"summary": copy.deepcopy(first)}},
        "candidate_cross_run": None,
        "status": "in_progress",
    }
    _finish(result)
    assert result["status"] == "in_progress"
    result["candidate_runs"][str(SEEDS[1])] = {"summary": copy.deepcopy(second)}
    _finish(result)
    assert result["status"] == "failed_early_before_new_holdout"
    assert result["candidate_cross_run"]["metrics"] == _compare_large([first, second])["metrics"]
    identical = copy.deepcopy(first)
    identical["seed"] = SEEDS[1]
    result["candidate_runs"][str(SEEDS[1])]["summary"] = identical
    _finish(result)
    assert result["status"] == "two_seed_exposed_case_passed"
    assert all(value == 0 for value in result["candidate_cross_run"]["metrics"].values())


def test_saved_exposed_case_failure_is_reproducible():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "failed_early_before_new_holdout"
    assert set(result["candidate_runs"]) == {str(seed) for seed in SEEDS}
    recalculated = copy.deepcopy(result)
    _finish(recalculated)
    assert recalculated["candidate_cross_run"] == result["candidate_cross_run"]
    checks = result["candidate_cross_run"]["checks"]
    assert {name for name, passed in checks.items() if not passed} == {
        "maximum_clutter_marginal_difference", "maximum_alive_count_tv"
    }
    for name, value in result["candidate_cross_run"]["metrics"].items():
        assert value < result["control_cross_run"]["metrics"][name]
    for run in result["candidate_runs"].values():
        assert run["cpu_seconds"] < result["protocol"]["maximum_process_cpu_seconds_per_candidate_run"]
        assert 0.07 <= run["mean_stage_pg_selection_fraction"] <= 0.13
