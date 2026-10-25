"""Frozen selective-PG development case and arm controls."""

import json
from pathlib import Path

import pytest

from sbulm.wp8_generalization import _metadata, make_case
from sbulm.wp11e_selective_pg_development import (
    ARMS,
    EXPECTED_CARDINALITIES,
    EXPECTED_OBSERVATION_SHA256,
    MAX_PAIR_RUNTIME_RATIO,
    N_PARTICLES,
    PG_FRACTION,
    REGIME,
    SEEDS,
    _controls,
    _protocol,
)


def test_new_development_observation_is_frozen() -> None:
    case = make_case(REGIME)
    metadata = _metadata(REGIME, case)["model_and_detections"]
    assert metadata["detections_sha256"] == EXPECTED_OBSERVATION_SHA256
    assert tuple(metadata["detection_cardinalities"]) == EXPECTED_CARDINALITIES
    protocol = _protocol()
    assert protocol["observation_sha256"] == EXPECTED_OBSERVATION_SHA256
    assert protocol["seeds"] == list(SEEDS) == [310, 311]
    assert protocol["arms"] == list(ARMS) == ["sparse_pg", "local"]
    assert protocol["n_particles"] == N_PARTICLES == 512
    assert protocol["max_pair_runtime_ratio"] == MAX_PAIR_RUNTIME_RATIO == 3


def test_sparse_pg_is_opt_in_and_local_control_unchanged() -> None:
    local = _controls("local")
    sparse = _controls("sparse_pg")
    assert local.paired_slot_scope == "local"
    assert local.paired_slot_pg_fraction == 1
    assert sparse.paired_slot_scope == "conditional_smc"
    assert sparse.paired_slot_pg_fraction == PG_FRACTION == 0.1
    assert sparse.paired_slot_path_particles == 8
    with pytest.raises(ValueError, match="unknown development arm"):
        _controls("unknown")


def test_development_artifact_keeps_frozen_protocol_and_status() -> None:
    result = json.loads(Path("results/wp11e_selective_pg_development.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["status"] in ("development_screen_passed", "stopped_before_holdout")
    assert set(result["baseline"]) == {str(seed) for seed in SEEDS}
    assert all(set(result["runs"][arm]) == {str(seed) for seed in SEEDS} for arm in ARMS)
    expected_status = (
        "development_screen_passed"
        if all(x["gate_passed"] for x in result["seed_comparisons"].values())
        else "stopped_before_holdout"
    )
    assert result["status"] == expected_status
