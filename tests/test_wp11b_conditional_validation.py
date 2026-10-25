"""Protocol regression for a reserved observation set and oracle-first gate."""

import json
from pathlib import Path

import pytest

from sbulm.wp7_convergence_audit import _case_metadata, _make_case
from sbulm.wp8_generalization import make_case
from sbulm.wp11b_conditional_validation import (
    ARMS,
    BRIDGE_HASH,
    ORACLE_SEEDS,
    RESERVED,
    RESERVED_HASH,
    RESERVED_SEEDS,
    _load_or_initialize,
    run_reserved,
)


def test_frozen_observations_and_blocked_reserved_inference(tmp_path: Path) -> None:
    assert _case_metadata(_make_case(bridge=True))["detections_sha256"] == BRIDGE_HASH
    assert _case_metadata(make_case(RESERVED))["detections_sha256"] == RESERVED_HASH
    result = _load_or_initialize(
        tmp_path / "new.json", Path("results/wp7_convergence_audit.json")
    )
    protocol = result["protocol"]
    assert protocol["arms_in_order"] == list(ARMS)
    assert protocol["oracle_seeds"] == list(ORACLE_SEEDS)
    assert protocol["reserved_seeds"] == list(RESERVED_SEEDS)
    assert protocol["seven_metric_gate"] == 0.10
    with pytest.raises(RuntimeError, match="oracle gate"):
        run_reserved(result, tmp_path / "new.json")
    (tmp_path / "new.json").write_text(json.dumps(result), encoding="utf-8")
    assert _load_or_initialize(
        tmp_path / "new.json", Path("results/wp7_convergence_audit.json")
    )["protocol"] == protocol
