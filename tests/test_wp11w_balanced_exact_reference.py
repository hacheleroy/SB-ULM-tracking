"""Check the frozen WP11w geometry and failed balance gate."""

import json
from pathlib import Path

import pytest

from sbulm.wp11w_balanced_exact_reference import (
    _accepted,
    _eligible,
    _validated_ledger,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_geometry_ledger_and_screen_outcome():
    ledger = _validated_ledger(ROOT / "results/wp11w_geometry_ledger.json")
    assert [x["index"] for x in _eligible(ledger)] == [0, 2, 4]
    assert "ROI" in ledger["candidates"][7]["generator_error"]
    result = json.loads((ROOT / "results/wp11w_balance_search.json").read_text())
    assert result["status"] == "frozen_ladder_exhausted_no_reference_no_sampler"
    assert [x["index"] for x in result["screens"]] == [0, 2, 4]
    for record in result["screens"]:
        assert record["status"] == "balance_failed_continue"
        assert not _accepted(record["masses"])
        assert record["numerical"]["minimum_raw_cell"] >= -1e-10
        assert record["numerical"]["maximum_relative_imaginary_residual"] <= 1e-8
        assert record["numerical"]["raw_mass_error"] <= 1e-8


def test_no_validation_without_passing_reference(tmp_path):
    with pytest.raises(ValueError, match="no candidate passed frozen balance rule"):
        validate(ROOT / "results/wp11w_geometry_ledger.json",
                 ROOT / "results/wp11w_screens", tmp_path / "reference.json")
    assert not (tmp_path / "reference.json").exists()
