"""Fresh geometry and numeric cross-check guards before exact screening."""

from pathlib import Path

import numpy as np
import pytest

from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp11ak_crowded_cohort_reference import (
    _checked_case,
    _cross_checked,
    _ledger,
    _ordered,
    _selection,
    screen,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/wp11ak_geometry_ledger.json"


def test_fresh_cohort_order_and_observation_hash() -> None:
    ledger = _ledger(LEDGER)
    assert len([base for base in ledger["bases"] if base["eligible"]]) == 19
    assert [_ordered(ledger, s)[0]["base_index"] for s in ("18", "19")] == [21, 124]
    for stratum in ("18", "19"):
        entry = _ordered(ledger, stratum)[0]
        case = _checked_case(entry)
        assert sum(case.model.sequence.cardinalities) == int(stratum)
        assert _case_metadata(case)["detections_sha256"] == entry["detection_sha256"]
    first, second = _ordered(ledger, "18")[:2]
    assert first["base_index"] != second["base_index"]


def test_numerical_budget_uses_paired_primary_and_direct_agreement() -> None:
    paired = {"joint": [0.4, 0.6], "log_partition": 1.0,
              "minimum_raw_cell": -1e-13, "clipping_tv": 1e-12,
              "maximum_relative_imaginary_residual": 0.0, "raw_mass_error": 0.0}
    direct = {**paired, "minimum_raw_cell": -1.3e-10,
              "clipping_tv": 4e-10, "log_partition": 1.0 + 8e-10,
              "joint": [0.4 + 1e-9, 0.6 - 1e-9]}
    assert _cross_checked(paired, direct)[0]
    assert not _cross_checked({**paired, "minimum_raw_cell": -2e-10}, direct)[0]
    assert not _cross_checked(paired, {**direct, "joint": [0.3, 0.7]})[0]


def test_no_skipped_screen_or_changed_ledger(tmp_path: Path) -> None:
    ledger = _ledger(LEDGER)
    first, second = _ordered(ledger, "18")[:2]
    next_entry, _ = _selection(ledger, tmp_path)
    assert next_entry["index"] == first["index"]
    with pytest.raises(ValueError, match="not the next frozen entry"):
        screen(LEDGER, tmp_path, second["index"])
    assert not list(tmp_path.iterdir())
    changed = tmp_path / "ledger.json"
    changed.write_bytes(LEDGER.read_bytes() + b" ")
    with pytest.raises(ValueError, match="unpinned or changed"):
        _ledger(changed)


def test_selected_observation_remains_within_original_frame_geometry() -> None:
    ledger = _ledger(LEDGER)
    entry = _ordered(ledger, "19")[0]
    case = _checked_case(entry)
    assert np.count_nonzero(case.model.sequence.reference_ids[5] == -1) == 1
    assert np.count_nonzero(case.model.sequence.reference_ids[7] == -1) == 1
