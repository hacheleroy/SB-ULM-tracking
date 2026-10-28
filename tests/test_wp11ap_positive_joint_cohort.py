"""Prospective geometry and numerical gate before any WP11ap posterior."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp11ap_positive_joint_cohort import (
    _checked_case,
    _ledger,
    _ordered,
    _positive_checked,
    _selection,
    manifest,
    screen,
)
from sbulm.wp11v_tempering_mode_audit import _ExactComputer, _toy_model

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/wp11ap_geometry_ledger.json"


def test_fresh_geometry_pinned_and_ordered(tmp_path: Path) -> None:
    ledger = _ledger(LEDGER)
    for stratum, count in (("18", 13), ("19", 15)):
        assert sum(base["eligible"] and sum(base["cardinalities"]) == int(stratum)
                   for base in ledger["bases"]) == count
        entry = _ordered(ledger, stratum)[0]
        assert _case_metadata(_checked_case(entry))["detections_sha256"] == (
            entry["detection_sha256"])
    first, _ = _selection(ledger, tmp_path)
    assert first == _ordered(ledger, "18")[0]
    with pytest.raises(ValueError, match="not the next frozen entry"):
        screen(LEDGER, tmp_path, _ordered(ledger, "18")[1]["index"])
    with pytest.raises(ValueError, match="incomplete cohort"):
        manifest(LEDGER, tmp_path, tmp_path, tmp_path / "manifest.json")
    assert not list(tmp_path.iterdir())
    changed = tmp_path / "ledger.json"
    changed.write_bytes(LEDGER.read_bytes() + b" ")
    with pytest.raises(ValueError, match="unpinned or changed"):
        _ledger(changed)


def test_full_joint_gate_detects_cell_error_with_unchanged_partition() -> None:
    computer = _ExactComputer(_toy_model(), (0, 2))
    paired = computer.compute(1.0, "paired")
    passed, _, good = _positive_checked(computer, 1.0, paired)
    assert passed
    broken = json.loads(json.dumps(good))
    cells = np.asarray(broken["joint"])
    largest = np.unravel_index(np.argmax(cells), cells.shape)
    other = (0, 0, 0) if largest != (0, 0, 0) else (1, 0, 0)
    cells[largest] -= 1e-4
    cells[other] += 1e-4
    broken["joint"] = cells.tolist()
    with patch("sbulm.wp11ap_positive_joint_cohort.positive_four_slot_joint",
               side_effect=[good, broken]):
        valid, comparison, _ = _positive_checked(computer, 1.0, paired)
    assert not valid
    assert comparison["paired_positive_logz_difference"] < 1e-8
    assert comparison["paired_positive_joint_tv"] > 1e-7
