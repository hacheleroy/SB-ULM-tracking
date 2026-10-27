"""Prospective geometry and ordered-screen integrity for the crowded cohort."""

from pathlib import Path

import numpy as np
import pytest

from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp8_generalization import make_case
from sbulm.wp11aj_crowded_cohort_reference import (
    _checked_case,
    _ledger,
    _ordered,
    _regime,
    _selection,
    screen,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/wp11aj_geometry_ledger.json"


def test_pinned_18_and_19_geometry_and_distinct_decoy_frames() -> None:
    ledger = _ledger(LEDGER)
    assert sum(base["eligible"] for base in ledger["bases"]) == 10
    assert [_ordered(ledger, stratum)[0]["base_index"]
            for stratum in ("18", "19")] == [173, 48]
    for stratum in ("18", "19"):
        entry = _ordered(ledger, stratum)[0]
        base = make_case(_regime(entry["base_index"]))
        case = _checked_case(entry)
        assert _case_metadata(case)["detections_sha256"] == entry["detection_sha256"]
        assert sum(case.model.sequence.cardinalities) == int(stratum)
        for frame in range(8):
            if frame not in (5, 7):
                np.testing.assert_array_equal(case.model.sequence.detections[frame],
                                              base.model.sequence.detections[frame])
        for frame, name in ((5, "sixth"), (7, "last")):
            index = entry["decoys"][name]["index"]
            assert case.model.sequence.reference_ids[frame][index] == -1
            assert case.model.sequence.detections[frame][index, 1] != (
                base.model.sequence.detections[frame][index, 1]
            )


def test_out_of_order_screen_and_changed_geometry_are_rejected(tmp_path: Path) -> None:
    ledger = _ledger(LEDGER)
    first, second = _ordered(ledger, "18")[:2]
    next_entry, selected = _selection(ledger, tmp_path)
    assert next_entry["index"] == first["index"]
    assert selected == {"18": [], "19": []}
    with pytest.raises(ValueError, match="not the next frozen entry"):
        screen(LEDGER, tmp_path, second["index"])
    assert list(tmp_path.iterdir()) == []
    changed = tmp_path / "changed_ledger.json"
    changed.write_bytes(LEDGER.read_bytes() + b" ")
    with pytest.raises(ValueError, match="unpinned or changed"):
        _ledger(changed)
