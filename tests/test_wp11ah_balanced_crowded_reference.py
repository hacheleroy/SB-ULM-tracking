"""Prospective geometry, integrity and ordered-screen guards."""

from pathlib import Path

import numpy as np
import pytest

from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp8_generalization import make_case
from sbulm.wp11ah_balanced_crowded_reference import (
    _checked_case,
    _ledger,
    _ordered,
    _regime,
    screen,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/wp11ah_geometry_ledger.json"


def test_pinned_unused_geometry_and_two_frame_transform() -> None:
    ledger = _ledger(LEDGER)
    ordered = _ordered(ledger)
    assert len(ordered) == 18
    assert [entry["base_index"] for entry in ordered[:5]] == [86, 97, 13, 28, 74]
    first = ordered[0]
    base = make_case(_regime(first["base_index"]))
    case = _checked_case(first)
    assert _case_metadata(case)["detections_sha256"] == first["detection_sha256"]
    assert case.model.sequence.cardinalities == base.model.sequence.cardinalities
    for frame in range(8):
        if frame not in (5, 7):
            np.testing.assert_array_equal(case.model.sequence.detections[frame],
                                          base.model.sequence.detections[frame])
            np.testing.assert_array_equal(case.model.sequence.reference_ids[frame],
                                          base.model.sequence.reference_ids[frame])
    for frame, name in ((5, "sixth"), (7, "last")):
        index = first["decoys"][name]["index"]
        assert case.model.sequence.reference_ids[frame][index] == -1
        assert case.model.sequence.detections[frame][index, 1] != (
            base.model.sequence.detections[frame][index, 1]
        )


def test_future_screen_cannot_skip_frozen_order(tmp_path: Path) -> None:
    second = _ordered(_ledger(LEDGER))[1]
    with pytest.raises(ValueError, match="earlier screen missing"):
        screen(LEDGER, tmp_path, second["index"])
