"""Check the frozen WP11x observation and selected absolute reference."""

import json
from pathlib import Path

import numpy as np

from sbulm.wp8_generalization import make_case
from sbulm.wp11x_frame_six_decoy_reference import (
    _checked_case,
    _ledger,
    _regime,
    _screen_order,
    validated_reference,
)

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_revision_precedes_exact_screen_and_decoy_is_local():
    strict = json.loads((ROOT / "results/wp11x_strict_geometry_feasibility.json").read_text())
    assert sum(entry["eligible"] for entry in strict["bases"]) == 0
    ledger = _ledger(ROOT / "results/wp11x_geometry_ledger.json")
    ordered = _screen_order(ledger)
    assert len(ordered) == 9
    assert [(entry["base_index"], entry["offset_mm"]) for entry in ordered[:3]] == [
        (12, .010), (12, .016), (12, .022)
    ]
    entry = ordered[1]
    original = make_case(_regime(entry["base_index"]))
    selected = _checked_case(entry)
    index = entry["replaced_frame_six_index"]
    assert selected.model.sequence.reference_ids[5][index] == -1
    np.testing.assert_allclose(selected.model.sequence.detections[5][index, 0],
                               original.model.sequence.detections[5][index, 0])
    assert np.isclose(abs(selected.model.sequence.detections[5][index, 1]
                          - original.model.sequence.detections[5][index, 1]), .016)
    for frame in range(8):
        if frame != 5:
            np.testing.assert_array_equal(selected.model.sequence.detections[frame],
                                          original.model.sequence.detections[frame])


def test_first_passing_reference_is_pinned_and_numerically_valid():
    screens = [json.loads((ROOT / f"results/wp11x_screens/{i}.json").read_text())
               for i in (0, 1)]
    assert [x["status"] for x in screens] == [
        "balance_failed_continue", "balance_accepted_validate"
    ]
    result = validated_reference(ROOT / "results/wp11x_frame_six_exact_reference.json")
    assert result["variant_index"] == 1
    for beta in ("0.0", "0.5", "1.0"):
        check = result["selected_checks"][beta]
        assert check["paired_numerical"]["joint_tv"] < 1e-7
        assert check["paired_numerical"]["log_partition_absolute_difference"] < 1e-8
        assert check["direct"]["minimum_raw_cell"] >= -1e-10
    joint = np.asarray(result["selected_checks"]["1.0"]["direct"]["joint"])
    assert joint.shape == (33, 4, 3)
    np.testing.assert_allclose(joint.sum(), 1, atol=1e-12)
    assert joint[22:].sum() >= .18
    assert np.count_nonzero(joint.sum(axis=(0, 2)) >= .15) >= 2
