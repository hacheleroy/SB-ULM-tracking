"""The future sampler gate binds to the independently frozen reference."""

from pathlib import Path

import pytest

from sbulm.wp11ai_balanced_crowded_comparison import (
    REFERENCE_SHA256,
    SEEDS,
    _inputs,
    _protocol,
    run_one,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (ROOT / "results/wp11ah_geometry_ledger.json",
          ROOT / "results/wp11ah_screens/27.json",
          ROOT / "results/wp11ah_crowded_exact_reference.json",
          ROOT / "results/wp11ag_global_slot_comparison.json")


def test_pinned_balanced_absolute_joint_and_matched_schedule() -> None:
    case, joint, _ = _inputs(*INPUTS)
    assert tuple(case.model.sequence.cardinalities) == (3, 2, 2, 3, 1, 2, 2, 2)
    assert joint.shape == (33, 3, 3)
    assert _protocol()["exact_reference_sha256"] == REFERENCE_SHA256
    assert _protocol()["seeds"] == list(SEEDS)
    assert _protocol()["accuracy_and_replication"]["minimum_runs_meeting_mode_limits"] == 3


def test_changed_reference_and_unscheduled_seed_are_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "reference.json"
    changed.write_bytes(INPUTS[2].read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen input changed"):
        _inputs(INPUTS[0], INPUTS[1], changed, INPUTS[3])
    with pytest.raises(ValueError, match="outside frozen arms and seeds"):
        run_one("candidate", 1704, tmp_path, *INPUTS)
    assert list(tmp_path.glob("balanced_17_*.json")) == []
