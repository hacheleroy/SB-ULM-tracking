"""Frozen geometry and distinct numerical checks for WP11an."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sbulm.wp11an_positive_crowded_cohort import (
    _checked_case,
    _joint_checked,
    _ledger,
    _ordered,
    _partition_checked,
    _selection,
    manifest,
)

from sbulm.wp7_convergence_audit import _case_metadata

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/wp11an_geometry_ledger.json"


def test_fresh_geometry_is_pinned_and_next_screen_is_ordered(tmp_path: Path) -> None:
    ledger = _ledger(LEDGER)
    for stratum, count in (("18", 7), ("19", 10)):
        assert sum(base["eligible"] and sum(base["cardinalities"]) == int(stratum)
                   for base in ledger["bases"]) == count
        first = _ordered(ledger, stratum)[0]
        assert _case_metadata(_checked_case(first))["detections_sha256"] == (
            first["detection_sha256"])
    first, selected = _selection(ledger, tmp_path)
    assert first == _ordered(ledger, "18")[0]
    assert selected == {"18": [], "19": []}


def test_partition_gate_checks_absolute_agreement_and_precision() -> None:
    paired = {"log_partition": 100., "minimum_raw_cell": -1e-12,
              "clipping_tv": 1e-12,
              "maximum_relative_imaginary_residual": 0., "raw_mass_error": 0.}
    outputs = [{"log_partition": 100. + 1e-11},
               {"log_partition": 100. + 2e-11}]
    with patch("sbulm.wp11an_positive_crowded_cohort.positive_four_slot_partition",
               side_effect=outputs):
        assert _partition_checked(None, 1., paired)[0]
    outputs = [{"log_partition": 100. + 1e-11},
               {"log_partition": 100. + 2e-8}]
    with patch("sbulm.wp11an_positive_crowded_cohort.positive_four_slot_partition",
               side_effect=outputs):
        assert not _partition_checked(None, 1., paired)[0]


def test_direct_partition_discrepancy_is_diagnostic_but_joint_is_checked() -> None:
    paired = {"joint": np.array([.4, .6]), "log_partition": 100.}
    direct = {"joint": np.array([.4 + 1e-9, .6 - 1e-9]),
              "log_partition": 100. + 4.5e-8,
              "minimum_raw_cell": -1e-9, "clipping_tv": 1e-8,
              "maximum_relative_imaginary_residual": 0., "raw_mass_error": 0.}
    passed, comparison = _joint_checked(paired, direct)
    assert passed and comparison["diagnostic_direct_paired_logz_difference"] > 1e-8
    assert not _joint_checked(paired, {**direct, "joint": np.array([.3, .7])})[0]


def test_failed_validation_cannot_produce_complete_manifest(tmp_path: Path) -> None:
    runs = ROOT / "results/wp11an_screens"
    references = ROOT / "results/wp11an_references"
    diagnostic = json.loads((ROOT / "results/wp11an_terminal_validation_failure.json")
                            .read_text())
    assert diagnostic["partition_pass"] and not diagnostic["joint_pass"]
    assert diagnostic["joint"]["joint_tv"] > 1e-7
    assert json.loads((references / "84.json").read_text())["status"] == (
        "reference_in_progress")
    try:
        manifest(LEDGER, runs, references, tmp_path / "manifest.json")
    except ValueError as exc:
        assert "reference is incomplete" in str(exc)
    else:
        raise AssertionError("failed validation produced a complete manifest")
