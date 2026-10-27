"""Regression checks for the independently frozen WP11v reference and observer."""

import hashlib
from pathlib import Path

import numpy as np

from sbulm.slot_existence import ABSENT
from sbulm.wp11v_tempering_mode_audit import (
    CASE_CARDINALITIES,
    REFERENCE_SHA256,
    _case,
    _StageObserver,
    _toy_validations,
    _validated_reference,
)


def test_selected_observation_and_reference_are_pinned():
    _, metadata = _case()
    assert metadata["ladder_index"] == 2
    assert tuple(metadata["model_and_detections"]["detection_cardinalities"]) == (
        CASE_CARDINALITIES
    )
    path = Path(__file__).resolve().parents[1] / "results/wp11v_tempering_reference.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    assert _validated_reference(path)["status"] == "validated_reference_before_sampler"


def test_tempered_reference_matches_separate_exhaustive_enumeration():
    results = _toy_validations()
    for result in results.values():
        assert result["log_partition_absolute_error"] < 1e-9
        assert result["joint_tv"] < 1e-9
        assert result["maximum_cell_absolute_error"] < 1e-10


def test_stage_observer_records_exact_pre_and_post_resampling_joints():
    cards = (1, 0, 0, 0, 0, 1, 0, 1)
    states = np.full((3, 8, 4), ABSENT, dtype=int)
    states[0, 0, 0] = 0
    states[0, 5, 0] = 0
    states[1, 7, 1] = 0
    observer = _StageObserver(cards)
    observer.begin_stage(1, .3)
    observer.record_resampling(states, np.array([.6, .3, .1]), np.array([0, 0, 1]))
    observer.end_stage(states[[0, 0, 1]])
    record = observer.records[0]
    assert np.asarray(record["prior_unweighted"])[2, 0, 1] == 1 / 3
    assert np.asarray(record["likelihood_weighted"])[2, 0, 1] == .6
    assert np.asarray(record["resampled"])[2, 0, 1] == 2 / 3
    assert record["resampled"] == record["rejuvenated"]
    for key in ("prior_unweighted", "likelihood_weighted", "resampled", "rejuvenated"):
        assert abs(np.asarray(record[key]).sum() - 1) < 1e-12
