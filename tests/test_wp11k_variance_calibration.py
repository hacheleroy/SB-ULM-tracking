"""Frozen inputs and passive observer for the WP11k diagnostic."""

import json
from pathlib import Path

import numpy as np

from sbulm.slot_existence import ABSENT
from sbulm.wp11k_variance_calibration import (
    BANK_CARDINALITIES,
    BANK_HASHES,
    EXACT_HASH,
    EXACT_SUPPORT_SIZE,
    _exact_case,
    _JointObserver,
    _metadata_for_cases,
    _protocol,
    _run_order,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cases_and_order() -> None:
    metadata = _metadata_for_cases()
    assert metadata["exact"]["detections_sha256"] == EXACT_HASH
    assert _protocol()["exact_case"]["support_size"] == EXACT_SUPPORT_SIZE
    for name, expected in BANK_HASHES.items():
        info = metadata["bank"][name]["model_and_detections"]
        assert info["detections_sha256"] == expected
        assert tuple(info["detection_cardinalities"]) == BANK_CARDINALITIES[name]
    assert len(_run_order()) == 24
    assert _run_order()[0] == ("exact", 512, 820)
    assert _run_order()[-1] == ("wp11k_stress", 512, 813)


def test_observer_joint_distribution_and_resampling() -> None:
    observer = _JointObserver(_exact_case())
    states = np.full((3, 4, 3), ABSENT, dtype=int)
    states[0, -1, 0] = 0
    states[1, -1, 1] = -1
    states[2, :, 0] = -1
    observer.begin_stage(0, 1.0)
    observer.record_resampling(states, np.array([.5, .3, .2]), np.array([0, 0, 2]))
    observer.end_stage(states)
    record = observer.records[0]
    assert np.isclose(np.asarray(record["post_joint"]).sum(), 1)
    assert np.isclose(record["low_alive_band_mass"], 1)
    assert np.isclose(record["expected_clutter_at_frame"], 2 / 3)
    assert 0 <= record["weighted_to_resampled_joint_tv"] <= 1


def test_completed_checkpoint_matches_frozen_protocol_and_comparisons() -> None:
    result = json.loads((ROOT / "results/wp11k_variance_calibration.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata_for_cases()
    assert result["status"] == "audit_completed_no_candidate_gate"
    assert sum(len(seeds) for groups in result["runs"].values()
               for seeds in groups.values()) == len(_run_order()) == 24
    for case, particles, seed in _run_order():
        record = result["runs"][case][str(particles)][str(seed)]
        assert record["cpu_seconds"] < result["protocol"]["per_run_cpu_caps"][case][str(particles)]
        assert len(record["joint_trace"]) == record["summary"]["n_tempering_steps"]
        assert np.isclose(np.asarray(record["joint_trace"][-1]["post_joint"]).sum(), 1)
    reference = result["reference"]
    assert reference["support_size"] == EXACT_SUPPORT_SIZE
    assert np.isclose(sum(reference["alive_total_histogram"]), 1)
    assert np.isclose(sum(reference["count_signature_probabilities"]), 1)
    for particles in (512, 2048):
        runs = result["runs"]["exact"][str(particles)].values()
        for metric in ("alive_total_tv", "maximum_clutter_error", "count_signature_tv"):
            exceedances = sum(run["errors"][metric]
                              > reference["iid_baseline"][str(particles)][metric]["q95"]
                              for run in runs)
            assert (result["comparisons"]["exact"][str(particles)]
                    ["iid_q95_exceedances"][metric] == exceedances)
