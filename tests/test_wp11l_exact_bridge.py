"""Frozen eight-frame bridge construction and passive macro-mode observer."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sbulm.existence_smc import sample_slot_existence_posterior
from sbulm.slot_existence import ABSENT
from sbulm.wp11g_exposed_wp8_high_particle import _candidate_controls
from sbulm.wp11l_exact_bridge import (
    CARDINALITIES,
    CLUTTER_RATES,
    OBSERVATION_SHA256,
    SUPPORT_SIZE,
    _BridgeObserver,
    _case,
    _counts,
    _metadata,
    _mode,
    _protocol,
    _run_order,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_bridge_metadata_and_schedule() -> None:
    metadata = _metadata()
    assert metadata["observation"]["detections_sha256"] == OBSERVATION_SHA256
    assert tuple(metadata["observation"]["detection_cardinalities"]) == CARDINALITIES
    assert _protocol()["support_size"] == SUPPORT_SIZE
    assert len(metadata["models"]) == len(CLUTTER_RATES) == 8
    assert [model["expected_clutter_per_frame"] for model in metadata["models"]] == list(
        CLUTTER_RATES
    )
    assert len(_run_order()) == 12
    assert _run_order()[0] == (512, 900)
    assert _run_order()[-1] == (2048, 905)


def test_macromodes_are_label_invariant_and_observer_passive() -> None:
    states = np.full((3, 8, 2), ABSENT, dtype=int)
    states[0, :8, 0] = -1
    states[0, 5, 0] = 0  # A: one long-lived slot, one clutter detection
    states[1, :8, 0] = -1
    states[1, 2:7, 1] = -1
    states[1, 5] = (0, 1)  # B: two overlapping slots, no frame-six clutter
    states[2, :8, 0] = -1
    states[2, 2:7, 1] = -1
    states[2, 5] = (0, -1)  # Other: high alive count with one clutter detection
    before = states.copy()
    np.testing.assert_array_equal(_mode(states), [0, 1, 2])
    np.testing.assert_array_equal(_mode(states[:, :, ::-1]), [0, 1, 2])
    alive, clutter = _counts(states)
    np.testing.assert_array_equal(alive, [8, 13, 13])
    np.testing.assert_array_equal(clutter, [1, 0, 1])
    observer = _BridgeObserver()
    observer.begin_stage(1, 1.0)
    observer.record_resampling(states, np.array([.2, .3, .5]), np.array([2, 1, 1]))
    observer.record_pair(states, states[[1, 0, 2]], np.array([True, True, False]),
                         np.array([True, False, True]))
    observer.end_stage(states)
    record = observer.records[0]
    assert record["accepted_A_to_B"] == {"pg": 1, "other": 0}
    assert record["accepted_B_to_A"] == {"pg": 0, "other": 1}
    assert np.isclose(sum(record["post_mode"]), 1)
    np.testing.assert_array_equal(states, before)


def test_observer_does_not_change_sampler_trajectory() -> None:
    case = _case(1.5)
    controls = replace(
        _candidate_controls(910), n_particles=16, mh_sweeps=1, final_mh_sweeps=1,
    )
    kwargs = {
        "n_slots": case.model.n_slots,
        "localization_std": case.localization_std,
        "coordinate_std": case.coordinate_std,
        "length_scale": case.length_scale,
        "detection_probability": case.model.detection_probability,
        "expected_clutter_per_frame": case.model.expected_clutter_per_frame,
        "roi_lower": case.model.roi_lower,
        "roi_upper": case.model.roi_upper,
        "birth_death_prior": case.model.birth_death_prior,
        "config": controls,
    }
    original = sample_slot_existence_posterior(case.model.sequence, **kwargs)
    observer = _BridgeObserver()
    traced = sample_slot_existence_posterior(case.model.sequence, **kwargs, observer=observer)
    np.testing.assert_array_equal(original.associations, traced.associations)
    np.testing.assert_array_equal(original.latent_paths, traced.latent_paths)
    assert original.stages == traced.stages
    assert original.log_evidence_estimate == traced.log_evidence_estimate
    assert len(observer.records) == traced.n_tempering_steps


def test_completed_bridge_checkpoint() -> None:
    result = json.loads((ROOT / "results/wp11l_exact_bridge.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    assert result["status"] == "absolute_bridge_audit_complete_no_move_selected"
    assert result["selected_rate"] == min(
        result["reference_grid"],
        key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]),
    )["clutter_rate"]
    assert min(result["reference"]["exact"]["mode"][:2]) >= 0.10
    assert np.isclose(sum(result["reference"]["exact"]["mode"]), 1)
    assert np.isclose(np.asarray(result["reference"]["exact"]["joint_alive_by_clutter"]).sum(), 1)
    assert sum(len(runs) for runs in result["runs"].values()) == len(_run_order()) == 12
    for particles, seed in _run_order():
        run = result["runs"][str(particles)][str(seed)]
        assert run["cpu_seconds"] < result["protocol"]["per_run_cpu_caps"][str(particles)]
        assert len(run["stage_mode_trace"]) == run["summary"]["n_tempering_steps"]
        np.testing.assert_allclose(
            run["stage_mode_trace"][-1]["post_mode"],
            np.asarray(result["reference"]["exact"]["mode"])
            + np.asarray([
                run["errors"]["mode_A_signed_error"],
                run["errors"]["mode_B_signed_error"],
                -run["errors"]["mode_A_signed_error"]
                - run["errors"]["mode_B_signed_error"],
            ]),
            atol=1e-12,
        )
    for n in (512, 2048):
        runs = result["runs"][str(n)].values()
        for metric, baseline in result["reference"]["iid_baseline"][str(n)].items():
            observed = sum(abs(run["errors"][metric]) > baseline["q95"]
                           for run in runs)
            assert result["comparison"][str(n)]["iid_q95_exceedances"][metric] == observed
