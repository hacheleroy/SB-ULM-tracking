"""Read-only diagnostic must preserve the frozen sampler's random trajectory."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from sbulm.existence_smc import sample_slot_existence_posterior
from sbulm.wp11g_exposed_wp8_high_particle import _candidate_controls, _frozen_inputs
from sbulm.wp11h_joint_mode_diagnostic import (
    FRAME_INDEX,
    JointModeObserver,
    _joint,
    _protocol,
    _reference,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_reference_and_read_only_observer() -> None:
    case, _, _ = _frozen_inputs(ROOT / "results/wp8_generalization.json")
    reference = _reference(ROOT / "results/wp11g_exposed_wp8_high_particle.json")
    assert reference["status"] == "failed_early_before_new_holdout"
    assert _protocol()["maximum_seed_reproductions"] == 2
    assert FRAME_INDEX == 5
    controls = replace(
        _candidate_controls(100), n_particles=16, mh_sweeps=1, final_mh_sweeps=1,
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
    observer = JointModeObserver(case.model.sequence.cardinalities[FRAME_INDEX])
    traced = sample_slot_existence_posterior(case.model.sequence, **kwargs, observer=observer)
    np.testing.assert_array_equal(original.associations, traced.associations)
    np.testing.assert_array_equal(original.latent_paths, traced.latent_paths)
    assert original.stages == traced.stages
    assert original.log_evidence_estimate == traced.log_evidence_estimate
    assert len(observer.stages) == traced.n_tempering_steps
    assert all(
        np.asarray(stage["joint_alive_total_by_frame_six_clutter"]).sum()
        == controls.n_particles for stage in observer.stages
    )
    assert np.array_equal(
        np.asarray(observer.stages[-1]["joint_alive_total_by_frame_six_clutter"]),
        _joint(traced.associations, case.model.sequence.cardinalities[FRAME_INDEX]),
    )
