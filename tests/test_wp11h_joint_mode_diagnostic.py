"""Read-only diagnostic must preserve the frozen sampler's random trajectory."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from sbulm.existence_smc import sample_slot_existence_posterior
from sbulm.slot_existence import ABSENT
from sbulm.wp11g_exposed_wp8_high_particle import _candidate_controls, _frozen_inputs
from sbulm.wp11h_joint_mode_diagnostic import (
    FRAME_INDEX,
    JointModeObserver,
    _canonical_associations,
    _clutter_mask,
    _decode_associations,
    _decode_signatures,
    _diagnosis,
    _encode_associations,
    _encode_signatures,
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
    observer = JointModeObserver(
        case.model.sequence.cardinalities[FRAME_INDEX], case.model.sequence.n_frames
    )
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
    signatures = _decode_signatures(_encode_signatures(traced.associations))
    assert signatures.shape == (controls.n_particles, 2 * case.model.sequence.n_frames)
    np.testing.assert_array_equal(
        signatures[:, :case.model.sequence.n_frames],
        np.count_nonzero(traced.associations != ABSENT, axis=2),
    )
    decoded = _decode_associations(_encode_associations(traced.associations))
    np.testing.assert_array_equal(decoded, traced.associations)
    np.testing.assert_array_equal(
        _canonical_associations(decoded),
        _canonical_associations(decoded[:, :, ::-1]),
    )
    clutter_mask = _clutter_mask(decoded, case.model.sequence.cardinalities[FRAME_INDEX])
    assert all(int(mask).bit_count() == case.model.sequence.cardinalities[FRAME_INDEX]
               - int(np.count_nonzero(row[FRAME_INDEX] >= 0))
               for row, mask in zip(decoded, clutter_mask, strict=True))
    record = {
        "stages": observer.stages,
        "final_count_signatures": _encode_signatures(traced.associations),
        "final_associations": _encode_associations(traced.associations),
    }
    diagnosis = _diagnosis({"100": record, "200": record})
    assert all(stage["joint_tv"] == 0 for stage in diagnosis["stage_differences"])
    assert diagnosis["signature_comparisons"]["complete_count_signature"]["descriptive_tv"] == 0
