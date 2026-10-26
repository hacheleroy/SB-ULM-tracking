"""Frozen four-slot references and passive joint-cell observer."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sbulm.data import DetectionSequence
from sbulm.existence_smc import sample_slot_existence_posterior
from sbulm.slot_existence import (
    ABSENT,
    SlotExistenceMaternModel,
    enumerate_slot_existence_support,
)
from sbulm.wp5_scaling_calibration import DEFAULT_BIRTH_DEATH_PRIOR
from sbulm.wp11g_exposed_wp8_high_particle import _candidate_controls
from sbulm.wp11n_dense_three_slot_bridge import _support_blocks
from sbulm.wp11o_four_slot_joint_reference import (
    CARDINALITIES,
    JOINT_TAIL_MIN_EXCEEDANCES,
    OBSERVATION_HASHES,
    OBSERVATION_ORDER,
    PARTICLES,
    SEEDS,
    SUPPORT_SIZE,
    _case,
    _counts,
    _errors_from_joint,
    _finish,
    _from_joint,
    _metadata,
    _mode,
    _ModeObserver,
    _protocol,
    _run_order,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cases_and_schedule() -> None:
    cases = _metadata()
    assert tuple(_protocol()["cardinalities"]) == CARDINALITIES
    assert _protocol()["support_size_each"] == SUPPORT_SIZE
    assert JOINT_TAIL_MIN_EXCEEDANCES == 2
    assert list(cases) == list(OBSERVATION_ORDER)
    for observation in OBSERVATION_ORDER:
        assert cases[observation]["observation"]["detections_sha256"] == (
            OBSERVATION_HASHES[observation]
        )
        assert len(cases[observation]["models"]) == 8
    assert len(_run_order()) == 24
    assert _run_order()[0] == ("primary", 512, 1000)
    assert _run_order()[-1] == ("replica", 2048, 1015)


def test_label_invariant_mode_and_signed_joint_cell_error() -> None:
    states = np.full((3, 5, 4), ABSENT, dtype=int)
    states[:, :, 0] = -1
    states[0, 4, 0] = 0
    states[1:, :, 1] = -1
    states[1, 4, :2] = (0, 1)
    states[2, 4, 0] = 0
    original = states.copy()
    alive, clutter = _counts(states)
    np.testing.assert_array_equal(alive, [5, 10, 10])
    np.testing.assert_array_equal(clutter, [1, 0, 1])
    np.testing.assert_array_equal(_mode(alive, clutter), [0, 1, 2])
    np.testing.assert_array_equal(_mode(*_counts(states[:, :, ::-1])), [0, 1, 2])
    np.testing.assert_array_equal(states, original)

    exact = np.zeros((21, 3))
    exact[5, 1], exact[10, 0] = .5, .5
    observed = exact.copy()
    observed[5, 1] -= .1
    observed[5, 0] += .1
    errors = _errors_from_joint(observed, _from_joint(exact))
    assert np.isclose(errors["full_joint_tv"], .1)
    assert np.isclose(errors["maximum_cell_absolute_error"], .1)
    assert np.isclose(errors["mode_A_signed_error"], -.1)
    assert np.isclose(errors["mode_B_signed_error"], 0)


def test_streamed_support_matches_complete_small_enumeration() -> None:
    sequence = DetectionSequence(
        times=np.arange(3) * .001,
        detections=(np.array([[0.0, 0.0]]), np.empty((0, 2)),
                    np.array([[.002, 0.0]])),
    )
    model = SlotExistenceMaternModel(
        sequence, n_slots=2, localization_std=.005,
        coordinate_std=(.02, .01), length_scale=.003,
        detection_probability=.55, expected_clutter_per_frame=1.0,
        roi_lower=(-.04, -.04), roi_upper=(.04, .04),
        birth_death_prior=DEFAULT_BIRTH_DEATH_PRIOR,
    )
    exact, _ = enumerate_slot_existence_support(model)
    streamed = np.concatenate(list(_support_blocks(model)))
    np.testing.assert_array_equal(streamed, exact)


def test_observer_does_not_change_sampler_trajectory() -> None:
    case = _case("primary", .35)
    controls = replace(
        _candidate_controls(1020), n_particles=16, mh_sweeps=1, final_mh_sweeps=1,
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
    observer = _ModeObserver()
    traced = sample_slot_existence_posterior(case.model.sequence, **kwargs, observer=observer)
    np.testing.assert_array_equal(original.associations, traced.associations)
    np.testing.assert_array_equal(original.latent_paths, traced.latent_paths)
    assert original.stages == traced.stages
    assert original.log_evidence_estimate == traced.log_evidence_estimate
    assert len(observer.records) == traced.n_tempering_steps


def test_replication_requires_both_cases_and_same_signed_cell() -> None:
    exact = np.zeros((21, 3))
    exact[5, 0], exact[5, 1] = .5, .5
    per_run = {key: {"q95": .02} for key in (
        "maximum_mode_absolute_error", "full_joint_tv",
        "maximum_cell_absolute_error", "alive_total_tv", "clutter_mean_error",
    )}

    def fabricated(replica_sign: int):
        result = {"references": {}, "runs": {}}
        for observation in OBSERVATION_ORDER:
            result["references"][observation] = {
                "exact": _from_joint(exact),
                "iid_baseline": {
                    str(n): {"per_run": per_run, "group_max_cell_mean": {"q95": .01}}
                    for n in PARTICLES
                },
            }
            result["runs"][observation] = {}
            for n in PARTICLES:
                result["runs"][observation][str(n)] = {}
                for seed in SEEDS[observation]:
                    joint = exact.copy()
                    if n == 2048:
                        sign = replica_sign if observation == "replica" else 1
                        joint[5, 0] += .03 * sign
                        joint[5, 1] -= .03 * sign
                    errors = _errors_from_joint(joint, _from_joint(exact))
                    errors["log_evidence_error"] = 0.
                    result["runs"][observation][str(n)][str(seed)] = {
                        "joint_alive_by_clutter": joint.tolist(),
                        "errors": errors, "cpu_seconds": 1.,
                    }
        return result

    same = fabricated(1)
    _finish(same)
    assert same["replication"]["replicated_joint_tail"]
    assert same["replication"]["replicated_cell_shift"]
    assert len(same["replication"]["shared_signed_cells"]) == 2

    opposite = fabricated(-1)
    _finish(opposite)
    assert opposite["replication"]["replicated_joint_tail"]
    assert not opposite["replication"]["replicated_cell_shift"]


def test_completed_four_slot_checkpoint() -> None:
    result = json.loads((ROOT / "results/wp11o_four_slot_joint_reference.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    assert result["status"] == "four_slot_joint_audit_complete_no_replicated_signal"
    assert result["replication"]["replicated_joint_tail"] is False
    assert result["replication"]["replicated_cell_shift"] is False
    assert sum(len(group) for by_size in result["runs"].values()
               for group in by_size.values()) == len(_run_order()) == 24
    for observation in OBSERVATION_ORDER:
        ref = result["references"][observation]
        assert ref["support_size"] == SUPPORT_SIZE
        assert ref["selected_rate"] == min(
            ref["reference_grid"],
            key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]),
        )["clutter_rate"]
        assert ref["mode_competition_at_least_0_10"]
        assert ref["cpu_seconds"] < result["protocol"]["reference_cpu_cap_each"]
        exact = np.asarray(ref["exact"]["joint_alive_by_clutter"])
        assert np.isclose(exact.sum(), 1)
        for n in PARTICLES:
            runs = result["runs"][observation][str(n)]
            baseline = ref["iid_baseline"][str(n)]
            mean = np.mean([run["joint_alive_by_clutter"] for run in runs.values()], axis=0)
            group = result["comparison"][observation][str(n)]
            assert np.isclose(group["max_absolute_mean_cell_error"],
                              np.max(np.abs(mean - exact)))
            for metric, stats in baseline["per_run"].items():
                observed = sum(abs(run["errors"][metric]) > stats["q95"]
                               for run in runs.values())
                assert group["iid_q95_exceedances"][metric] == observed
    for observation, n, seed in _run_order():
        run = result["runs"][observation][str(n)][str(seed)]
        assert run["cpu_seconds"] < result["protocol"]["per_run_cpu_caps"][str(n)]
        assert len(run["stage_mode_trace"]) == run["summary"]["n_tempering_steps"]
        np.testing.assert_allclose(
            run["stage_mode_trace"][-1]["post_mode"],
            _from_joint(np.asarray(run["joint_alive_by_clutter"]))["mode"],
            atol=1e-11,
        )
