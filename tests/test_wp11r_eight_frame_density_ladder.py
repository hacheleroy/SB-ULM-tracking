"""Frozen two-density exact reference and three-dimensional joint summaries."""

import json
from pathlib import Path

import numpy as np

from sbulm.wp11q_seven_detection_exact_reference import _gp_from_table, _gp_table
from sbulm.wp11r_eight_frame_density_ladder import (
    CANONICAL_SUPPORT,
    CARDINALITIES,
    LABELLED_SUPPORT,
    OBSERVATION_HASHES,
    OBSERVATION_ORDER,
    _case,
    _counts,
    _errors,
    _from_joint,
    _metadata,
    _mode,
    _protocol,
    _run_order,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_density_cases_and_run_order() -> None:
    info = _metadata()
    assert list(info) == list(OBSERVATION_ORDER)
    for observation in OBSERVATION_ORDER:
        assert info[observation]["observation"]["detections_sha256"] == (
            OBSERVATION_HASHES[observation]
        )
        assert tuple(info[observation]["observation"]["detection_cardinalities"]) == (
            CARDINALITIES[observation]
        )
        assert len(info[observation]["models"]) == 11
        assert CANONICAL_SUPPORT[observation] < LABELLED_SUPPORT[observation]
    assert _protocol()["reference_before_sampler"]
    assert len(_run_order()) == 24
    assert _run_order()[0] == ("eight", 512, 1160)
    assert _run_order()[-1] == ("nine", 2048, 1175)


def test_gp_factorization_on_both_late_observations() -> None:
    for observation in OBSERVATION_ORDER:
        model = _case(observation, .35).model
        table, strides = _gp_table(model)
        assert len(table) == 96 * (CARDINALITIES[observation][-1] + 1)
        states = model.sample_base_associations(
            n_samples=32, rng=np.random.default_rng(1234)
        )
        np.testing.assert_allclose(
            _gp_from_table(states, table, strides),
            model.log_marginal_density(states), atol=1e-11, rtol=0,
        )


def test_full_joint_retains_both_clutter_frames_and_modes() -> None:
    joint = np.zeros((33, 3, 3))
    joint[12, 1, 2] = .4
    joint[17, 0, 0] = .6
    exact = _from_joint(joint)
    assert exact["mode"] == [.4, .6, 0.]
    assert exact["last_clutter"] == [.6, 0., .4]
    changed = joint.copy()
    changed[12, 1, 2] -= .1
    changed[12, 1, 1] += .1
    errors = _errors(changed, exact)
    assert np.isclose(errors["full_3d_joint_tv"], .1)
    assert np.isclose(errors["collapsed_2d_joint_tv"], 0.)
    assert np.isclose(errors["last_clutter_mean_error"], -.1)

    states = np.full((2, 8, 4), -2)
    states[:, :, 0] = -1
    states[0, 5, 0], states[0, 7, 0] = 0, 0
    states[1, :, 1] = -1
    states[1, 5, :2], states[1, 7, :2] = (0, 1), (0, 1)
    first = _counts(states, "nine")
    second = _counts(states[:, :, ::-1], "nine")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(_mode(np.array([15, 16]), np.array([1, 0])), [0, 1])


def test_completed_density_ladder_checkpoint() -> None:
    result = json.loads((ROOT / "results/wp11r_eight_frame_density_ladder.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    assert result["status"] == "density_ladder_no_persistent_flag"
    assert not any(result["diagnostic_flags"].values())
    assert result["shared_signed_cells"] == []
    for observation in OBSERVATION_ORDER:
        ref = result["references"][observation]
        assert ref["canonical_support_size"] == CANONICAL_SUPPORT[observation]
        assert ref["labelled_support_size"] == LABELLED_SUPPORT[observation]
        assert ref["difficulty_valid"]
        assert ref["selected_rate"] == min(
            ref["reference_grid"],
            key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]),
        )["clutter_rate"]
        assert min(ref["exact"]["mode"][:2]) >= .10
        assert ref["tempered_mode_curve"][2]["mode"][1] <= .5 * ref["exact"]["mode"][1]
        assert ref["cpu_seconds"] < _protocol()["reference_cpu_cap_each"]
        exact = np.asarray(ref["exact"]["joint_alive_by_sixth_by_last_clutter"])
        assert exact.shape == (33, 3, CARDINALITIES[observation][-1] + 1)
        assert np.isclose(exact.sum(), 1)
        for n in (512, 2048):
            runs = result["runs"][observation][str(n)]
            assert tuple(map(int, runs)) == tuple(_protocol()["seeds"][observation])
            means = np.mean([run["joint_alive_by_sixth_by_last_clutter"]
                             for run in runs.values()], axis=0)
            group = result["comparison"][observation][str(n)]
            np.testing.assert_allclose(group["mean_signed_3d_cell_error"], means - exact)
            assert not group["familywise_flagged_cells"]
            for key, baseline in ref["iid_baseline"][str(n)]["per_run"].items():
                count = sum(abs(run["errors"][key]) > baseline["q95"]
                            for run in runs.values())
                assert group["iid_q95_exceedances"][key] == count
            assert all(run["cpu_seconds"] < _protocol()["run_cpu_caps"][str(n)]
                       for run in runs.values())
