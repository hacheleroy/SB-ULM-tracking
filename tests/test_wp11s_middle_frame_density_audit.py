"""Independent support counts and frozen four-way middle-density diagnostics."""

import json
from itertools import combinations_with_replacement
from math import comb, factorial, prod
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sbulm.wp11q_seven_detection_exact_reference import _Accumulation, _gp_from_table, _gp_table
from sbulm.wp11s_middle_frame_density_audit import (
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
    _reweight_clutter,
    _run_order,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_middle_cases_and_independent_combinatorial_counts() -> None:
    info = _metadata()
    assert list(info) == list(OBSERVATION_ORDER)
    masks = _case("early", .05).model.interval_masks()
    for observation in OBSERVATION_ORDER:
        assert info[observation]["observation"]["detections_sha256"] == (
            OBSERVATION_HASHES[observation]
        )
        assert tuple(info[observation]["observation"]["detection_cardinalities"]) == (
            CARDINALITIES[observation]
        )
        assert len(info[observation]["models"]) == 11
        canonical = labelled = 0
        for indices in combinations_with_replacement(range(len(masks)), 4):
            active = masks[list(indices)].sum(axis=0)
            per_frame = [
                sum(comb(int(a), k) * factorial(n) // factorial(n - k)
                    for k in range(min(int(a), n) + 1))
                for a, n in zip(active, CARDINALITIES[observation], strict=True)
            ]
            count = prod(per_frame)
            canonical += count
            multiplicity = factorial(4)
            for index in set(indices):
                multiplicity //= factorial(indices.count(index))
            labelled += count * multiplicity
        assert (canonical, labelled) == (
            CANONICAL_SUPPORT[observation], LABELLED_SUPPORT[observation]
        )
    assert _protocol()["reference_before_sampler"]
    assert len(_run_order()) == 24
    assert _run_order()[0] == ("early", 512, 1180)
    assert _run_order()[-1] == ("middle", 2048, 1195)


def test_gp_factorization_for_both_middle_augmentations() -> None:
    for observation in OBSERVATION_ORDER:
        model = _case(observation, .35).model
        table, strides = _gp_table(model)
        assert len(table) == 432
        states = model.sample_base_associations(
            n_samples=32, rng=np.random.default_rng(1234)
        )
        np.testing.assert_allclose(
            _gp_from_table(states, table, strides),
            model.log_marginal_density(states), atol=1e-11, rtol=0,
        )


def test_rate_reweight_matches_direct_log_sum_at_both_endpoints() -> None:
    scores = np.array([-13., -6., -9., -3., -18., -5.])
    clutter = np.array([0, 2, 1, 2, 0, 1])
    cells = np.array([0, 1, 0, 2, 2, 1])
    acc = _Accumulation(9)
    acc.update(scores[:3], 3 * clutter[:3] + cells[:3])
    acc.update(scores[3:], 3 * clutter[3:] + cells[3:])
    for rate in (.05, 3.):
        probs, log_total = _reweight_clutter(acc, rate, 2, (3,))
        direct = scores + clutter * np.log(rate)
        np.testing.assert_allclose(log_total, logsumexp(direct), atol=1e-13)
        np.testing.assert_allclose(
            probs, np.bincount(cells, weights=np.exp(direct - log_total), minlength=3),
            atol=1e-14,
        )


def test_four_way_joint_exposes_middle_clutter_discrepancy() -> None:
    joint = np.zeros((33, 3, 3, 3))
    joint[12, 1, 1, 2] = .4
    joint[17, 0, 0, 0] = .6
    exact = _from_joint(joint)
    assert exact["mode"] == [.4, .6, 0.]
    changed = joint.copy()
    changed[12, 1, 1, 2] -= .1
    changed[12, 2, 1, 2] += .1
    errors = _errors(changed, exact)
    assert np.isclose(errors["full_4d_joint_tv"], .1)
    assert np.isclose(errors["collapsed_3d_joint_tv"], 0.)
    assert np.isclose(errors["middle_clutter_mean_error"], .1)

    states = np.full((2, 8, 4), -2)
    states[:, :, 0] = -1
    states[0, 2, 0], states[0, 5, 0], states[0, 7, 0] = 0, 0, 0
    states[1, :, 1] = -1
    states[1, 2, :2], states[1, 5, :2], states[1, 7, :2] = (0, 1), (0, 1), (0, 1)
    first = _counts(states, "early")
    second = _counts(states[:, :, ::-1], "early")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(_mode(np.array([15, 16]), np.array([1, 0])), [0, 1])


def test_completed_middle_density_checkpoint() -> None:
    path = ROOT / "results/wp11s_middle_frame_density_audit.json"
    if not path.exists():
        return  # Active after the frozen protocol's exact checkpoints.
    result = json.loads(path.read_text())
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    if result["status"] == "in_progress":
        return
    for observation in OBSERVATION_ORDER:
        ref = result["references"][observation]
        assert ref["canonical_support_size"] == CANONICAL_SUPPORT[observation]
        assert ref["labelled_support_size"] == LABELLED_SUPPORT[observation]
        assert ref["selected_rate"] == min(
            ref["reference_grid"],
            key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]),
        )["clutter_rate"]
        assert ref["cpu_seconds"] < _protocol()["reference_cpu_cap_each"]
        exact = np.asarray(ref["exact"]["joint_alive_by_middle_by_sixth_by_last_clutter"])
        assert exact.shape == (33, 3, 3, 3)
        assert np.isclose(exact.sum(), 1)
    if result["status"] == "middle_density_invalid_stop":
        assert not all(ref["difficulty_valid"] for ref in result["references"].values())
        return
    assert result["status"] == "middle_density_no_persistent_flag"
    assert result["diagnostic_flags"] == {
        "persistent_mode_tail": False,
        "persistent_joint_tail": False,
        "shared_cell_shift": False,
    }
    assert result["shared_signed_cells"] == []
    assert all(ref["difficulty_valid"] for ref in result["references"].values())
    for observation in OBSERVATION_ORDER:
        ref = result["references"][observation]
        truth = np.asarray(ref["exact"]["joint_alive_by_middle_by_sixth_by_last_clutter"])
        for n in (512, 2048):
            runs = result["runs"][observation][str(n)]
            assert tuple(map(int, runs)) == tuple(_protocol()["seeds"][observation])
            means = np.mean([run["joint_alive_by_middle_by_sixth_by_last_clutter"]
                             for run in runs.values()], axis=0)
            group = result["comparison"][observation][str(n)]
            np.testing.assert_allclose(group["mean_signed_4d_cell_error"], means - truth)
            np.testing.assert_allclose(
                group["mean_signed_collapsed_3d_cell_error"],
                (means - truth).sum(axis=1),
            )
            for key, baseline in ref["iid_baseline"][str(n)]["per_run"].items():
                count = sum(abs(run["errors"][key]) > baseline["q95"]
                            for run in runs.values())
                assert group["iid_q95_exceedances"][key] == count
            assert all(run["cpu_seconds"] < _protocol()["run_cpu_caps"][str(n)]
                       for run in runs.values())
