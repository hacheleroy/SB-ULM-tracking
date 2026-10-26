"""Squarefree subset identity and independent ten-case absolute validation."""

import json
from pathlib import Path

import numpy as np

from sbulm.wp11t_subset_exact_primary import (
    PRIMARY_CARDINALITIES,
    PRIMARY_HASH,
    SEEDS,
    _case,
    _errors,
    _exact_joint,
    _fourth_power,
    _metadata,
    _protocol,
    _summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_primary_geometry_and_reference_before_sampler() -> None:
    info = _metadata()["model_and_detections"]
    assert info["detections_sha256"] == PRIMARY_HASH
    assert tuple(info["detection_cardinalities"]) == PRIMARY_CARDINALITIES
    assert _case().model.n_slots == 4
    protocol = _protocol()
    assert protocol["reference_before_new_sampler"]
    assert protocol["seeds"] == list(SEEDS)
    assert protocol["numerical_validation"]["stop_sampler_if_any_fail"]


def test_squarefree_fourth_power_matches_independent_disjoint_enumeration() -> None:
    f = np.array([.31, .13, .19, .08, .17, .04, .03, .05,
                  .10, .02, .01, .04, .03, .02, .01, .01], dtype=np.longdouble)
    counts = np.array([s.bit_count() for s in range(16)], dtype=np.int8)
    expected = np.zeros(16, dtype=np.longdouble)
    for a in range(16):
        for b in range(16):
            if a & b:
                continue
            for c in range(16):
                if (a | b) & c:
                    continue
                for d in range(16):
                    if (a | b | c) & d:
                        continue
                    expected[a | b | c | d] += f[a] * f[b] * f[c] * f[d]
    for method in ("direct", "paired"):
        actual = _fourth_power(f, 4, counts, method)
        np.testing.assert_allclose(actual, expected, atol=1e-17, rtol=0)


def test_subsets_reproduce_both_independent_ten_detection_exact_tables() -> None:
    from sbulm.wp11s_middle_frame_density_audit import _case as ten_case

    saved = json.loads((ROOT / "results/wp11s_middle_frame_density_audit.json").read_text())
    for name, frame in (("early", 2), ("middle", 4)):
        direct = _exact_joint(ten_case(name, 1.5).model, frame, "direct")
        reference = saved["references"][name]
        actual = np.asarray(direct["joint"])
        exact = np.asarray(reference["exact"]["joint_alive_by_middle_by_sixth_by_last_clutter"])
        assert abs(direct["log_evidence"] - reference["log_evidence"]) < 1e-9
        assert np.abs(actual - exact).sum() / 2 < 1e-9
        assert np.max(np.abs(actual - exact)) < 1e-10


def test_band_and_full_joint_diagnostics_preserve_final_clutter() -> None:
    joint = np.zeros((33, 3, 2))
    joint[12, 1, 0], joint[21, 0, 1], joint[25, 2, 1] = .2, .3, .5
    exact = _summary(joint)
    assert exact["band_mass"] == [.2, .3, .5]
    changed = joint.copy()
    changed[12, 1, 0] -= .1
    changed[12, 1, 1] += .1
    errors = _errors(changed, exact)
    assert np.isclose(errors["full_3d_joint_tv"], .1)
    assert np.isclose(errors["band_by_sixth_joint_tv"], 0)
    assert np.isclose(errors["last_clutter_mean_error"], .1)


def test_completed_primary_checkpoint_when_available() -> None:
    path = ROOT / "results/wp11t_subset_exact_primary.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    assert result["protocol"] == _protocol()
    assert result["case"] == _metadata()
    if result["reference"] is None:
        return
    if not result["reference"]["valid"]:
        assert result["status"] == "primary_numerical_validation_failed_stop"
        return
    joint = np.asarray(result["reference"]["joint_alive_by_sixth_by_last_clutter"])
    assert joint.shape == (33, 3, 2)
    assert np.isclose(joint.sum(), 1, atol=1e-8)
    assert set(result["validations"]) == {"early", "middle"}
    if result["status"] == "validated_reference_before_sampler":
        return
    assert set(result["runs"]) == {str(seed) for seed in SEEDS}
    mean = np.mean([r["joint_alive_by_sixth_by_last_clutter"]
                    for r in result["runs"].values()], axis=0)
    np.testing.assert_allclose(result["comparison"]["mean_signed_3d_cell_error"],
                               mean - joint)
    assert all(r["cpu_seconds"] < _protocol()["run_cpu_cap"]
               for r in result["runs"].values())
