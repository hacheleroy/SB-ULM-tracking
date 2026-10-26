"""Four-slot seven-detection exact bridge and GP factor lookup."""

import json
from pathlib import Path

import numpy as np

from sbulm.wp11q_seven_detection_exact_reference import (
    CANONICAL_SUPPORT_SIZE,
    CARDINALITIES,
    CLUTTER_RATES,
    LABELLED_SUPPORT_SIZE,
    OBSERVATION_HASH,
    SEEDS,
    _case,
    _gp_from_table,
    _gp_table,
    _metadata,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_four_slot_geometry_and_schedule() -> None:
    info = _metadata()
    assert info["observation"]["detections_sha256"] == OBSERVATION_HASH
    assert tuple(info["observation"]["detection_cardinalities"]) == CARDINALITIES
    assert len(info["models"]) == len(CLUTTER_RATES) == 11
    assert _case(1.0).model.n_slots == 4
    assert _protocol()["canonical_support_size"] == CANONICAL_SUPPORT_SIZE
    assert _protocol()["labelled_support_size"] == LABELLED_SUPPORT_SIZE
    assert _protocol()["seeds"] == list(SEEDS)
    assert _protocol()["reference_before_sampler"]


def test_slot_factor_lookup_matches_direct_collapsed_model() -> None:
    model = _case(.35).model
    table, strides = _gp_table(model)
    assert len(table) == 96
    states = model.sample_base_associations(
        n_samples=48, rng=np.random.default_rng(12345)
    )
    np.testing.assert_allclose(
        _gp_from_table(states, table, strides),
        model.log_marginal_density(states), atol=1e-11, rtol=0,
    )


def test_completed_absolute_checkpoint_and_frozen_flags() -> None:
    result = json.loads((ROOT / "results/wp11q_seven_detection_exact_reference.json").read_text())
    ref = result["reference"]
    assert result["protocol"] == _protocol()
    assert result["cases"] == _metadata()
    assert ref["difficulty_valid"]
    assert ref["canonical_support_size"] == CANONICAL_SUPPORT_SIZE
    assert ref["labelled_support_size"] == LABELLED_SUPPORT_SIZE
    assert ref["selected_rate"] == min(
        ref["reference_grid"],
        key=lambda row: (-min(row["mode"][:2]), row["clutter_rate"]),
    )["clutter_rate"]
    assert min(ref["exact"]["mode"][:2]) >= .10
    assert ref["tempered_mode_curve"][2]["mode"][1] <= .5 * ref["exact"]["mode"][1]
    assert result["status"] == "eight_frame_absolute_audit_no_flag"
    assert not any(result["diagnostic_flags"].values())
    for n in (512, 2048):
        runs = result["runs"][str(n)]
        assert tuple(map(int, runs)) == SEEDS
        exact_joint = np.asarray(ref["exact"]["joint_alive_by_clutter"])
        means = np.mean([run["joint_alive_by_clutter"] for run in runs.values()], axis=0)
        group = result["comparison"][str(n)]
        np.testing.assert_allclose(group["mean_signed_cell_error"], means - exact_joint)
        assert not group["familywise_flagged_cells"]
        assert all(run["cpu_seconds"] < _protocol()["run_cpu_caps"][str(n)]
                   for run in runs.values())
