"""Check the fixed reference and absolute joint-cell comparison."""

from pathlib import Path

import numpy as np

from sbulm.wp11g_exposed_wp8_high_particle import _candidate_controls
from sbulm.wp11y_independent_absolute_sampler import (
    N_PARTICLES,
    _bands,
    _controls,
    _decision,
    _inputs,
    _metrics,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_input_is_the_pinned_wp11x_case_and_controls_are_unchanged():
    case, exact, _ = _inputs(ROOT / "results/wp11x_frame_six_exact_reference.json",
                            ROOT / "results/wp11x_geometry_ledger.json")
    assert tuple(case.model.sequence.cardinalities) == (1, 2, 2, 2, 2, 3, 3, 2)
    assert exact.shape == (33, 4, 3)
    assert abs(exact.sum() - 1) < 1e-12
    assert _controls(1500).paired_slot_pg_fraction == _candidate_controls(
        1500
    ).paired_slot_pg_fraction
    assert _controls(1500).n_particles == N_PARTICLES
    assert _protocol()["adequacy_screen"]["at_least_runs_meeting_all_mode_limits"] == 3


def test_alive_band_and_decoy_cell_errors_have_fixed_axes():
    exact = np.zeros((33, 4, 3))
    exact[20, 0, 0], exact[21, 1, 1], exact[22, 1, 1] = .5, .2, .3
    observed = exact.copy()
    observed[20, 0, 0] -= .1
    observed[22, 1, 1] += .1
    errors = _metrics(observed, exact)
    np.testing.assert_allclose(_bands(exact).sum(axis=(1, 2)), [.5, .2, .3])
    np.testing.assert_allclose(errors["alive_band_mass_error"], [-.1, 0, .1])
    np.testing.assert_allclose(errors["sixth_clutter_mass_error"], [-.1, .1, 0, 0])
    np.testing.assert_allclose(errors["last_clutter_mass_error"], [-.1, .1, 0])
    assert np.isclose(errors["full_joint_tv"], .1)
    assert np.isclose(errors["band_by_sixth_tv"], .1)
    assert np.isclose(errors["maximum_joint_cell_error"], .1)


def test_replication_rule_can_fail_when_all_medians_pass():
    rows = {}
    for seed in (1500, 1501, 1502, 1503):
        rows[str(seed)] = {"errors": {
            "full_joint_tv": .09, "band_by_sixth_tv": .04,
            "maximum_alive_band_mass_error": .02,
            "maximum_joint_cell_error": .01,
            "sixth_clutter_mass_error": [0, .01, 0, 0],
            "last_clutter_mass_error": [0, .01, 0],
        }}
    rows["1500"]["errors"]["sixth_clutter_mass_error"][1] = .07
    rows["1501"]["errors"]["last_clutter_mass_error"][1] = .07
    result = _decision(rows)
    assert result["gates"]["median_full_joint"]
    assert not result["gates"]["replicated_modes"]
    assert not result["passed"]
