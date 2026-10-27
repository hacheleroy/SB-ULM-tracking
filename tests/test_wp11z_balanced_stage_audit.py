"""Check frozen replay inputs and onset semantics without sampler inference."""

from pathlib import Path

import numpy as np

from sbulm.wp11z_balanced_stage_audit import (
    FLOORS,
    _first_sustained,
    _iid_q95,
    _protocol,
    _values,
    _wp11y,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wp11y_checkpoint_is_pinned_before_stage_replay():
    previous = _wp11y(ROOT / "results/wp11y_independent_absolute_sampler.json")
    assert previous["comparison"]["per_run_mode_within_limits"] == {
        "1500": False, "1501": False, "1502": True, "1503": False
    }
    assert _protocol()["same_seeds_and_controls_as_wp11y"] == [1500, 1501, 1502, 1503]


def test_count_and_clutter_errors_use_absolute_joint_axes():
    exact = np.zeros((33, 4, 3))
    exact[20, 0, 0], exact[22, 1, 1] = .6, .4
    observed = exact.copy()
    observed[20, 0, 0] -= .1
    observed[22, 1, 1] += .1
    errors = _values(observed, exact)
    assert np.isclose(errors["alive_band"], .1)
    assert np.isclose(errors["sixth_clutter_one_signed"], .1)
    assert np.isclose(errors["last_clutter_one_signed"], .1)
    assert np.isclose(errors["band_by_last"], .1)
    q95 = _iid_q95(exact, 1500, 1)
    assert set(q95) == set(FLOORS)
    assert all(value >= 0 for value in q95.values())


def test_onset_requires_consecutive_rejuvenated_exceedances():
    stages = [{"step": i, "beta": .1 * i,
               "errors": {"rejuvenated": {"alive_band": value}},
               "limits": {"alive_band": .06},
               "first_above_operation": {"alive_band": operation}}
              for i, value, operation in ((1, .07, "resampled"),
                                          (2, .03, None),
                                          (3, .08, "likelihood_weighted"),
                                          (4, .09, "carried_from_previous_stage"))]
    onset = _first_sustained(stages, "alive_band")
    assert onset["step"] == 3
    assert np.isclose(onset["beta"], .3)
    assert onset["first_above_operation"] == "likelihood_weighted"
