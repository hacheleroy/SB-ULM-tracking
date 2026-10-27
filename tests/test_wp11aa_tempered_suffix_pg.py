"""Check the opt-in schedule and the two pinned absolute comparisons."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sbulm.data import DetectionSequence
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    _metropolis_birth_death_rejuvenation,
    _paired_suffix_pg_fraction,
)
from sbulm.slot_existence import SlotExistenceMaternModel
from sbulm.wp11aa_tempered_suffix_pg import (
    TemperedSuffixSMCConfig,
    _case,
    _controls,
    _decision,
    _metrics,
    _previous,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def test_existing_control_serialization_and_pinned_references():
    assert "paired_slot_suffix_pg_beta_switch" not in asdict(_controls(1300, "control"))
    candidate = _controls(1300, "candidate")
    assert candidate.paired_slot_suffix_pg_beta_switch == .75
    assert candidate.paired_slot_suffix_pg_final_fraction == .4
    assert candidate.final_mh_sweeps == _controls(1300, "control").final_mh_sweeps
    previous = _previous(ROOT / "results/wp11u_late_suffix_pg.json",
                         ROOT / "results/wp11y_independent_absolute_sampler.json",
                         ROOT / "results/wp11z_balanced_stage_summary.json")
    assert len(previous["primary_19"]["runs"]) == 6
    for name, shape in (("primary_19", (33, 3, 2)), ("decoy_17", (33, 4, 3))):
        _, exact, _ = _case(name, ROOT / "results/wp11t_subset_exact_primary.json",
                            ROOT / "results/wp11x_frame_six_exact_reference.json",
                            ROOT / "results/wp11x_geometry_ledger.json")
        assert exact.shape == shape
        assert _metrics(exact, exact)["full_joint_tv"] == 0
    assert _protocol()["decision"].startswith("both exposed-case gates")


def _toy_model():
    return SlotExistenceMaternModel(
        DetectionSequence(times=np.arange(3) * .001,
                          detections=(np.array([[-.008, .002]]), np.empty((0, 2)),
                                      np.array([[.008, -.002]]))),
        n_slots=2, localization_std=.005, coordinate_std=(.02, .01),
        length_scale=.003, detection_probability=.7,
        expected_clutter_per_frame=.15,
        roi_lower=(-.04, -.04), roi_upper=(.04, .04),
    )


def test_tempered_suffix_starts_after_switch_and_early_path_is_identical():
    base = BirthDeathSMCConfig(
        n_particles=48, association_move_probability=0, interval_move_probability=0,
        slot_swap_probability=0, paired_slot_move_probability=1,
        paired_slot_scope="conditional_smc", paired_slot_pg_fraction=.1,
        paired_slot_path_particles=4,
    )
    options = asdict(base)
    options.update(paired_slot_suffix_pg_final_fraction=.4,
                   paired_slot_suffix_pg_start_frame=1,
                   paired_slot_suffix_pg_beta_switch=.75)
    candidate = TemperedSuffixSMCConfig(**options)
    assert [_paired_suffix_pg_fraction(candidate, b) for b in (.0, .74, .75, 1.)] == [
        0, 0, .4, .4
    ]
    model = _toy_model()
    states = model.sample_base_associations(n_samples=48, rng=np.random.default_rng(19))

    def move(controls, beta):
        return _metropolis_birth_death_rejuvenation(
            states.copy(), model.log_marginal_density(states),
            model.log_count_density(states, include_poisson_void=False),
            model.log_existence_prior(states), model, beta=beta,
            n_sweeps=2, config=controls, rng=np.random.default_rng(49),
        )

    control_early, candidate_early = move(base, .74), move(candidate, .74)
    for left, right in zip(control_early[:4], candidate_early[:4], strict=True):
        np.testing.assert_array_equal(left, right)
    assert control_early[-1] == candidate_early[-1]
    assert move(candidate, .8)[-1][12] > 0


def test_invalid_tempered_schedule_is_rejected():
    with pytest.raises(ValueError, match="positive fraction and switch"):
        TemperedSuffixSMCConfig(paired_slot_suffix_pg_beta_switch=0)
    with pytest.raises(ValueError, match="fractions exceed one"):
        TemperedSuffixSMCConfig(paired_slot_scope="conditional_smc",
                                paired_slot_pg_fraction=.1,
                                paired_slot_pg_beta_switch=.5,
                                paired_slot_pg_early_fraction=.8,
                                paired_slot_suffix_pg_final_fraction=.4)


def test_decoy_replication_rule_is_separate_from_median_targets():
    row = {"cpu_seconds": 100.0, "errors": {
        "full_joint_tv": .09, "band_by_sixth_tv": .04,
        "maximum_alive_band_error": .02, "alive_total_tv": .02,
        "maximum_joint_cell_error": .01,
        "sixth_clutter_mean_absolute_error": .02,
        "last_clutter_mean_absolute_error": .02,
        "sixth_clutter_signed_errors": [0, .01, 0, 0],
        "last_clutter_signed_errors": [0, .01, 0],
    }}
    controls = [row for _ in range(4)]
    candidates = [{"cpu_seconds": 110.0, "errors": dict(row["errors"])}
                  for _ in range(4)]
    for candidate in candidates[:2]:
        candidate["errors"]["sixth_clutter_signed_errors"] = [0, .07, 0, 0]
    decision = _decision("decoy_17", controls, candidates)
    assert decision["gates"]["absolute_targets"]
    assert not decision["gates"]["replicated_modes"]
    assert not decision["passed"]
