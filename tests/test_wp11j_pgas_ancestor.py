"""Exact backward-weight, opt-in controls and frozen PGAS gates."""

import json
from pathlib import Path

import numpy as np
import pytest

from sbulm.conditional_pair_pg import ConditionalPairPath
from sbulm.existence_smc import BirthDeathSMCConfig
from sbulm.slot_existence import enumerate_slot_existence_support
from sbulm.wp8_generalization import _metadata, make_case
from sbulm.wp11c_full_history_pilot import _pilot_model
from sbulm.wp11j_pgas_development import (
    CARDINALITIES,
    OBSERVATION_SHA256,
    REGIME,
)
from sbulm.wp11j_pgas_development import (
    _controls as development_controls,
)
from sbulm.wp11j_pgas_development import (
    _protocol as development_protocol,
)
from sbulm.wp11j_pgas_oracle import _protocol as oracle_protocol


@pytest.mark.parametrize("beta", [0.0, 0.41, 1.0])
def test_full_suffix_ratio_agrees_with_exact_target(beta: float) -> None:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    checked = 0
    for reference in support[::17]:
        kernel = ConditionalPairPath(model, reference, (0, 1))
        for prefix in support[::19]:
            spliced = reference.copy()
            spliced[:2] = prefix[:2]
            suffix = kernel._reference_suffix_score(prefix, reference, 2, beta)
            try:
                model._validate_associations(spliced[None])
            except ValueError:
                assert suffix == -np.inf
                continue
            previous = None
            prefix_score = 0.0
            for frame in range(2):
                states, options, scores = kernel.options(prefix, previous, frame, beta)
                found = np.flatnonzero(np.all(options == prefix[frame], axis=1))
                assert len(found) == 1
                previous = int(states[found[0]])
                prefix_score += float(scores[found[0]])
            full = (model.log_existence_prior(spliced[None])[0]
                    + model.log_count_density(spliced[None], include_poisson_void=False)[0]
                    + beta * model.log_marginal_density(spliced[None])[0])
            assert suffix == pytest.approx(full - prefix_score, abs=1e-10)
            checked += 1
    assert checked > 10


def test_ancestor_sampling_is_opt_in() -> None:
    assert BirthDeathSMCConfig().paired_slot_pg_ancestor_sampling is False
    with pytest.raises(ValueError, match="requires conditional_smc"):
        BirthDeathSMCConfig(paired_slot_pg_ancestor_sampling=True)
    assert BirthDeathSMCConfig(
        paired_slot_scope="conditional_smc", paired_slot_pg_fraction=0.1,
        paired_slot_pg_ancestor_sampling=True,
    ).paired_slot_pg_ancestor_sampling


def test_fresh_observation_and_equal_pg_allocation() -> None:
    observed = _metadata(REGIME, make_case(REGIME))["model_and_detections"]
    assert observed["detections_sha256"] == OBSERVATION_SHA256
    assert tuple(observed["detection_cardinalities"]) == CARDINALITIES
    pg, pgas = development_controls(530, "pg"), development_controls(530, "pgas")
    assert pg.n_particles == pgas.n_particles == 512
    assert pg.paired_slot_pg_fraction == pgas.paired_slot_pg_fraction == 0.1
    assert pg.paired_slot_path_particles == pgas.paired_slot_path_particles == 4
    assert not pg.paired_slot_pg_ancestor_sampling
    assert pgas.paired_slot_pg_ancestor_sampling
    assert development_protocol()["arm_order_by_seed"] == {
        "530": ["pg", "pgas"], "531": ["pgas", "pg"]
    }


def test_saved_exact_oracle_matches_frozen_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/wp11j_pgas_oracle.json").read_text())
    protocol = oracle_protocol()
    assert result["protocol"] == protocol
    assert all(set(result["runs"][arm]) == {"620", "621", "622"}
               for arm in protocol["arms"])
    passed = (
        all(run["within_exact_envelope"] for arm in result["runs"].values()
            for run in arm.values())
        and result["mean_first_frame_state_change_rate"]["pgas"]
        > result["mean_first_frame_state_change_rate"]["pg"]
        and all(value <= protocol["maximum_pgas_pg_cpu_ratio_each_seed"]
                for value in result["pgas_pg_cpu_ratios"].values())
    )
    assert result["status"] == (
        "exact_oracle_passed" if passed else "stopped_before_new_development_case"
    )


def test_saved_development_matches_frozen_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/wp11j_pgas_development.json").read_text())
    assert result["protocol"] == development_protocol()
    assert all(set(result["runs"][str(seed)]) == {"pg", "pgas"} for seed in (530, 531))
    passed = (
        all(check["passed"] for check in result["seed_checks"].values())
        and result["target_metrics_improved"]
        and result["other_metrics_within_tolerance"]
    )
    assert result["status"] == (
        "development_screen_passed" if passed else "stopped_before_high_particle_test"
    )
