"""Frozen WP11f interleaved timing and exact-oracle gates."""

import json
from pathlib import Path

from sbulm.wp8_generalization import _metadata, make_case
from sbulm.wp11c_full_history_pilot import _pilot_model
from sbulm.wp11f_four_path_development import (
    EXPECTED_CARDINALITIES,
    EXPECTED_OBSERVATION_SHA256,
    REGIME,
)
from sbulm.wp11f_four_path_development import _controls as four_path_controls
from sbulm.wp11f_four_path_development import _protocol as four_path_protocol
from sbulm.wp11f_interleaved_cost import EXPECTED_BASELINE_HASHES, N_REPEATS
from sbulm.wp11f_interleaved_cost import _protocol as cost_protocol
from sbulm.wp11f_path_count_oracle import (
    ARMS,
    MIN_MEAN_FOUR_COUNT_CHANGE,
    SEEDS,
)
from sbulm.wp11f_path_count_oracle import _protocol as oracle_protocol


def test_cost_protocol_alternates_arms_on_same_exposed_starts() -> None:
    protocol = cost_protocol()
    assert protocol["arm_order"] == [
        ["local", "sparse_pg"], ["sparse_pg", "local"],
        ["local", "sparse_pg"], ["sparse_pg", "local"],
    ]
    assert protocol["repeats"] == N_REPEATS == 4
    assert protocol["baseline_hashes"] == EXPECTED_BASELINE_HASHES
    assert protocol["thread_env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert protocol["max_median_cpu_ratio_per_seed"] == 3.0


def test_oracle_protocol_is_frozen_before_path_count_comparison() -> None:
    protocol = oracle_protocol(_pilot_model())
    assert protocol["seeds"] == list(SEEDS) == [81, 82, 83]
    assert protocol["arms_path_particles"] == list(ARMS) == [4, 8]
    assert protocol["minimum_mean_four_count_change"] == MIN_MEAN_FOUR_COUNT_CHANGE == 0.2
    assert protocol["maximum_median_four_eight_cpu_ratio"] == 0.75
    assert protocol["arm_order"] == [[4, 8], [8, 4], [4, 8]]


def test_new_four_path_development_case_is_frozen() -> None:
    metadata = _metadata(REGIME, make_case(REGIME))["model_and_detections"]
    assert metadata["detections_sha256"] == EXPECTED_OBSERVATION_SHA256
    assert tuple(metadata["detection_cardinalities"]) == EXPECTED_CARDINALITIES
    protocol = four_path_protocol()
    assert protocol["seeds"] == [410, 411]
    assert protocol["maximum_median_cpu_ratio_per_seed"] == 2.0
    assert protocol["path_particles"] == 4
    assert four_path_controls("local").paired_slot_scope == "local"
    assert four_path_controls("sparse_pg4").paired_slot_path_particles == 4


def test_wp11f_artifacts_preserve_the_gates() -> None:
    cost = json.loads(Path("results/wp11f_interleaved_cost.json").read_text())
    oracle = json.loads(Path("results/wp11f_path_count_oracle.json").read_text())
    assert cost["protocol"] == cost_protocol()
    assert oracle["protocol"] == oracle_protocol(_pilot_model())
    assert oracle["status"] in (
        "small_oracle_path_count_passed", "stopped_before_new_development_case"
    )
    assert cost["status"] in (
        "controlled_cost_screen_passed", "controlled_cost_screen_failed"
    )
    assert set(cost["comparisons"]) == {"310", "311"}
    assert all(set(oracle["runs"][str(size)]) == {str(seed) for seed in SEEDS} for size in ARMS)


def test_four_path_development_artifact_preserves_gate() -> None:
    result = json.loads(Path("results/wp11f_four_path_development.json").read_text())
    assert result["protocol"] == four_path_protocol()
    assert result["status"] in ("four_path_development_screen_passed", "stopped_before_holdout")
    assert set(result["comparisons"]) == {"410", "411"}
