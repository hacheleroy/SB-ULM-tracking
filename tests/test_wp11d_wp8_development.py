"""Predeclared development comparison and label-free diagnostics."""

import json
from pathlib import Path

import numpy as np
import pytest

from sbulm.wp8_generalization import EXPECTED_OBSERVATIONS
from sbulm.wp11d_wp8_development import (
    ARMS,
    N_PARTICLES,
    N_PATH_PARTICLES,
    N_SWEEPS,
    SEEDS,
    _cross_seed,
    _histograms,
    _protocol,
)


def test_wp8_development_protocol_is_not_a_new_holdout() -> None:
    protocol = _protocol()
    assert protocol["observation_sha256"] == EXPECTED_OBSERVATIONS[
        "cluttered_staggered_crossing"
    ]["detections_sha256"]
    assert protocol["seeds"] == list(SEEDS) == [100, 200]
    assert protocol["n_particles"] == N_PARTICLES == 512
    assert protocol["post_baseline_pair_sweeps"] == N_SWEEPS == 2
    assert protocol["path_particles"] == N_PATH_PARTICLES == 8
    assert protocol["arms"] == list(ARMS) == ["conditional_smc", "local"]


def test_count_histograms_are_slot_permutation_invariant() -> None:
    states = np.asarray(
        [[[-2, 0], [1, -1]], [[-1, 0], [-2, 0]]], dtype=np.int64
    )
    first = _histograms(states, 2)
    second = _histograms(states[:, :, ::-1], 2)
    for x, y in zip(first, second, strict=True):
        assert np.array_equal(x, y)
        assert np.allclose(np.sum(x, axis=1), 1.0)
    summary = {
        "alive_count_histograms": first[0].tolist(),
        "assigned_count_histograms": first[1].tolist(),
        "expected_clutter_by_frame": [0.5, 0.0],
    }
    assert _cross_seed(summary, summary) == {
        "max_alive_count_tv": 0.0,
        "max_assigned_count_tv": 0.0,
        "max_expected_clutter_difference": 0.0,
    }


def test_cross_seed_count_difference() -> None:
    one = {
        "alive_count_histograms": [[1, 0], [0.5, 0.5]],
        "assigned_count_histograms": [[1, 0], [0, 1]],
        "expected_clutter_by_frame": [0, 1],
    }
    two = {
        "alive_count_histograms": [[0, 1], [0.5, 0.5]],
        "assigned_count_histograms": [[1, 0], [1, 0]],
        "expected_clutter_by_frame": [0.5, 0],
    }
    result = _cross_seed(one, two)
    assert result["max_alive_count_tv"] == pytest.approx(1)
    assert result["max_assigned_count_tv"] == pytest.approx(1)
    assert result["max_expected_clutter_difference"] == pytest.approx(1)


def test_wp8_development_artifact_preserves_early_stop() -> None:
    result = json.loads(Path("results/wp11d_wp8_development.json").read_text())
    assert result["protocol"] == _protocol()
    assert result["status"] == "stopped_before_large_inference"
    assert set(result["baseline"]) == {str(seed) for seed in SEEDS}
    assert all(set(result["runs"][arm]) == {str(seed) for seed in SEEDS} for arm in ARMS)
    assert all(
        not result["seed_comparisons"][str(seed)]["gate_passed"] for seed in SEEDS
    )
    assert all(
        result["seed_comparisons"][str(seed)]["count_change_advantage"] > 0
        for seed in SEEDS
    )
