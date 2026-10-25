"""Frozen protocol and label-invariant pilot diagnostics."""

import numpy as np

from sbulm.wp11d_conditional_pair_pilot import (
    ARMS,
    MINIMUM_MEAN_COUNT_CHANGE_RATE,
    N_PARTICLES,
    N_PATH_PARTICLES,
    SEEDS,
    _count_signatures,
    _protocol,
)


def test_wp11d_protocol_freezes_oracle_before_larger_development() -> None:
    protocol = _protocol()
    assert protocol["observation_sha256"] == (
        "c645fab464abe15579b843a35081078e62983d6fdfd014c7692a88a368dace72"
    )
    assert protocol["seeds"] == list(SEEDS) == [74, 75, 76]
    assert protocol["arms"] == list(ARMS) == ["conditional_smc", "local"]
    assert protocol["n_particles"] == N_PARTICLES == 512
    assert protocol["path_particles"] == N_PATH_PARTICLES == 8
    assert protocol["minimum_mean_count_signature_change_rate"] == MINIMUM_MEAN_COUNT_CHANGE_RATE
    assert MINIMUM_MEAN_COUNT_CHANGE_RATE == 0.01


def test_count_signatures_ignore_slot_labels_but_retain_clutter() -> None:
    states = np.asarray(
        [[[-2, 0], [-1, 0]], [[-2, 0], [-1, -1]]], dtype=np.int64
    )
    assert np.array_equal(_count_signatures(states), _count_signatures(states[:, :, ::-1]))
    assert not np.array_equal(_count_signatures(states)[0], _count_signatures(states)[1])
