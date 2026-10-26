"""Four-slot seven-detection exact bridge and GP factor lookup."""

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
