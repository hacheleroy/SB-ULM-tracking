"""Frozen held-out synthetic conditions for the WP8 stability study."""

import json
from dataclasses import asdict

from sbulm.wp8_generalization import (
    EXPECTED_OBSERVATIONS,
    REGIMES,
    SEEDS,
    _load_or_initialize,
    _metadata,
    make_case,
)


def test_regimes_are_new_and_reproducible() -> None:
    assert SEEDS == (100, 200, 101, 201, 102, 202)
    assert len(REGIMES) == 2
    for regime in REGIMES:
        case = make_case(regime)
        metadata = _metadata(regime, case)
        observed = metadata["model_and_detections"]
        assert {
            key: observed[key] for key in EXPECTED_OBSERVATIONS[regime.name]
        } == EXPECTED_OBSERVATIONS[regime.name]
        assert case.model.n_slots == 4
        assert case.model.sequence.n_frames == 8
        assert metadata["truth_diagnostics"]["alive_slot_frames"] > 0
        assert json.loads(json.dumps(asdict(regime))) == json.loads(json.dumps(metadata["regime"]))


def test_frozen_protocol_survives_checkpoint_round_trip(tmp_path) -> None:
    output = tmp_path / "audit.json"
    fresh = _load_or_initialize(output, 8_192)
    output.write_text(json.dumps(fresh), encoding="utf-8")
    resumed = _load_or_initialize(output, 8_192)
    assert resumed == fresh
