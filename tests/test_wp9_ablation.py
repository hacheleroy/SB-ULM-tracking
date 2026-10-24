"""Checks that the counterfactual WP9 inputs were frozen before inference."""

import json
from pathlib import Path

import numpy as np

from sbulm.wp8_generalization import make_case
from sbulm.wp9_ablation import (
    ABLATIONS,
    BASE,
    EXPECTED_OBSERVATIONS,
    _load_or_initialize,
    _metadata,
    make_ablation_case,
)


def test_four_frozen_observation_counterfactuals() -> None:
    baseline = make_case(BASE)
    reference = baseline.model.sequence
    assert set(EXPECTED_OBSERVATIONS) == set(ABLATIONS) | {"baseline"}
    for name in ABLATIONS:
        case = make_ablation_case(name)
        metadata = _metadata(name, case)
        observed = metadata["model_and_detections"]
        assert {key: observed[key] for key in EXPECTED_OBSERVATIONS[name]} == EXPECTED_OBSERVATIONS[
            name
        ]
        assert case.model.n_slots == 4
        assert case.model.sequence.n_frames == 8
        assert case.coordinate_std == baseline.coordinate_std
        assert case.length_scale == baseline.length_scale

    no_clutter = make_ablation_case("remove_observed_clutter").model.sequence
    complete = make_ablation_case("complete_true_detections").model.sequence
    wider = make_ablation_case("widen_crossing").model.sequence
    schedule = make_ablation_case("wp7_existence_schedule").model.sequence
    assert no_clutter.reference_ids is not None
    assert complete.reference_ids is not None
    assert wider.reference_ids is not None
    assert schedule.reference_ids is not None
    assert reference.reference_ids is not None
    assert all(
        np.array_equal(ids, original[original >= 0])
        for ids, original in zip(no_clutter.reference_ids, reference.reference_ids, strict=True)
    )
    for altered, original, ids in zip(
        no_clutter.detections, reference.detections, reference.reference_ids, strict=True
    ):
        np.testing.assert_array_equal(altered, original[ids >= 0])
    assert sum(np.count_nonzero(ids >= 0) for ids in complete.reference_ids) == 25
    for altered, original in zip(complete.detections, reference.detections, strict=True):
        np.testing.assert_array_equal(altered[: len(original)], original)
    assert all(
        np.array_equal(ids, original)
        for ids, original in zip(wider.reference_ids, reference.reference_ids, strict=True)
    )
    for altered, original, ids in zip(
        wider.detections, reference.detections, reference.reference_ids, strict=True
    ):
        np.testing.assert_array_equal(altered[ids < 0], original[ids < 0])
    assert sum(np.count_nonzero(ids >= 0) for ids in schedule.reference_ids) == 13


def test_ablation_checkpoint_round_trip(tmp_path) -> None:
    source = tmp_path / "wp8.json"
    source.write_text(
        Path("results/wp8_generalization.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output = tmp_path / "wp9.json"
    fresh = _load_or_initialize(output, source)
    output.write_text(json.dumps(fresh), encoding="utf-8")
    assert _load_or_initialize(output, source) == fresh
