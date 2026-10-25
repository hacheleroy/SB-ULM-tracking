"""Frozen observation and label-invariant WP11 diagnostic checks."""

from pathlib import Path

import numpy as np
import pytest

from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp8_generalization import make_case
from sbulm.wp11_tracklet_validation import (
    CASE_REGIMES,
    EXPECTED_HASHES,
    _count_signature_histogram,
    _edge_and_clutter_marginals,
    _load_or_initialize,
    _maximum_nested_difference,
    _signature_comparison,
)


def test_observation_hashes_and_inherited_thresholds_are_frozen(tmp_path: Path) -> None:
    for regime, digest in zip(CASE_REGIMES, EXPECTED_HASHES, strict=True):
        assert _case_metadata(make_case(regime))["detections_sha256"] == digest
    result = _load_or_initialize(
        tmp_path / "fresh.json",
        Path("results/wp7_convergence_audit.json"),
        Path("results/wp8_generalization.json"),
    )
    assert result["protocol"]["oracle_thresholds"]["maximum_event_error"] == 0.06
    assert result["protocol"]["inherited_seven_metric_gate"] == 0.10
    assert result["wp8_failed_baseline"]["stability_passed"] is False


def test_association_edge_and_clutter_summaries_ignore_slot_permutation() -> None:
    case = make_case(CASE_REGIMES[0])
    rng = np.random.default_rng(5)
    states = case.model.sample_base_associations(n_samples=30, rng=rng)
    permuted = states[:, :, ::-1]
    first = _edge_and_clutter_marginals(states, case)
    second = _edge_and_clutter_marginals(permuted, case)
    assert _maximum_nested_difference(first, second) == pytest.approx(0.0)
    assert _count_signature_histogram(states) == _count_signature_histogram(permuted)
    assert _signature_comparison(
        _count_signature_histogram(states), _count_signature_histogram(permuted)
    )["histogram_tv"] == pytest.approx(0.0)
