import numpy as np
import pytest

from sbulm.baselines import hungarian_tracker
from sbulm.inference import enumerate_association_posterior
from sbulm.metrics import (
    association_accuracy,
    posterior_reference_metrics,
    true_in_credible_set,
)
from sbulm.simulation import CrossingConfig, simulate_crossing


def test_enumerated_posterior_is_normalized() -> None:
    config = CrossingConfig(n_frames=6)
    _, detections = simulate_crossing(config, seed=3)
    posterior = enumerate_association_posterior(
        detections,
        localization_std=config.localization_std,
        acceleration_std=0.08,
    )
    assert posterior.associations.shape == (2 ** 5, 6, 2)
    np.testing.assert_allclose(posterior.probabilities.sum(), 1.0)
    assert np.all(posterior.probabilities >= 0)
    assert 1 <= posterior.credible_set_size(0.95) <= posterior.probabilities.size


def test_reference_metrics_are_bounded() -> None:
    config = CrossingConfig(n_frames=6)
    _, detections = simulate_crossing(config, seed=8)
    associations, _ = hungarian_tracker(detections, constant_velocity=True)
    assert 0.0 <= association_accuracy(detections, associations) <= 1.0
    posterior = enumerate_association_posterior(
        detections,
        localization_std=config.localization_std,
    )
    metrics = posterior_reference_metrics(detections, posterior)
    assert 0.0 <= metrics["true_hypothesis_mass"] <= 1.0
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0


def test_credible_set_mass_is_validated() -> None:
    _, detections = simulate_crossing(CrossingConfig(n_frames=4), seed=0)
    posterior = enumerate_association_posterior(
        detections, localization_std=0.075
    )
    assert isinstance(true_in_credible_set(detections, posterior, 0.95), bool)
    with pytest.raises(ValueError, match="mass"):
        true_in_credible_set(detections, posterior, 0.0)

