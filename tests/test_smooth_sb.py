import numpy as np
import pytest

from sbulm.data import DetectionSequence
from sbulm.inference import enumerate_association_posterior
from sbulm.simulation import CrossingConfig, simulate_crossing
from sbulm.smooth_sb import (
    compare_smooth_sb_samples,
    enumerate_smooth_sb_target,
    export_smooth_sb_npz,
    to_smooth_sb_array,
)


def test_export_is_compatible_and_keeps_truth_separate(tmp_path) -> None:
    truth, detections = simulate_crossing(seed=3)
    path = export_smooth_sb_npz(detections, tmp_path / "crossing.npz", truth=truth)
    with np.load(path) as exported:
        assert exported["arr_0"].shape == (9, 2, 2)
        np.testing.assert_allclose(exported["arr_0"], to_smooth_sb_array(detections))
        np.testing.assert_allclose(exported["truth_positions"], truth.positions)
        np.testing.assert_array_equal(
            exported["reference_ids"], np.stack(detections.reference_ids)
        )


def test_export_rejects_variable_cardinality(tmp_path) -> None:
    sequence = DetectionSequence(
        times=np.array([0.0, 1.0]),
        detections=(np.zeros((2, 2)), np.zeros((1, 2))),
    )
    with pytest.raises(ValueError, match="same number"):
        export_smooth_sb_npz(sequence, tmp_path / "invalid.npz")


def test_exact_samples_have_zero_distribution_error() -> None:
    config = CrossingConfig(n_frames=4, central_separation=0.1, localization_std=0.05)
    _, detections = simulate_crossing(config, seed=2)
    posterior = enumerate_association_posterior(
        detections,
        localization_std=config.localization_std,
        acceleration_std=0.08,
        initial_velocity_std=0.6,
    )
    counts = np.round(posterior.probabilities * 100_000).astype(int)
    samples = np.repeat(posterior.associations, counts, axis=0)
    comparison = compare_smooth_sb_samples(samples, posterior)
    assert comparison.invalid_sample_fraction == 0.0
    assert comparison.total_variation_distance < 1e-4
    assert comparison.jensen_shannon_divergence < 1e-7


def test_non_bijective_sample_is_reported_as_invalid_mass() -> None:
    config = CrossingConfig(n_frames=4, localization_std=0.05)
    _, detections = simulate_crossing(config, seed=1)
    posterior = enumerate_association_posterior(
        detections,
        localization_std=config.localization_std,
    )
    samples = posterior.associations[[posterior.map_index]].copy()
    samples[0, 2] = 0
    comparison = compare_smooth_sb_samples(samples, posterior)
    assert comparison.invalid_sample_fraction == 1.0
    assert comparison.total_variation_distance == pytest.approx(1.0)
    assert comparison.valid_conditioned_total_variation_distance is None


def test_smooth_sb_target_prefers_the_smoother_crossing_path() -> None:
    config = CrossingConfig(central_separation=0.05, localization_std=0.075)
    _, detections = simulate_crossing(config, seed=0)
    posterior = enumerate_smooth_sb_target(detections)
    ordered = np.argsort(posterior.probabilities)[::-1]
    assert posterior.probabilities[ordered[0]] > 0.999
    assert posterior.associations.shape == (256, 9, 2)


def test_physical_model_times_match_equivalent_rescaling() -> None:
    config = CrossingConfig(
        n_frames=5,
        frame_period=0.5,
        central_separation=0.05,
        localization_std=0.075,
    )
    _, detections = simulate_crossing(config, seed=0)
    rescaled = enumerate_smooth_sb_target(
        detections, t_end=2.0, length_scale=3.0
    )
    physical = enumerate_smooth_sb_target(
        detections, model_times=detections.times, length_scale=3.0
    )
    np.testing.assert_allclose(rescaled.probabilities, physical.probabilities)
