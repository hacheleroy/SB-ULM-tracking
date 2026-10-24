import numpy as np
import pytest

from sbulm.calibration import physical_crossing_config
from sbulm.simulation import simulate_crossing
from sbulm.tsbm_reference import (
    brownian_bridge_covariance,
    canonicalize_continuous_paths,
    canonicalize_global_particle_labels,
    collapse_global_label_probabilities,
    collapse_global_label_symmetry,
    compare_association_probabilities,
    compare_tsbm_association_samples,
    endpoint_permutation_joint,
    endpoint_permutation_marginals,
    enumerate_tsbm_brownian_localization_posterior,
    finite_sample_baseline,
    nearest_detection_associations,
    observation_set_rms,
    rao_blackwellized_association_probabilities,
    sample_exact_tsbm_paths,
    sliced_wasserstein_distance,
)


def test_brownian_bridge_covariance_is_symmetric_positive_definite() -> None:
    covariance = brownian_bridge_covariance(
        np.asarray([0.25, 0.5, 0.75]),
        start_time=0.0,
        end_time=1.0,
        diffusion_std=0.4,
    )
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_matched_tsbm_target_has_exchangeable_boundary_marginals() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=2)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    assert posterior.associations.shape == (32, 5, 2)
    assert posterior.probabilities.sum() == pytest.approx(1.0)
    initial, final = endpoint_permutation_marginals(posterior)
    np.testing.assert_allclose(initial, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(final, [0.5, 0.5], atol=1e-12)
    joint = endpoint_permutation_joint(posterior)
    np.testing.assert_allclose(joint.sum(axis=1), [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(joint.sum(axis=0), [0.5, 0.5], atol=1e-12)


def test_continuous_paths_project_to_exact_detection_associations() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=1)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    selected = posterior.trajectories[[0, 3, 7]]
    projected = nearest_detection_associations(selected, detections)
    np.testing.assert_array_equal(projected, posterior.associations[[0, 3, 7]])


def test_exact_frequency_sample_has_small_distribution_error() -> None:
    config = physical_crossing_config(n_frames=4)
    _, detections = simulate_crossing(config, seed=0)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    counts = np.round(200_000 * posterior.probabilities).astype(int)
    samples = np.repeat(posterior.associations, counts, axis=0)
    comparison = compare_tsbm_association_samples(samples, posterior)
    assert comparison.invalid_sample_fraction == 0.0
    assert comparison.total_variation_distance < 1e-4
    assert comparison.jensen_shannon_divergence < 1e-7
    assert comparison.exact_mass_covered_by_empirical_support > 0.9999
    assert comparison.credible_set_recall_95 == 1.0


def test_global_label_symmetry_collapses_paired_hypotheses() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=2)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    collapsed = collapse_global_label_symmetry(posterior)
    assert collapsed.associations.shape[0] == posterior.associations.shape[0] // 2
    np.testing.assert_array_equal(collapsed.associations[:, 0], [[0, 1]] * 16)
    assert collapsed.probabilities.sum() == pytest.approx(1.0)
    canonical = canonicalize_global_particle_labels(posterior.associations)
    assert np.unique(canonical, axis=0).shape[0] == 16


def test_finite_sample_baseline_shrinks_with_more_samples() -> None:
    config = physical_crossing_config(n_frames=4)
    _, detections = simulate_crossing(config, seed=0)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    small = finite_sample_baseline(
        posterior, n_samples=100, n_replicates=100, seed=4
    )
    large = finite_sample_baseline(
        posterior, n_samples=10_000, n_replicates=100, seed=4
    )
    assert large.total_variation_median < small.total_variation_median
    assert large.jensen_shannon_median < small.jensen_shannon_median
    assert (
        large.credible_set_recall_95_q05
        >= small.credible_set_recall_95_q05
    )
    assert large.exact_mass_covered_q05 >= small.exact_mass_covered_q05


def test_exact_continuous_sampler_respects_endpoints_and_global_canonicalization() -> None:
    config = physical_crossing_config(n_frames=5)
    _, detections = simulate_crossing(config, seed=3)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    paths, associations = sample_exact_tsbm_paths(
        detections,
        posterior,
        localization_std=config.localization_std,
        diffusion_std=0.02,
        n_samples=500,
        seed=9,
    )
    projected = nearest_detection_associations(paths, detections)
    np.testing.assert_array_equal(projected[:, [0, -1]], associations[:, [0, -1]])
    canonical = canonicalize_continuous_paths(paths, associations)
    expected_start = np.broadcast_to(detections.detections[0], canonical[:, 0].shape)
    np.testing.assert_allclose(canonical[:, 0], expected_start)
    residuals = observation_set_rms(paths, detections)
    np.testing.assert_allclose(residuals[:, [0, -1]], 0.0, atol=1e-12)


def test_sliced_wasserstein_is_zero_for_identical_samples() -> None:
    samples = np.random.default_rng(0).normal(size=(100, 6))
    assert sliced_wasserstein_distance(samples, samples, seed=2) == pytest.approx(0.0)


def test_rao_blackwellized_probabilities_recover_exact_target_from_exact_paths() -> None:
    config = physical_crossing_config(n_frames=4, localization_std_um=5.0)
    _, detections = simulate_crossing(config, seed=5)
    posterior = enumerate_tsbm_brownian_localization_posterior(
        detections,
        localization_std=config.localization_std,
        diffusion_std=0.02,
    )
    paths, _ = sample_exact_tsbm_paths(
        detections,
        posterior,
        localization_std=config.localization_std,
        diffusion_std=0.02,
        n_samples=20_000,
        seed=12,
    )
    estimated = rao_blackwellized_association_probabilities(
        paths,
        detections,
        posterior.associations,
        localization_std=config.localization_std,
    )
    comparison = compare_association_probabilities(
        estimated, posterior, n_path_samples=paths.shape[0]
    )
    assert estimated.sum() == pytest.approx(1.0)
    assert comparison.total_variation_distance < 0.02

    canonical_exact = collapse_global_label_symmetry(posterior)
    canonical_support, canonical_estimated = collapse_global_label_probabilities(
        posterior.associations, estimated
    )
    np.testing.assert_array_equal(canonical_support, canonical_exact.associations)
    assert canonical_estimated.sum() == pytest.approx(1.0)
