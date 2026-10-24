from math import factorial, log

import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.localization import (
    MaternLocalizationModel,
    enumerate_matern_localization_posterior,
)
from sbulm.simulation import simulate_crossing
from sbulm.smc import (
    AnnealedSMCConfig,
    effective_sample_size,
    sample_matern_localization_posterior,
    systematic_resample,
)
from sbulm.tsbm_reference import (
    compare_tsbm_association_samples,
    finite_sample_baseline,
)


def _small_problem() -> tuple[object, object, tuple[float, ...], float]:
    config = physical_crossing_config(
        n_frames=5,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, detections = simulate_crossing(config, seed=4)
    coordinate_std = crossing_design_coordinate_std(config)
    calibration = calibrate_matern_from_coordinate_std(
        coordinate_std, rms_speed=6.84
    )
    return config, detections, calibration.coordinate_std, calibration.length_scale


def test_collapsed_model_scores_match_exact_enumerator() -> None:
    config, detections, coordinate_std, length_scale = _small_problem()
    exact = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    model = MaternLocalizationModel(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    log_scores = model.log_marginal_density(exact.associations)
    probabilities = np.exp(log_scores - logsumexp(log_scores))
    np.testing.assert_allclose(probabilities, exact.probabilities, atol=1e-14)


def test_conditional_path_draws_match_gaussian_moments() -> None:
    config, detections, coordinate_std, length_scale = _small_problem()
    exact = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    model = MaternLocalizationModel(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    association = exact.associations[exact.map_index : exact.map_index + 1]
    expected_mean, expected_covariance = model.conditional_path_moments(association)
    repeated = np.repeat(association, 12_000, axis=0)
    paths = model.sample_conditional_paths(
        repeated, rng=np.random.default_rng(19)
    )
    coordinate = paths[:, :, 0, 0]
    mean_standard_error = np.sqrt(
        np.diag(expected_covariance[0]) / coordinate.shape[0]
    )
    np.testing.assert_allclose(
        coordinate.mean(axis=0),
        expected_mean[0, :, 0, 0],
        atol=5.0 * float(np.max(mean_standard_error)),
    )
    empirical_covariance = np.cov(coordinate, rowvar=False)
    np.testing.assert_allclose(
        np.diag(empirical_covariance),
        np.diag(expected_covariance[0]),
        rtol=0.04,
    )
    relative_frobenius_error = np.linalg.norm(
        empirical_covariance - expected_covariance[0]
    ) / np.linalg.norm(expected_covariance[0])
    assert relative_frobenius_error < 0.035


def test_weight_and_systematic_resampling_utilities() -> None:
    assert effective_sample_size(np.zeros(8)) == pytest.approx(8.0)
    ancestors = systematic_resample(np.ones(8), np.random.default_rng(2))
    np.testing.assert_array_equal(ancestors, np.arange(8))
    with pytest.raises(ValueError, match="positive total mass"):
        systematic_resample(np.zeros(3), np.random.default_rng(0))


def test_annealed_smc_recovers_enumerable_posterior() -> None:
    config, detections, coordinate_std, length_scale = _small_problem()
    exact = enumerate_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    controls = AnnealedSMCConfig(
        n_particles=2_048,
        target_ess_fraction=0.80,
        mh_sweeps=6,
        seed=31,
    )
    result = sample_matern_localization_posterior(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
        config=controls,
    )
    comparison = compare_tsbm_association_samples(result.associations, exact)
    finite_floor = finite_sample_baseline(
        exact, n_samples=controls.n_particles, n_replicates=200, seed=9
    )
    assert comparison.invalid_sample_fraction == 0.0
    assert comparison.credible_set_recall_95 == 1.0
    assert comparison.total_variation_distance <= 2.5 * finite_floor.total_variation_q95
    assert result.latent_paths.shape == (*result.associations.shape, 2)
    assert all(
        0.0 < stage.beta <= 1.0 and stage.effective_sample_size > 0
        for stage in result.stages
    )
    assert np.all(np.diff([stage.beta for stage in result.stages]) > 0)
    assert result.stages[-1].mh_sweeps == (
        controls.mh_sweeps + controls.final_mh_sweeps
    )

    model = MaternLocalizationModel(
        detections,
        localization_std=config.localization_std,
        coordinate_std=coordinate_std,
        length_scale=length_scale,
    )
    exact_log_evidence = float(
        logsumexp(model.log_marginal_density(exact.associations))
        - (detections.n_frames - 1)
        * log(factorial(detections.cardinalities[0]))
    )
    assert result.log_evidence_estimate == pytest.approx(
        exact_log_evidence, abs=0.12
    )


def test_smc_configuration_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="at least two"):
        AnnealedSMCConfig(n_particles=1)
    with pytest.raises(ValueError, match="target_ess_fraction"):
        AnnealedSMCConfig(target_ess_fraction=1.0)
    with pytest.raises(ValueError, match="final_mh_sweeps"):
        AnnealedSMCConfig(final_mh_sweeps=-1)
