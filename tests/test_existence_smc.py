import numpy as np
import pytest
from scipy.special import logsumexp

from sbulm.calibration import (
    calibrate_matern_from_coordinate_std,
    crossing_design_coordinate_std,
    physical_crossing_config,
)
from sbulm.existence_smc import (
    BirthDeathSMCConfig,
    sample_slot_existence_posterior,
)
from sbulm.simulation import (
    DetectionCorruptionConfig,
    simulate_crossing,
    simulate_slot_existence_observations,
)
from sbulm.slot_existence import (
    ABSENT,
    SlotExistenceMaternModel,
    enumerate_slot_existence_posterior,
    slot_event_marginals,
)
from sbulm.tsbm_reference import (
    compare_tsbm_association_samples,
    finite_sample_baseline,
)


def test_birth_death_smc_recovers_enumerated_posterior() -> None:
    config = physical_crossing_config(
        n_frames=4,
        central_separation_um=0.0,
        localization_std_um=5.0,
    )
    _, clean = simulate_crossing(config, seed=2)
    truth_mask = np.asarray(
        [[True, True], [True, True], [False, True], [False, True]]
    )
    detections = simulate_slot_existence_observations(
        clean,
        truth_mask,
        DetectionCorruptionConfig(
            detection_probability=0.82,
            expected_clutter_per_frame=0.20,
            roi_lower=(-0.04, -0.04),
            roi_upper=(0.04, 0.04),
            anchor_first_frame=False,
            max_clutter_per_frame=1,
        ),
        seed=1,
    )
    calibration = calibrate_matern_from_coordinate_std(
        crossing_design_coordinate_std(config), rms_speed=6.84
    )
    kwargs = {
        "n_slots": 2,
        "localization_std": config.localization_std,
        "coordinate_std": calibration.coordinate_std,
        "length_scale": calibration.length_scale,
        "detection_probability": 0.82,
        "expected_clutter_per_frame": 0.20,
        "roi_lower": (-0.04, -0.04),
        "roi_upper": (0.04, 0.04),
    }
    exact = enumerate_slot_existence_posterior(detections, **kwargs)
    controls = BirthDeathSMCConfig(
        n_particles=2_048,
        target_ess_fraction=0.80,
        mh_sweeps=6,
        final_mh_sweeps=8,
        seed=19,
    )
    result = sample_slot_existence_posterior(
        detections, **kwargs, config=controls
    )
    comparison = compare_tsbm_association_samples(result.associations, exact)
    finite_floor = finite_sample_baseline(
        exact, n_samples=controls.n_particles, n_replicates=200, seed=8
    )
    assert comparison.invalid_sample_fraction == 0.0
    assert comparison.total_variation_distance <= max(
        0.06, 3.0 * finite_floor.total_variation_q95
    )
    assert comparison.exact_mass_covered_by_empirical_support >= 0.90
    assert result.stages[-1].beta == 1.0
    np.testing.assert_array_equal(result.existence_masks, result.associations != ABSENT)
    np.testing.assert_array_equal(
        np.isnan(result.latent_paths[..., 0]), ~result.existence_masks
    )

    exact_events = slot_event_marginals(
        exact.associations,
        detections,
        probabilities=exact.probabilities,
    )
    sampled_events = slot_event_marginals(result.associations, detections)
    assert np.max(
        np.abs(
            sampled_events.existence_probabilities
            - exact_events.existence_probabilities
        )
    ) < 0.06
    assert np.max(
        np.abs(
            sampled_events.birth_time_probabilities
            - exact_events.birth_time_probabilities
        )
    ) < 0.06
    assert np.max(
        np.abs(
            sampled_events.death_time_probabilities
            - exact_events.death_time_probabilities
        )
    ) < 0.06

    model = SlotExistenceMaternModel(detections, **kwargs)
    exact_log_evidence = float(
        logsumexp(model.log_joint_density(exact.associations))
    )
    assert result.log_evidence_estimate == pytest.approx(
        exact_log_evidence, abs=0.30
    )
