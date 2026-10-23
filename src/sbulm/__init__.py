"""Posterior benchmarks for uncertain ULM microbubble trajectories."""

from .calibration import (
    MaternVelocityCalibration,
    calibrate_matern_from_coordinate_std,
    calibrate_matern_length_scale,
    crossing_design_coordinate_std,
    enumerate_velocity_calibrated_target,
    physical_crossing_config,
)
from .data import DetectionSequence, Track, TrackCollection, TrajectorySet
from .existence_smc import (
    BirthDeathSMCConfig,
    BirthDeathSMCResult,
    sample_slot_existence_posterior,
)
from .inference import (
    AssociationPosterior,
    enumerate_association_posterior,
    enumerate_association_support,
)
from .localization import (
    MaternLocalizationModel,
    enumerate_matern_localization_posterior,
    unordered_gaussian_localization_gradient,
    unordered_gaussian_localization_potential,
)
from .null_association import (
    NullAssociationMaternModel,
    NullAssociationPosterior,
    enumerate_frame_null_associations,
    enumerate_null_association_posterior,
    null_and_clutter_marginals,
    reference_null_association,
)
from .null_smc import (
    NullAssociationSMCResult,
    sample_null_association_posterior,
)
from .simulation import (
    CrossingConfig,
    DetectionCorruptionConfig,
    MultiSlotCrossingConfig,
    corrupt_detections_with_misses_and_clutter,
    simulate_crossing,
    simulate_multislot_crossing,
    simulate_slot_existence_observations,
)
from .slot_existence import (
    ABSENT,
    MISSED,
    BirthDeathPrior,
    SlotEventMarginals,
    SlotExistenceMaternModel,
    SlotExistencePosterior,
    enumerate_contiguous_existence_masks,
    enumerate_frame_slot_associations,
    enumerate_slot_existence_posterior,
    enumerate_slot_existence_support,
    reference_slot_association,
    slot_event_marginals,
    slot_existence_support_size,
)
from .smc import (
    AnnealedSMCConfig,
    MaternSMCResult,
    sample_matern_localization_posterior,
)

__all__ = [
    "ABSENT",
    "MISSED",
    "AnnealedSMCConfig",
    "AssociationPosterior",
    "BirthDeathPrior",
    "BirthDeathSMCConfig",
    "BirthDeathSMCResult",
    "CrossingConfig",
    "DetectionCorruptionConfig",
    "DetectionSequence",
    "MaternLocalizationModel",
    "MaternSMCResult",
    "MaternVelocityCalibration",
    "MultiSlotCrossingConfig",
    "NullAssociationMaternModel",
    "NullAssociationPosterior",
    "NullAssociationSMCResult",
    "SlotEventMarginals",
    "SlotExistenceMaternModel",
    "SlotExistencePosterior",
    "Track",
    "TrackCollection",
    "TrajectorySet",
    "calibrate_matern_from_coordinate_std",
    "calibrate_matern_length_scale",
    "corrupt_detections_with_misses_and_clutter",
    "crossing_design_coordinate_std",
    "enumerate_association_posterior",
    "enumerate_association_support",
    "enumerate_contiguous_existence_masks",
    "enumerate_frame_null_associations",
    "enumerate_frame_slot_associations",
    "enumerate_matern_localization_posterior",
    "enumerate_null_association_posterior",
    "enumerate_slot_existence_posterior",
    "enumerate_slot_existence_support",
    "enumerate_velocity_calibrated_target",
    "null_and_clutter_marginals",
    "physical_crossing_config",
    "reference_null_association",
    "reference_slot_association",
    "sample_matern_localization_posterior",
    "sample_null_association_posterior",
    "sample_slot_existence_posterior",
    "simulate_crossing",
    "simulate_multislot_crossing",
    "simulate_slot_existence_observations",
    "slot_event_marginals",
    "slot_existence_support_size",
    "unordered_gaussian_localization_gradient",
    "unordered_gaussian_localization_potential",
]

__version__ = "0.10.0"

