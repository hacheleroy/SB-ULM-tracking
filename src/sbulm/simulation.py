"""Controlled ULM localization-level simulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DetectionSequence, FloatArray, IntArray, TrajectorySet


@dataclass(frozen=True)
class CrossingConfig:
    """Configuration for two bubbles passing through a vessel crossing.

    Bubble 0 follows the horizontal branch and bubble 1 the vertical branch.
    ``central_separation`` is their distance in the central acquired frame.
    It therefore controls the observable association difficulty directly.
    """

    n_frames: int = 9
    frame_period: float = 1.0
    half_span: float = 1.0
    central_separation: float = 0.05
    localization_std: float = 0.075

    def __post_init__(self) -> None:
        if self.n_frames < 4:
            raise ValueError("n_frames must be at least four")
        if self.frame_period <= 0 or self.half_span <= 0:
            raise ValueError("frame_period and half_span must be positive")
        if self.central_separation < 0:
            raise ValueError("central_separation cannot be negative")
        if self.localization_std < 0:
            raise ValueError("localization_std cannot be negative")


@dataclass(frozen=True)
class MultiSlotCrossingConfig:
    """Configuration for several bubbles traversing intersecting branches.

    Branch directions are evenly spaced over a half-circle.  The optional
    ``central_separation`` offsets neighboring slots along their own branch at
    the temporal midpoint; zero creates a simultaneous multi-way crossing.
    """

    n_slots: int = 3
    n_frames: int = 5
    frame_period: float = 1.0
    half_span: float = 1.0
    central_separation: float = 0.0
    localization_std: float = 0.075

    def __post_init__(self) -> None:
        if self.n_slots < 2:
            raise ValueError("n_slots must be at least two")
        if self.n_frames < 3:
            raise ValueError("n_frames must be at least three")
        if self.frame_period <= 0 or self.half_span <= 0:
            raise ValueError("frame_period and half_span must be positive")
        if self.central_separation < 0:
            raise ValueError("central_separation cannot be negative")
        if self.localization_std < 0:
            raise ValueError("localization_std cannot be negative")


@dataclass(frozen=True)
class DetectionCorruptionConfig:
    """Missed-detection and uniform-clutter controls for a localization sequence.

    The first frame can be kept complete and clutter-free to anchor otherwise
    arbitrary particle labels. ``max_clutter_per_frame`` is a simulator guard
    used by enumerable calibration problems; conditioning on the realized
    detection count means that this truncation does not change association
    probabilities within a frame.
    """

    detection_probability: float = 0.80
    expected_clutter_per_frame: float = 0.50
    roi_lower: tuple[float, ...] = (-0.04, -0.04)
    roi_upper: tuple[float, ...] = (0.04, 0.04)
    anchor_first_frame: bool = True
    max_clutter_per_frame: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.detection_probability < 1:
            raise ValueError("detection_probability must lie strictly between zero and one")
        if self.expected_clutter_per_frame <= 0 or not np.isfinite(self.expected_clutter_per_frame):
            raise ValueError("expected_clutter_per_frame must be finite and positive")
        lower = np.asarray(self.roi_lower, dtype=float)
        upper = np.asarray(self.roi_upper, dtype=float)
        if lower.ndim != 1 or lower.size == 0 or upper.shape != lower.shape:
            raise ValueError("ROI bounds must be non-empty vectors with matching shapes")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("ROI bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("every ROI upper bound must exceed its lower bound")
        if self.max_clutter_per_frame is not None and self.max_clutter_per_frame < 0:
            raise ValueError("max_clutter_per_frame must be non-negative")


def simulate_crossing(
    config: CrossingConfig | None = None, *, seed: int = 0
) -> tuple[TrajectorySet, DetectionSequence]:
    """Generate two latent trajectories and shuffled noisy localizations.

    The paths lie on two orthogonal vessel branches. Bubble 0 reaches the
    crossing in the central frame; bubble 1 is displaced along its branch by
    ``central_separation``. At zero separation, both bubbles occupy the same
    latent position in that frame.
    """

    if config is None:
        config = CrossingConfig()
    rng = np.random.default_rng(seed)
    times = np.arange(config.n_frames, dtype=float) * config.frame_period
    normalized = np.linspace(-1.0, 1.0, config.n_frames)

    progress = config.half_span * normalized
    horizontal_path = np.column_stack([progress, np.zeros_like(progress)])
    vertical_path = np.column_stack(
        [np.zeros_like(progress), progress + config.central_separation]
    )
    positions = np.stack([horizontal_path, vertical_path], axis=1)
    truth = TrajectorySet(times=times, positions=positions)

    noisy = positions + rng.normal(scale=config.localization_std, size=positions.shape)
    detection_frames = []
    reference_ids = []
    for frame in noisy:
        permutation = rng.permutation(2)
        detection_frames.append(frame[permutation])
        reference_ids.append(permutation.astype(np.int64))
    detections = DetectionSequence(
        times=times,
        detections=tuple(detection_frames),
        reference_ids=tuple(reference_ids),
    )
    return truth, detections


def simulate_multislot_crossing(
    config: MultiSlotCrossingConfig | None = None, *, seed: int = 0
) -> tuple[TrajectorySet, DetectionSequence]:
    """Generate shuffled localizations on several intersecting straight paths."""

    controls = MultiSlotCrossingConfig() if config is None else config
    rng = np.random.default_rng(seed)
    times = np.arange(controls.n_frames, dtype=float) * controls.frame_period
    progress = np.linspace(-controls.half_span, controls.half_span, controls.n_frames)
    angles = np.linspace(0.0, np.pi, controls.n_slots, endpoint=False)
    offsets = controls.central_separation * (
        np.arange(controls.n_slots, dtype=float) - 0.5 * (controls.n_slots - 1)
    )
    positions = np.empty((controls.n_frames, controls.n_slots, 2), dtype=float)
    for slot, (angle, offset) in enumerate(zip(angles, offsets, strict=True)):
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        positions[:, slot] = (progress + offset)[:, None] * direction

    truth = TrajectorySet(times=times, positions=positions)
    noisy = positions + rng.normal(scale=controls.localization_std, size=positions.shape)
    detection_frames: list[FloatArray] = []
    reference_ids: list[IntArray] = []
    for frame in noisy:
        permutation = rng.permutation(controls.n_slots)
        detection_frames.append(np.asarray(frame[permutation], dtype=float))
        reference_ids.append(permutation.astype(np.int64))
    detections = DetectionSequence(
        times=times,
        detections=tuple(detection_frames),
        reference_ids=tuple(reference_ids),
    )
    return truth, detections


def corrupt_detections_with_misses_and_clutter(
    sequence: DetectionSequence,
    config: DetectionCorruptionConfig | None = None,
    *,
    seed: int = 0,
) -> DetectionSequence:
    """Add independent missed detections and uniform Poisson clutter.

    Simulation-only ``reference_ids`` retain the source particle identity;
    clutter receives the sentinel ID ``-1``. When ``anchor_first_frame`` is
    enabled, frame zero contains every true detection, no clutter, and is
    ordered by particle ID. All later frames are shuffled after corruption.
    """

    controls = DetectionCorruptionConfig() if config is None else config
    if sequence.reference_ids is None:
        raise ValueError("reference_ids are required to corrupt labelled simulations")
    if sequence.dimension != len(controls.roi_lower):
        raise ValueError("ROI dimension must match the detection dimension")

    nonnegative_ids = np.concatenate(
        [frame_ids[frame_ids >= 0] for frame_ids in sequence.reference_ids]
    )
    if nonnegative_ids.size == 0:
        raise ValueError("at least one non-clutter reference ID is required")
    unique_ids = np.unique(nonnegative_ids)
    expected_ids = np.arange(unique_ids.size, dtype=np.int64)
    if not np.array_equal(unique_ids, expected_ids):
        raise ValueError("non-clutter reference IDs must be consecutive from zero")
    n_particles = expected_ids.size

    lower = np.asarray(controls.roi_lower, dtype=float)
    upper = np.asarray(controls.roi_upper, dtype=float)
    for frame in sequence.detections:
        if np.any(frame < lower) or np.any(frame > upper):
            raise ValueError("the ROI must contain every source detection")

    rng = np.random.default_rng(seed)
    corrupted_frames: list[FloatArray] = []
    corrupted_ids: list[IntArray] = []
    for time_index, (frame, frame_ids) in enumerate(
        zip(sequence.detections, sequence.reference_ids, strict=True)
    ):
        if np.unique(frame_ids[frame_ids >= 0]).size != np.count_nonzero(frame_ids >= 0):
            raise ValueError("a source frame contains a duplicate particle ID")

        if time_index == 0 and controls.anchor_first_frame:
            lookup = {int(identity): index for index, identity in enumerate(frame_ids)}
            if any(identity not in lookup for identity in range(n_particles)):
                raise ValueError("the anchor frame must contain every particle")
            indices = np.asarray([lookup[identity] for identity in range(n_particles)])
            corrupted_frames.append(np.asarray(frame[indices], dtype=float))
            corrupted_ids.append(expected_ids.copy())
            continue

        true_mask = frame_ids >= 0
        true_frame = frame[true_mask]
        true_ids = frame_ids[true_mask]
        retained = rng.random(true_ids.size) < controls.detection_probability
        kept_frame = true_frame[retained]
        kept_ids = true_ids[retained]

        n_clutter = int(rng.poisson(controls.expected_clutter_per_frame))
        if controls.max_clutter_per_frame is not None:
            n_clutter = min(n_clutter, controls.max_clutter_per_frame)
        clutter = rng.uniform(lower, upper, size=(n_clutter, sequence.dimension))
        clutter_ids = np.full(n_clutter, -1, dtype=np.int64)

        combined = np.concatenate([kept_frame, clutter], axis=0)
        combined_ids = np.concatenate([kept_ids, clutter_ids])
        permutation = rng.permutation(combined.shape[0])
        corrupted_frames.append(np.asarray(combined[permutation], dtype=float))
        corrupted_ids.append(np.asarray(combined_ids[permutation], dtype=np.int64))

    return DetectionSequence(
        times=sequence.times,
        detections=tuple(corrupted_frames),
        reference_ids=tuple(corrupted_ids),
    )


def simulate_slot_existence_observations(
    sequence: DetectionSequence,
    existence_mask: np.ndarray,
    config: DetectionCorruptionConfig | None = None,
    *,
    seed: int = 0,
) -> DetectionSequence:
    """Apply explicit slot births/deaths, misses and clutter to labelled data.

    ``sequence`` supplies one clean localization per slot and frame.  A true
    localization is eligible for detection only while its slot exists.  The
    first frame is deliberately *not* anchored: exchangeable slot labels and
    birth-at-zero uncertainty remain part of the posterior.
    """

    controls = DetectionCorruptionConfig(anchor_first_frame=False) if config is None else config
    if controls.anchor_first_frame:
        raise ValueError("slot-existence simulation cannot anchor the first frame")
    if sequence.reference_ids is None:
        raise ValueError("reference_ids are required for labelled slot simulation")
    if sequence.dimension != len(controls.roi_lower):
        raise ValueError("ROI dimension must match the detection dimension")

    raw_mask = np.asarray(existence_mask)
    if raw_mask.ndim != 2 or raw_mask.shape[0] != sequence.n_frames:
        raise ValueError("existence_mask must have shape (T, K)")
    mask = raw_mask.astype(bool, copy=False)
    if not np.array_equal(raw_mask, mask):
        raise ValueError("existence_mask must be boolean or contain only zero and one")
    n_slots = mask.shape[1]
    if n_slots < 1:
        raise ValueError("existence_mask must contain at least one slot")

    # Validate the one-birth/one-death structure without coupling this module
    # to the inference implementation.
    for slot in range(n_slots):
        alive = np.flatnonzero(mask[:, slot])
        if alive.size and not np.all(mask[alive[0] : alive[-1] + 1, slot]):
            raise ValueError("each slot must have one contiguous existence interval")

    lower = np.asarray(controls.roi_lower, dtype=float)
    upper = np.asarray(controls.roi_upper, dtype=float)
    rng = np.random.default_rng(seed)
    corrupted_frames: list[FloatArray] = []
    corrupted_ids: list[IntArray] = []
    expected_ids = np.arange(n_slots, dtype=np.int64)

    for time_index, (frame, frame_ids) in enumerate(
        zip(sequence.detections, sequence.reference_ids, strict=True)
    ):
        if np.any(frame < lower) or np.any(frame > upper):
            raise ValueError("the ROI must contain every source detection")
        lookup: dict[int, int] = {}
        for detection_index, identity in enumerate(frame_ids):
            if identity < 0:
                continue
            if int(identity) in lookup:
                raise ValueError("a source frame contains a duplicate particle ID")
            lookup[int(identity)] = detection_index
        if any(int(identity) not in lookup for identity in expected_ids):
            raise ValueError("every source frame must contain every slot exactly once")

        active_ids = expected_ids[mask[time_index]]
        active_indices = np.asarray(
            [lookup[int(identity)] for identity in active_ids], dtype=np.int64
        )
        active_frame = frame[active_indices]
        retained = rng.random(active_ids.size) < controls.detection_probability
        kept_frame = active_frame[retained]
        kept_ids = active_ids[retained]

        n_clutter = int(rng.poisson(controls.expected_clutter_per_frame))
        if controls.max_clutter_per_frame is not None:
            n_clutter = min(n_clutter, controls.max_clutter_per_frame)
        clutter = rng.uniform(lower, upper, size=(n_clutter, sequence.dimension))
        clutter_ids = np.full(n_clutter, -1, dtype=np.int64)

        combined = np.concatenate([kept_frame, clutter], axis=0)
        combined_ids = np.concatenate([kept_ids, clutter_ids])
        permutation = rng.permutation(combined.shape[0])
        corrupted_frames.append(np.asarray(combined[permutation], dtype=float))
        corrupted_ids.append(np.asarray(combined_ids[permutation], dtype=np.int64))

    return DetectionSequence(
        times=sequence.times,
        detections=tuple(corrupted_frames),
        reference_ids=tuple(corrupted_ids),
    )
