"""Core data structures shared by simulation, baselines, and future SB models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_float_array(value: Any, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class TrajectorySet:
    """Dense fixed-cardinality trajectories with shape ``(T, N, D)``."""

    times: FloatArray
    positions: FloatArray
    particle_ids: IntArray | None = None

    def __post_init__(self) -> None:
        times = _as_float_array(self.times, ndim=1, name="times")
        positions = _as_float_array(self.positions, ndim=3, name="positions")
        if positions.shape[0] != times.size:
            raise ValueError("positions and times have different frame counts")
        if times.size < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing and contain at least two frames")
        particle_ids = (
            np.arange(positions.shape[1], dtype=np.int64)
            if self.particle_ids is None
            else np.asarray(self.particle_ids, dtype=np.int64)
        )
        if particle_ids.shape != (positions.shape[1],):
            raise ValueError("particle_ids must have shape (N,)")
        if np.unique(particle_ids).size != particle_ids.size:
            raise ValueError("particle_ids must be unique")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "particle_ids", particle_ids)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_particles(self) -> int:
        return self.positions.shape[1]

    @property
    def dimension(self) -> int:
        return self.positions.shape[2]


@dataclass(frozen=True)
class DetectionSequence:
    """Unordered detections for each frame.

    ``reference_ids`` are optional audit labels. In simulations they are exact
    identities; for real ULM data they may instead be pipeline-provided track
    identifiers and must not automatically be interpreted as ground truth.
    """

    times: FloatArray
    detections: tuple[FloatArray, ...]
    reference_ids: tuple[IntArray, ...] | None = None

    def __post_init__(self) -> None:
        times = _as_float_array(self.times, ndim=1, name="times")
        detections = tuple(
            _as_float_array(frame, ndim=2, name=f"detections[{index}]")
            for index, frame in enumerate(self.detections)
        )
        if len(detections) != times.size:
            raise ValueError("detections and times have different frame counts")
        dimensions = {frame.shape[1] for frame in detections}
        if len(dimensions) != 1:
            raise ValueError("all detection frames must have the same spatial dimension")
        if self.reference_ids is not None:
            ids = tuple(np.asarray(frame, dtype=np.int64) for frame in self.reference_ids)
            if len(ids) != len(detections):
                raise ValueError("reference_ids and detections have different frame counts")
            for index, (frame_ids, frame) in enumerate(zip(ids, detections, strict=True)):
                if frame_ids.shape != (frame.shape[0],):
                    raise ValueError(f"reference_ids[{index}] has the wrong shape")
            object.__setattr__(self, "reference_ids", ids)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "detections", detections)

    @property
    def n_frames(self) -> int:
        return len(self.detections)

    @property
    def dimension(self) -> int:
        return self.detections[0].shape[1]

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(frame.shape[0] for frame in self.detections)

    @property
    def has_constant_cardinality(self) -> bool:
        return len(set(self.cardinalities)) == 1


@dataclass(frozen=True)
class Track:
    """One possibly partial trajectory imported from an external ULM pipeline."""

    track_id: int | str
    frames: IntArray
    positions: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames, dtype=np.int64)
        positions = _as_float_array(self.positions, ndim=2, name="positions")
        if frames.ndim != 1 or frames.size != positions.shape[0]:
            raise ValueError("frames must have one entry per position")
        if frames.size == 0 or not np.all(np.diff(frames) > 0):
            raise ValueError("track frames must be non-empty, unique, and increasing")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class TrackCollection:
    tracks: tuple[Track, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if tracks:
            dimensions = {track.positions.shape[1] for track in tracks}
            if len(dimensions) != 1:
                raise ValueError("all tracks must have the same spatial dimension")
        object.__setattr__(self, "tracks", tracks)

    @property
    def dimension(self) -> int | None:
        return None if not self.tracks else self.tracks[0].positions.shape[1]

    def to_detections(self, *, frame_period: float = 1.0) -> DetectionSequence:
        """Convert tracks into per-frame detections while retaining reference IDs."""

        if not self.tracks:
            raise ValueError("cannot convert an empty track collection")
        first = min(int(track.frames[0]) for track in self.tracks)
        last = max(int(track.frames[-1]) for track in self.tracks)
        frames = np.arange(first, last + 1, dtype=np.int64)
        detections: list[list[FloatArray]] = [[] for _ in frames]
        ids: list[list[int]] = [[] for _ in frames]
        stable_ids = {track.track_id: index for index, track in enumerate(self.tracks)}
        for track in self.tracks:
            for frame, position in zip(track.frames, track.positions, strict=True):
                slot = int(frame) - first
                detections[slot].append(position)
                ids[slot].append(stable_ids[track.track_id])
        dimension = int(self.dimension or 2)
        detection_arrays = tuple(
            np.asarray(frame, dtype=float).reshape((-1, dimension)) for frame in detections
        )
        id_arrays = tuple(np.asarray(frame, dtype=np.int64) for frame in ids)
        return DetectionSequence(
            times=(frames - first).astype(float) * frame_period,
            detections=detection_arrays,
            reference_ids=id_arrays,
        )


def stack_assigned_detections(
    sequence: DetectionSequence, associations: Iterable[IntArray]
) -> FloatArray:
    """Return detections reordered as ``(T, N, D)`` by an association path."""

    association_tuple = tuple(np.asarray(item, dtype=np.int64) for item in associations)
    if len(association_tuple) != sequence.n_frames:
        raise ValueError("one association vector is required per frame")
    return np.stack(
        [frame[index] for frame, index in zip(sequence.detections, association_tuple, strict=True)]
    )
