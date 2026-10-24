"""Readers for lightweight trajectory exports from ULM pipelines."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np

from .data import Track, TrackCollection


def _read_json(path: Path) -> Any:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_track(item: Any, default_id: int) -> Track:
    if isinstance(item, dict):
        track_id = item.get("track_id", item.get("id", default_id))
        frames_raw = _first_present(item, ("frames", "frame", "frame_indices", "t"))
        positions_raw = _first_present(item, ("positions", "coordinates", "points", "xyz"))
        if positions_raw is None:
            x = _first_present(item, ("x", "X", "lateral"))
            z = _first_present(item, ("z", "Z", "axial", "y", "Y"))
            if x is not None and z is not None:
                positions_raw = np.column_stack([x, z])
        if positions_raw is None:
            raise ValueError(
                "track dictionary has no recognized positions; expected positions/points "
                "or coordinate arrays such as x and z"
            )
        positions = np.asarray(positions_raw, dtype=float)
        if positions.ndim != 2:
            raise ValueError("track positions must be a two-dimensional array")
        frames = (
            np.arange(positions.shape[0], dtype=np.int64)
            if frames_raw is None
            else np.asarray(frames_raw, dtype=np.int64)
        )
        consumed = {
            "track_id",
            "id",
            "frames",
            "frame",
            "frame_indices",
            "t",
            "positions",
            "coordinates",
            "points",
            "xyz",
            "x",
            "X",
            "lateral",
            "z",
            "Z",
            "axial",
            "y",
            "Y",
        }
        metadata = {key: value for key, value in item.items() if key not in consumed}
        return Track(track_id=track_id, frames=frames, positions=positions, metadata=metadata)

    array = np.asarray(item, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("array tracks must have columns [frame, coordinate_1, coordinate_2, ...]")
    return Track(
        track_id=default_id,
        frames=array[:, 0].astype(np.int64),
        positions=array[:, 1:],
    )


def load_track_json(path: str | Path) -> TrackCollection:
    """Load a JSON or JSON.XZ track export using common ULM layouts.

    Supported top levels are a list of tracks or a dictionary containing a
    ``tracks``/``trajectories`` list. The reader is intentionally strict after
    schema detection so that silent coordinate mistakes cannot pass unnoticed.
    """

    path = Path(path)
    payload = _read_json(path)
    metadata: dict[str, Any] = {"source": str(path)}
    if isinstance(payload, dict):
        tracks_payload = _first_present(payload, ("tracks", "trajectories", "data"))
        if tracks_payload is None:
            raise ValueError("JSON object has no tracks, trajectories, or data field")
        metadata.update({key: value for key, value in payload.items() if value is not tracks_payload})
    elif isinstance(payload, list):
        tracks_payload = payload
    else:
        raise TypeError("track JSON must contain a list or an object")
    if not isinstance(tracks_payload, list):
        raise TypeError("the detected track container is not a list")
    tracks = tuple(_parse_track(item, index) for index, item in enumerate(tracks_payload))
    return TrackCollection(tracks=tracks, metadata=metadata)
