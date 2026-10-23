import json
import lzma

import numpy as np

from sbulm.io import load_track_json


def test_load_compressed_track_json(tmp_path) -> None:
    payload = {
        "tracks": [
            {"id": "a", "frames": [0, 1, 2], "x": [0.0, 0.1, 0.2], "z": [1, 1, 1]},
            {"id": "b", "frames": [1, 2], "positions": [[0.8, 1.2], [0.7, 1.2]]},
        ],
        "units": "mm",
    }
    path = tmp_path / "tracks.json.xz"
    with lzma.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    collection = load_track_json(path)
    assert len(collection.tracks) == 2
    assert collection.metadata["units"] == "mm"
    sequence = collection.to_detections(frame_period=0.001)
    assert sequence.cardinalities == (1, 2, 2)
    np.testing.assert_allclose(sequence.times, [0.0, 0.001, 0.002])

