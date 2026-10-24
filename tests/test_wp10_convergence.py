"""Frozen controls for the WP10 hard-case convergence screen."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from sbulm.wp5_scaling_calibration import _controls
from sbulm.wp7_convergence_audit import _case_metadata
from sbulm.wp8_generalization import make_case
from sbulm.wp10_convergence import (
    ARMS,
    BASE,
    BASE_HASH,
    SEEDS,
    _load_or_initialize,
    _sample_controlled,
)


def test_arms_keep_the_same_target_and_move_mixture() -> None:
    assert SEEDS == (100, 200)
    assert [(arm.name, arm.n_particles, arm.mh_sweeps, arm.final_mh_sweeps) for arm in ARMS] == [
        ("double_particles", 16_384, 8, 8),
        ("double_rejuvenation", 8_192, 16, 16),
    ]
    assert _case_metadata(make_case(BASE))["detections_sha256"] == BASE_HASH
    baseline = _controls("paired", n_particles=8_192, seed=100)
    doubled = replace(baseline, mh_sweeps=16, final_mh_sweeps=16)
    differences = {key for key, value in asdict(baseline).items() if asdict(doubled)[key] != value}
    assert differences == {"mh_sweeps", "final_mh_sweeps"}


def test_checkpoint_and_small_rejuvenation_smoke(tmp_path) -> None:
    source = tmp_path / "wp8.json"
    source.write_text(Path("results/wp8_generalization.json").read_text(encoding="utf-8"))
    output = tmp_path / "wp10.json"
    fresh = _load_or_initialize(output, source)
    output.write_text(json.dumps(fresh), encoding="utf-8")
    assert _load_or_initialize(output, source) == fresh

    short_arm = replace(ARMS[1], n_particles=32)
    sampled, runtime = _sample_controlled(make_case(BASE), short_arm, seed=999)
    assert sampled.associations.shape == (32, 8, 4)
    assert runtime > 0
    assert sampled.config.mh_sweeps == 16
    assert sampled.config.final_mh_sweeps == 16
