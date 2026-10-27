"""Frozen independent four-slot tempering-stage count/clutter audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .wp8_generalization import (
    STAGGERED_TRUTH_MASK,
    GeneralizationRegime,
    _metadata,
    make_case,
)

SEED_PAIRS = ((101, 103), (107, 109), (113, 127), (131, 137),
              (139, 149), (151, 157), (163, 167), (173, 179))
SEEDS = (1400, 1401, 1402, 1403)
N_PARTICLES = 2_048
RUN_CPU_CAP = 750
IID_SEED = 34_621
IID_REPLICATES = 1_000
CASE_HASH: str | None = None  # pinned after deterministic cardinality-only selection
CASE_CARDINALITIES: tuple[int, ...] | None = None


def _select_case() -> tuple[int, GeneralizationRegime, Any]:
    """First prespecified geometry passing cardinality-only constraints."""

    for index, (trajectory_seed, corruption_seed) in enumerate(SEED_PAIRS):
        regime = GeneralizationRegime(
            name="wp11v_independent", detection_probability=0.50,
            expected_clutter_per_frame=0.35, central_separation=0.006,
            trajectory_seed=trajectory_seed, corruption_seed=corruption_seed,
            max_clutter_per_frame=2, truth_mask=STAGGERED_TRUTH_MASK,
        )
        case = make_case(regime)
        cards = tuple(case.model.sequence.cardinalities)
        if (11 <= sum(cards) <= 14 and cards[5] >= 1 and cards[7] >= 1
                and sum(cards[2:5]) >= 4):
            return index, regime, case
    raise RuntimeError("prespecified independent density ladder exhausted")


def _case() -> tuple[Any, dict[str, Any]]:
    index, regime, case = _select_case()
    metadata = json.loads(json.dumps(_metadata(regime, case)))
    info = metadata["model_and_detections"]
    if CASE_HASH is not None and info["detections_sha256"] != CASE_HASH:
        raise ValueError("selected independent observation hash changed")
    if CASE_CARDINALITIES is not None and tuple(info["detection_cardinalities"]) != (
        CASE_CARDINALITIES
    ):
        raise ValueError("selected independent cardinalities changed")
    return case, {"ladder_index": index, **metadata}


def _protocol() -> dict[str, Any]:
    return {
        "seed_pairs": [list(pair) for pair in SEED_PAIRS],
        "selection": "first 11..14 points, frame-six and last >=1, middle frames 3..5 >=4",
        "regime": {"detection_probability": 0.50, "expected_clutter_per_frame": 0.35,
                   "central_separation": 0.006, "max_clutter_per_frame": 2,
                   "truth_mask": [list(row) for row in STAGGERED_TRUTH_MASK]},
        "selected_hash": CASE_HASH,
        "selected_cardinalities": list(CASE_CARDINALITIES) if CASE_CARDINALITIES else None,
        "seeds": list(SEEDS), "n_particles": N_PARTICLES,
        "per_run_cpu_cap": RUN_CPU_CAP,
        "iid_seed": IID_SEED, "iid_replicates": IID_REPLICATES,
        "onset": "two consecutive rejuvenated band-by-sixth TVs > max(.05,2*iid_q95)",
        "reference": "exact beta-scaled four-slot squarefree subset convolution",
        "no_move_or_holdout": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("geometry",), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, metadata = _case()
    result = {"protocol": _protocol(), "case": metadata}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
