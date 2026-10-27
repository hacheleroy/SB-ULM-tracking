"""Retrospective decomposition of WP11ab's exposed crowded-case error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .wp7_convergence_audit import _save

WP11AB_SHA256 = "ab05dec063aa142e0504cf4a6c23a3abfb0db58730d569556f1e288251355c4c"
PRIMARY_SHA256 = "52f13725b9f7dd8851b44ee97025afebf98f042791b111cf508dcbd5c0c0c14c"
SEEDS = tuple(range(1300, 1306))


def _pinned(path: Path, digest: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"pinned input changed: {path}")
    return json.loads(raw)


def diagnose(comparison: Path, reference: Path, runs: Path, output: Path) -> dict[str, Any]:
    previous = _pinned(comparison, WP11AB_SHA256)
    exact = _pinned(reference, PRIMARY_SHA256)["reference"]
    if (previous["status"] != "two_exposed_case_gate_failed_stop"
            or set(previous["cases"]["primary_19"]["runs"]) != set(map(str, SEEDS))
            or not exact["valid"]):
        raise ValueError("exposed diagnostic inputs are incomplete")
    rows: dict[str, Any] = {}
    for seed in SEEDS:
        pair = previous["cases"]["primary_19"]["runs"][str(seed)]
        run = json.loads((runs / f"primary_19_{seed}.json").read_text())
        if (run["protocol"] != previous["protocol"] or run["seed"] != seed
                or run["case"] != "primary_19"
                or not np.array_equal(
                    np.asarray(run["joint_alive_by_sixth_by_last_clutter"]),
                    np.asarray(pair["candidate"]["joint_alive_by_sixth_by_last_clutter"]),
                )):
            raise ValueError("archived stage checkpoint does not match comparison")
        arms = {}
        for arm in ("control", "candidate"):
            if arm == "candidate":
                joint = np.asarray(run["joint_alive_by_sixth_by_last_clutter"])
            else:
                # Aggregated control retains the signed errors but no raw joint.
                joint = None
            error = pair[arm]["errors"]
            band_signed = np.asarray(error["alive_band_signed_errors"])
            band_tv = float(np.abs(band_signed).sum() / 2)
            full_tv = float(error["full_joint_tv"])
            if joint is not None:
                collapsed = joint.sum(axis=(1, 2))
                actual = np.array((collapsed[:21].sum(), collapsed[21],
                                   collapsed[22:].sum()))
                if not np.allclose(actual - exact["band_mass"], band_signed, atol=1e-12):
                    raise ValueError("alive-band decomposition differs from exact joint")
                if band_tv > full_tv + 1e-12:
                    raise ValueError("marginal TV exceeds full-joint TV")
            arms[arm] = {
                "signed_alive_band_errors": band_signed.tolist(),
                "alive_band_tv_lower_bound": band_tv,
                "full_joint_tv": full_tv,
                "fraction_of_full_tv_lower_bounded_by_alive_band_tv": band_tv / full_tv,
                "sixth_clutter_one_signed_error": error["sixth_clutter_signed_errors"][1],
                "last_clutter_one_signed_error": error["last_clutter_signed_errors"][1],
            }
        late = [stage for stage in run["stages"] if stage["beta"] >= .75]
        rows[str(seed)] = {
            "arms": arms,
            "late_stage_count_signature_change_rate_mean": float(np.mean([
                stage["paired_slot_count_signature_change_rate"] for stage in late
            ])),
            "late_stage_count_signature_change_rate_last": late[-1][
                "paired_slot_count_signature_change_rate"
            ],
        }
    severe = [seed for seed, row in rows.items() if row["arms"]["candidate"][
        "alive_band_tv_lower_bound"] >= .10]
    result = {
        "experiment": "wp11ac_retrospective_count_band_diagnostic",
        "inputs_sha256": {"wp11ab": WP11AB_SHA256, "primary_exact": PRIMARY_SHA256},
        "exact_alive_band_mass": exact["band_mass"],
        "bands": "low <=20, middle 21, high >=22 alive slot-frames",
        "seeds": rows,
        "candidate_seeds_with_band_tv_at_least_point_10": severe,
        "interpretation": "marginal TV is a rigorous lower bound on full-joint TV; "
                          "stage count-signature changes do not establish between-band flow",
        "status": "retrospective_exposed_diagnostic_only",
    }
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path,
                        default=Path("results/wp11ab_late_pair_window.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/wp11t_subset_exact_primary.json"))
    parser.add_argument("--runs", type=Path, default=Path("results/wp11ab_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wp11ac_count_band_diagnostic.json"))
    args = parser.parse_args()
    print(diagnose(args.comparison, args.reference, args.runs, args.output)["status"])


if __name__ == "__main__":
    main()
