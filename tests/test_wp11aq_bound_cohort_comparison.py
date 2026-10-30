"""Binding and decision checks for the prospective WP11al cohort runner."""

import tempfile
import unittest
from pathlib import Path

from sbulm.wp11aq_bound_cohort_comparison import (
    CASE_BINDINGS, _case_decision, _cohort_decision, _inputs, _next_job,
    _schedule, aggregate,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/wp11ap_crowded_cohort_manifest.json"
LEDGER = ROOT / "results/wp11ap_geometry_ledger.json"
SCREENS = ROOT / "results/wp11ap_screens"
REFERENCES = ROOT / "results/wp11ap_references"
RULE = ROOT / "docs/wp11al_conditional_cohort_gate.md"


class BoundCohortTests(unittest.TestCase):
    def test_pinned_inputs_and_no_partial_decision(self) -> None:
        cases = _inputs(MANIFEST, LEDGER, SCREENS, REFERENCES, RULE)
        self.assertEqual(list(cases), list(CASE_BINDINGS))
        self.assertEqual(len(_schedule()), 24)
        self.assertEqual(_schedule()[0], ("crowded_18", "control", 1800))
        self.assertEqual(_schedule()[-1], ("crowded_19_second", "candidate", 1823))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertEqual(_next_job(directory, cases), _schedule()[0])
            with self.assertRaisesRegex(ValueError, "all 24 frozen jobs"):
                aggregate(directory, directory / "aggregate.json", MANIFEST,
                          LEDGER, SCREENS, REFERENCES, RULE)
            changed = directory / "manifest.json"
            changed.write_bytes(MANIFEST.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "frozen input hash changed"):
                _inputs(changed, LEDGER, SCREENS, REFERENCES, RULE)

    def test_frozen_case_and_pooled_replication_boundaries(self) -> None:
        def row(full: float, band: float, alive: float, cell: float,
                clutter: float, cpu: float) -> dict:
            return {"cpu_seconds": cpu, "errors": {
                "full_joint_tv": full, "band_by_sixth_tv": band,
                "maximum_alive_band_error": alive, "alive_total_tv": alive,
                "maximum_joint_cell_error": cell,
                "sixth_clutter_mean_absolute_error": clutter,
                "last_clutter_mean_absolute_error": clutter,
                "sixth_clutter_signed_errors": [0.0, clutter],
                "last_clutter_signed_errors": [0.0, clutter],
            }}

        control = [row(.2, .12, .09, .05, .07, 100) for _ in range(4)]
        candidate = [row(.1, .08, .06, .03, .04, 110) for _ in range(4)]
        comparison = _case_decision(CASE_BINDINGS["crowded_18"]["seeds"],
                                    control, candidate)
        self.assertTrue(comparison["passed"])
        self.assertEqual(set(comparison["per_seed_modes"]),
                         {"1800", "1801", "1802", "1803"})
        self.assertTrue(_cohort_decision({name: comparison for name in CASE_BINDINGS})[
            "passed"
        ])
        insufficient_wins = {"passed": True, "paired_full_joint_wins": 3,
                             "per_seed_modes": {str(i): True for i in range(4)}}
        pooled = _cohort_decision({name: insufficient_wins for name in CASE_BINDINGS})
        self.assertEqual(pooled["pooled_full_joint_wins"], 9)
        self.assertFalse(pooled["gates"]["pooled_full_joint_wins"])
        insufficient_modes = {"passed": True, "paired_full_joint_wins": 4,
                              "per_seed_modes": {str(i): i < 3 for i in range(4)}}
        pooled = _cohort_decision({name: insufficient_modes for name in CASE_BINDINGS})
        self.assertEqual(pooled["pooled_simultaneous_modes"], 9)
        self.assertFalse(pooled["gates"]["pooled_simultaneous_modes"])


if __name__ == "__main__":
    unittest.main()
