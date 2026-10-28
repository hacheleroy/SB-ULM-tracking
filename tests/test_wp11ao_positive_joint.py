"""Positive full-joint arithmetic against independent toy enumeration."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sbulm.slot_existence import ABSENT, enumerate_slot_existence_support
from sbulm.wp11ao_positive_joint import positive_four_slot_joint
from sbulm.wp11v_tempering_mode_audit import _ExactComputer, _toy_model


def test_positive_joint_matches_full_toy_enumeration() -> None:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    base = (model.log_existence_prior(support)
            + model.log_count_density(support, include_poisson_void=True))
    gp = model.log_marginal_density(support)
    alive = np.count_nonzero(support != ABSENT, axis=(1, 2))
    first = 1 - np.count_nonzero(support[:, 0] >= 0, axis=1)
    last = 1 - np.count_nonzero(support[:, 2] >= 0, axis=1)
    cells = 4 * alive + 2 * first + last
    computer = _ExactComputer(model, (0, 2))
    for beta in (0.0, .5, 1.0):
        scores = base + beta * gp
        logz = float(logsumexp(scores))
        expected = np.bincount(cells, weights=np.exp(scores - logz),
                               minlength=52).reshape(13, 2, 2)
        double = positive_four_slot_joint(computer, beta, extended=False)
        extended = positive_four_slot_joint(computer, beta)
        assert abs(extended["log_partition"] - logz) < 1e-11
        assert np.abs(np.asarray(extended["joint"]) - expected).sum() / 2 < 1e-11
        assert np.abs(np.asarray(double["joint"])
                      - np.asarray(extended["joint"])).sum() / 2 < 1e-11
        assert extended["minimum_cell"] >= 0


def test_stopped_case_audit_pins_full_joint_comparison() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "results/wp11ao_positive_joint_audit.json").read_text())
    stopped = audit["exposed"]["84"]["checks"]["1.0"]
    terminal = json.loads((root / "results/wp11an_terminal_validation_failure.json")
                          .read_text())
    assert stopped["paired_comparison"]["joint_tv"] < 1e-9
    assert stopped["double_extended_comparison"]["joint_tv"] < 1e-12
    assert terminal["joint"]["joint_tv"] > 1e-7
