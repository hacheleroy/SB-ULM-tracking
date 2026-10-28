"""Independent positive partition check against direct enumeration."""

import numpy as np
from sbulm.wp11am_positive_partition import positive_four_slot_partition
from scipy.special import logsumexp

from sbulm.slot_existence import enumerate_slot_existence_support
from sbulm.wp11v_tempering_mode_audit import _ExactComputer, _toy_model


def test_positive_partition_matches_full_toy_enumeration_at_three_betas() -> None:
    model = _toy_model()
    support, _ = enumerate_slot_existence_support(model)
    scores = (model.log_existence_prior(support)
              + model.log_count_density(support, include_poisson_void=True))
    gp = model.log_marginal_density(support)
    computer = _ExactComputer(model, (0, 2))
    for beta in (0.0, 0.5, 1.0):
        expected = float(logsumexp(scores + beta * gp))
        double = positive_four_slot_partition(computer, beta, extended=False)
        extended = positive_four_slot_partition(computer, beta)
        np.testing.assert_allclose(extended["log_partition"], expected, atol=1e-11)
        np.testing.assert_allclose(double["log_partition"], extended["log_partition"],
                                   atol=1e-11)
        assert extended["normalized_four_slot_mass"] > 0


def test_positive_partition_requires_four_slots_and_beta_range() -> None:
    import pytest

    computer = _ExactComputer(_toy_model(), (0, 2))
    with pytest.raises(ValueError, match="beta"):
        positive_four_slot_partition(computer, -0.1)
