"""Exact backward-weight and opt-in PGAS checks."""

import numpy as np
import pytest

from sbulm.conditional_pair_pg import ConditionalPairPath
from sbulm.existence_smc import BirthDeathSMCConfig
from sbulm.slot_existence import enumerate_slot_existence_support
from sbulm.wp11c_full_history_pilot import _pilot_model


@pytest.mark.parametrize("beta", [0.0, 0.41, 1.0])
def test_full_suffix_ratio_agrees_with_exact_target(beta: float) -> None:
    model = _pilot_model()
    support, _ = enumerate_slot_existence_support(model)
    checked = 0
    for reference in support[::17]:
        kernel = ConditionalPairPath(model, reference, (0, 1))
        for prefix in support[::19]:
            spliced = reference.copy()
            spliced[:2] = prefix[:2]
            suffix = kernel._reference_suffix_score(prefix, reference, 2, beta)
            try:
                model._validate_associations(spliced[None])
            except ValueError:
                assert suffix == -np.inf
                continue
            previous = None
            prefix_score = 0.0
            for frame in range(2):
                states, options, scores = kernel.options(prefix, previous, frame, beta)
                found = np.flatnonzero(np.all(options == prefix[frame], axis=1))
                assert len(found) == 1
                previous = int(states[found[0]])
                prefix_score += float(scores[found[0]])
            full = (model.log_existence_prior(spliced[None])[0]
                    + model.log_count_density(spliced[None], include_poisson_void=False)[0]
                    + beta * model.log_marginal_density(spliced[None])[0])
            assert suffix == pytest.approx(full - prefix_score, abs=1e-10)
            checked += 1
    assert checked > 10


def test_ancestor_sampling_is_opt_in() -> None:
    assert BirthDeathSMCConfig().paired_slot_pg_ancestor_sampling is False
    with pytest.raises(ValueError, match="requires conditional_smc"):
        BirthDeathSMCConfig(paired_slot_pg_ancestor_sampling=True)
    assert BirthDeathSMCConfig(
        paired_slot_scope="conditional_smc", paired_slot_pg_fraction=0.1,
        paired_slot_pg_ancestor_sampling=True,
    ).paired_slot_pg_ancestor_sampling
