import numpy as np

from sbulm.simulation import CrossingConfig, simulate_crossing


def test_crossing_shapes_and_reference_ids() -> None:
    config = CrossingConfig(n_frames=7, localization_std=0.01)
    truth, detections = simulate_crossing(config, seed=4)
    assert truth.positions.shape == (7, 2, 2)
    assert detections.cardinalities == (2,) * 7
    assert detections.reference_ids is not None
    for ids in detections.reference_ids:
        np.testing.assert_array_equal(np.sort(ids), [0, 1])


def test_central_separation_has_direct_geometric_meaning() -> None:
    config = CrossingConfig(
        n_frames=9, central_separation=0.12, localization_std=0.0
    )
    truth, _ = simulate_crossing(config, seed=0)
    central = truth.positions[config.n_frames // 2]
    np.testing.assert_allclose(np.linalg.norm(central[0] - central[1]), 0.12)


def test_simulation_is_seed_reproducible() -> None:
    _, first = simulate_crossing(seed=12)
    _, second = simulate_crossing(seed=12)
    for left, right in zip(first.detections, second.detections, strict=True):
        np.testing.assert_allclose(left, right)

