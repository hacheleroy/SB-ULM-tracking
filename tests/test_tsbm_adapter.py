import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sbulm.calibration import physical_crossing_config
from sbulm.localization import (
    unordered_gaussian_localization_gradient,
    unordered_gaussian_localization_potential,
)
from sbulm.simulation import simulate_crossing
from sbulm.tsbm_adapter import ULMGaussianLocalizationCost


def test_tsbm_cost_matches_numpy_values_and_gradients() -> None:
    config = physical_crossing_config(n_frames=5, localization_std_um=5.0)
    _, sequence = simulate_crossing(config, seed=4)
    detections = np.stack(sequence.detections)[1:-1]
    times = np.linspace(0.0, 1.0, sequence.n_frames)[1:-1]
    cost = ULMGaussianLocalizationCost(
        detections,
        localization_std=config.localization_std,
        y_times=times,
        t_final=1.0,
    ).double()
    latent = detections + 0.001
    latent_torch = torch.tensor(
        latent.reshape(latent.shape[0], -1), dtype=torch.float64, requires_grad=True
    )
    indices = torch.arange(detections.shape[0])
    torch_values = cost(latent_torch, indices)
    torch_values.sum().backward()

    interior_sequence = type(sequence)(
        times=sequence.times[1:-1],
        detections=sequence.detections[1:-1],
    )
    numpy_values = unordered_gaussian_localization_potential(
        latent,
        interior_sequence,
        localization_std=config.localization_std,
    )
    numpy_gradient = unordered_gaussian_localization_gradient(
        latent,
        interior_sequence,
        localization_std=config.localization_std,
    )
    np.testing.assert_allclose(torch_values.detach().numpy(), numpy_values, rtol=1e-10)
    np.testing.assert_allclose(
        latent_torch.grad.detach().numpy().reshape(latent.shape),
        numpy_gradient,
        rtol=1e-10,
        atol=1e-10,
    )
