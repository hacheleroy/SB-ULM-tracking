"""PyTorch observation potential compatible with the public TSBM interface.

PyTorch is an optional dependency. Import this module only in the external
TSBM environment or after installing ``sbulm[tsbm]``.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial, log, pi

import numpy as np
import torch
from numpy.typing import ArrayLike


class ULMGaussianLocalizationCost(torch.nn.Module):
    r"""Discrete unordered-set likelihood for flattened particle states.

    ``xt`` follows TSBM's ``(..., K, D_flat)`` convention when every cost time
    is evaluated together, or ``(B, D_flat)`` when per-sample cost indices are
    supplied. The flattened state is interpreted as ``(N_particles, D_space)``.
    """

    def __init__(
        self,
        detections: ArrayLike,
        *,
        localization_std: float,
        y_times: ArrayLike,
        t_final: float,
        attr: float = 1.0,
        include_normalization: bool = True,
    ) -> None:
        super().__init__()
        observations = np.asarray(detections, dtype=np.float64)
        times = np.asarray(y_times, dtype=np.float32)
        if observations.ndim != 3:
            raise ValueError("detections must have shape (K, N_particles, D_space)")
        if times.shape != (observations.shape[0],):
            raise ValueError("y_times must contain one time per detection set")
        if localization_std <= 0 or not np.isfinite(localization_std):
            raise ValueError("localization_std must be finite and positive")
        if attr <= 0 or not np.isfinite(attr):
            raise ValueError("attr must be finite and positive")
        if t_final <= 0 or np.any(times <= 0) or np.any(times >= t_final):
            raise ValueError("cost times must lie strictly inside (0, t_final)")

        self.K, self.n_particles, self.spatial_dim = observations.shape
        self.dim = self.n_particles * self.spatial_dim
        self.localization_std = float(localization_std)
        self.t_final = float(t_final)
        self.attr = float(attr)
        self.include_normalization = bool(include_normalization)
        permutation_array = np.asarray(
            tuple(permutations(range(self.n_particles))), dtype=np.int64
        )
        self.register_buffer("observations", torch.from_numpy(observations))
        self.register_buffer("permutations", torch.from_numpy(permutation_array))
        self.register_buffer("y_times", torch.from_numpy(times))
        self.min_t_cost = float(times[0] / t_final)
        self.max_t_cost = float(times[-1] / t_final)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond_sampler: object | None = None,
        imf_train: bool = False,
    ) -> torch.Tensor:
        del cond_sampler, imf_train
        if xt.shape[-1] != self.dim:
            raise ValueError(f"the final state dimension must be {self.dim}")
        indices = torch.as_tensor(t, device=xt.device).long()
        if torch.any(indices < 0) or torch.any(indices >= self.K):
            raise ValueError("cost-time indices are outside the configured range")

        observations = self.observations.to(device=xt.device, dtype=xt.dtype)
        permutations_tensor = self.permutations.to(device=xt.device)
        selected = observations[indices]
        assigned = selected[..., permutations_tensor, :]
        latent = xt.unflatten(-1, (self.n_particles, self.spatial_dim))
        residual = latent.unsqueeze(-3) - assigned
        costs = 0.5 * residual.square().sum(dim=(-1, -2))
        costs = costs / self.localization_std**2
        potential = -torch.logsumexp(-costs, dim=-1) + log(
            factorial(self.n_particles)
        )
        if self.include_normalization:
            potential = potential + self.n_particles * self.spatial_dim * log(
                self.localization_std * np.sqrt(2.0 * pi)
            )
        return self.attr * potential
