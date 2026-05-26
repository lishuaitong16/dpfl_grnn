"""Noise utilities for DP/robustness experiments.

This project uses these mechanisms to inject physical noise and test how
training/GRNN attack behavior changes as noise grows. It does not perform
clipping or compute a formal DP sensitivity bound.
"""

import math
import torch


def add_laplace_noise(vec: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add zero-mean Laplace noise with the requested per-coordinate std."""
    if noise_std <= 0:
        return vec
    scale = noise_std / math.sqrt(2.0)
    lap = torch.distributions.Laplace(loc=0.0, scale=scale)
    noise = lap.sample(vec.shape).to(device=vec.device, dtype=vec.dtype)
    return vec + noise


def add_gaussian_noise(vec: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise with the requested per-coordinate std."""
    if noise_std <= 0:
        return vec
    return vec + torch.randn_like(vec) * noise_std


def privatize(vec: torch.Tensor, mechanism: str, noise_std: float) -> torch.Tensor:
    """Add physical noise to a flattened update/gradient vector.

    Args:
        vec: flattened update/gradient vector.
        mechanism: "laplace", "gaussian", or "none".
        noise_std: per-coordinate standard deviation of the injected noise.
    """
    mechanism = mechanism.lower()
    if mechanism == "none":
        return vec
    if mechanism == "laplace":
        return add_laplace_noise(vec, noise_std=noise_std)
    if mechanism == "gaussian":
        return add_gaussian_noise(vec, noise_std=noise_std)
    raise ValueError(f"未知机制: {mechanism}")
