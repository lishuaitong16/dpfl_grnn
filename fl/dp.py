"""Per-round delta DP: clip client update to L2 norm C, then add calibrated noise.

Standard workflow (called once per round per client in federated_train):
  1. clip_gradient(delta, C)          -- bound L2 sensitivity to C
  2. privatize(delta, mechanism, ε, C, δ)  -- add noise scaled to C/ε

Composition: ε_total = T × ε_round  (sequential composition, T = number of rounds).
"""

import math
import torch


def clip_gradient(vec: torch.Tensor, C: float) -> torch.Tensor:
    """Project vec onto the L2 ball of radius C. No-op if ‖vec‖ ≤ C."""
    norm = vec.norm(2).item()
    if norm > C:
        vec = vec * (C / norm)
    return vec


def noise_std_from_epsilon(mechanism: str, epsilon: float, C: float, delta: float = 1e-5) -> float:
    """Per-coordinate noise std derived from (ε, C, δ).

    Laplace (ε-DP):       std = C√2 / ε
    Gaussian ((ε,δ)-DP):  std = C√(2 ln(1.25/δ)) / ε
    """
    if mechanism == "laplace":
        return C * math.sqrt(2) / epsilon
    if mechanism == "gaussian":
        return C * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    raise ValueError(f"未知机制: {mechanism}")


def add_laplace_noise(vec: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add zero-mean Laplace noise with the given per-coordinate std."""
    scale = noise_std / math.sqrt(2)
    noise = torch.distributions.Laplace(0.0, scale).sample(vec.shape).to(vec.device, vec.dtype)
    return vec + noise


def add_gaussian_noise(vec: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise with the given per-coordinate std."""
    return vec + torch.randn_like(vec) * noise_std


def privatize(vec: torch.Tensor, mechanism: str, epsilon: float, C: float,
              delta: float = 1e-5) -> torch.Tensor:
    """Clip vec to C then add noise calibrated to (ε, C, δ).

    Args:
        vec:       flat gradient or delta vector.
        mechanism: "laplace" or "gaussian".
        epsilon:   per-round privacy budget ε_round.
        C:         L2 clipping norm (= sensitivity bound Δf).
        delta:     failure probability δ (Gaussian only, default 1e-5).
    """
    vec = clip_gradient(vec, C)
    noise_std = noise_std_from_epsilon(mechanism, epsilon, C, delta)
    if mechanism == "laplace":
        return add_laplace_noise(vec, noise_std)
    if mechanism == "gaussian":
        return add_gaussian_noise(vec, noise_std)
    raise ValueError(f"未知机制: {mechanism}")
