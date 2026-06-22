"""Velocity normalization and temporally correlated Gaussian priors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def _std(cfg: Mapping[str, Any], base: str, stride: int | None) -> float:
    if stride is not None:
        by_stride = cfg.get(f"{base}_BY_STRIDE", {})
        if stride in by_stride:
            return float(by_stride[stride])
        if str(stride) in by_stride:
            return float(by_stride[str(stride)])
    return float(cfg.get(base, 1.0))


def normalize_velocities(
    v: torch.Tensor,
    omega: torch.Tensor,
    torsion_rate: torch.Tensor,
    cfg: Mapping[str, Any],
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omega, torsion_rate
    return (
        v / _std(cfg, "V_STD", stride),
        omega / _std(cfg, "OMG_STD", stride),
        torsion_rate / _std(cfg, "THDOT_STD", stride),
    )


def denormalize_velocities(
    v: torch.Tensor,
    omega: torch.Tensor,
    torsion_rate: torch.Tensor,
    cfg: Mapping[str, Any],
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omega, torsion_rate
    return (
        v * _std(cfg, "V_STD", stride),
        omega * _std(cfg, "OMG_STD", stride),
        torsion_rate * _std(cfg, "THDOT_STD", stride),
    )


def sample_ar1_noise(
    shape: tuple[int, int, int, int],
    rho: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample AR(1) noise along the window dimension.

    Args:
        shape: ``(batch, window, residues, channels)``.
        rho: Lag-one correlation coefficient in ``[-1, 1]``.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected a four-dimensional shape, got {shape}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if rho == 0.0:
        return eps
    out = torch.zeros_like(eps)
    out[:, 0] = eps[:, 0]
    scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    for k in range(1, shape[1]):
        out[:, k] = rho * out[:, k - 1] + scale * eps[:, k]
    return out
