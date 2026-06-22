"""SO(3) and torsion utilities adapted from the final training notebook."""

from __future__ import annotations

import math
import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to the half-open interval [-pi, pi)."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class SO3:
    @staticmethod
    def hat(w: torch.Tensor) -> torch.Tensor:
        """Map rotation vectors (..., 3) to skew matrices (..., 3, 3)."""
        wx, wy, wz = w.unbind(-1)
        z = torch.zeros_like(wx)
        return torch.stack(
            [z, -wz, wy, wz, z, -wx, -wy, wx, z], dim=-1
        ).reshape(w.shape[:-1] + (3, 3))

    @staticmethod
    def exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable Rodrigues exponential map."""
        theta = torch.linalg.norm(w, dim=-1, keepdim=True)
        theta2 = theta.square()
        theta_safe = theta.clamp_min(eps)
        small = theta < 1e-4
        a = torch.where(
            small,
            1.0 - theta2 / 6.0 + theta2.square() / 120.0,
            torch.sin(theta) / theta_safe,
        )
        b = torch.where(
            small,
            0.5 - theta2 / 24.0 + theta2.square() / 720.0,
            (1.0 - torch.cos(theta)) / theta_safe.square(),
        )
        w_hat = SO3.hat(w)
        eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(w_hat.shape)
        return eye + a.unsqueeze(-1) * w_hat + b.unsqueeze(-1) * (w_hat @ w_hat)

    @staticmethod
    def log(r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Logarithm map for proper rotation matrices.

        This follows the research notebook implementation, including a stable
        near-zero branch.  Near-pi inputs should be treated with care in all
        differentiable SO(3) implementations.
        """
        trace = r.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        skew = r - r.transpose(-1, -2)
        sin_theta = torch.sin(theta).clamp_min(eps)
        factor = theta / (2.0 * sin_theta)
        w_hat = factor[..., None, None] * skew
        w = torch.stack(
            [w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], dim=-1
        )
        small = theta < 1e-6
        if small.any():
            w_small = 0.5 * torch.stack(
                [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1
            )
            w = torch.where(small[..., None], w_small, w)
        return w

    @staticmethod
    def orthonormalize_safe(r: torch.Tensor) -> torch.Tensor:
        """Project matrices onto SO(3) by SVD and correct reflections."""
        u, _, vh = torch.linalg.svd(r)
        uv = u @ vh
        det = torch.linalg.det(uv)
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        correction = torch.eye(3, device=r.device, dtype=r.dtype).expand(r.shape).clone()
        correction[..., 2, 2] = sign
        return u @ correction @ vh


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    return SO3.exp(w)


def so3_log(r: torch.Tensor) -> torch.Tensor:
    return SO3.log(r)
