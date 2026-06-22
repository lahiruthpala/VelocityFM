"""Mask-safe structural losses adapted from the training notebook."""

from __future__ import annotations

import torch

from velocityfm.geometry.so3 import wrap_to_pi


def rotation_geodesic_mse(
    r_pred: torch.Tensor,
    r_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    rp, rt = r_pred.float(), r_true.float()
    m = mask.bool() if mask.dtype == torch.bool else mask > 0.5
    identity = torch.eye(3, device=rp.device, dtype=rp.dtype)
    identity = identity.view((1,) * (rp.ndim - 2) + (3, 3))
    rp = torch.where(m[..., None, None], rp, identity)
    rt = torch.where(m[..., None, None], rt, identity)
    relative = rp.transpose(-1, -2) @ rt
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = (0.5 * (trace - 1.0)).clamp(-1.0 + eps, 1.0 - eps)
    wx = relative[..., 2, 1] - relative[..., 1, 2]
    wy = relative[..., 0, 2] - relative[..., 2, 0]
    wz = relative[..., 1, 0] - relative[..., 0, 1]
    sine = 0.5 * torch.sqrt((wx.square() + wy.square() + wz.square()).clamp_min(eps))
    angle = torch.atan2(sine, cosine)
    loss = torch.where(m, angle.square(), torch.zeros_like(angle))
    return loss.sum() / m.float().sum().clamp_min(1.0)


def torsion_periodic_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    torsion_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    mask = (torsion_mask > 0.5) & (residue_mask.unsqueeze(-1) > 0.5)
    delta = torch.where(mask, pred.float() - true.float(), torch.zeros_like(pred.float()))
    loss = 1.0 - torch.cos(wrap_to_pi(delta))
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    return loss.sum() / mask.float().sum().clamp_min(1.0)


def distogram_mse(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    mask = residue_mask.bool() if residue_mask.dtype == torch.bool else residue_mask > 0.5
    xp = torch.where(mask[..., None], x_pred.float(), torch.zeros_like(x_pred.float()))
    xt = torch.where(mask[..., None], x_true.float(), torch.zeros_like(x_true.float()))
    dp = torch.cdist(xp, xp)
    dt = torch.cdist(xt, xt)
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    error = torch.where(pair_mask, (dp - dt).square(), torch.zeros_like(dp))
    return error.sum() / pair_mask.float().sum().clamp_min(1.0)
