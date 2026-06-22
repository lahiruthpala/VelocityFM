"""First-order physical-time integration used by VelocityFM rollouts."""

from __future__ import annotations

import torch

from velocityfm.geometry.so3 import SO3, so3_exp, so3_log, wrap_to_pi


def integrate_velocities_to_window(
    x0: torch.Tensor,
    r0: torch.Tensor,
    torsion0: torch.Tensor,
    v_local: torch.Tensor,
    omega: torch.Tensor,
    torsion_rate: torch.Tensor,
    dt: float,
    anchor_alpha: float = 0.0,
    orthonormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate one velocity window using residue-local translational velocity."""
    if v_local.ndim != 4:
        raise ValueError("v_local must have shape (B, W, N, 3)")
    b, w, n, c = v_local.shape
    if c != 3:
        raise ValueError("The final translational channel dimension must be 3")
    x, r, torsion = x0, r0, torsion0
    xs: list[torch.Tensor] = []
    rs: list[torch.Tensor] = []
    torsions: list[torch.Tensor] = []

    for k in range(w):
        dx_global = (r @ (v_local[:, k] * dt).unsqueeze(-1)).squeeze(-1)
        x_euler = x + dx_global
        dr = so3_exp((omega[:, k] * dt).reshape(-1, 3)).reshape(b, n, 3, 3)
        r_euler = r @ dr
        torsion_euler = wrap_to_pi(torsion + torsion_rate[:, k] * dt)

        if anchor_alpha > 0.0:
            denom = max(w - 1, 1)
            alpha_k = anchor_alpha * (1.0 - float(k) / float(denom))
            x = (1.0 - alpha_k) * x_euler + alpha_k * x0
            relative = r0.transpose(-1, -2) @ r_euler
            tangent = so3_log(relative.reshape(-1, 3, 3))
            r = r0 @ so3_exp((1.0 - alpha_k) * tangent).reshape(b, n, 3, 3)
            torsion = wrap_to_pi(
                (1.0 - alpha_k) * torsion_euler + alpha_k * torsion0
            )
        else:
            x, r, torsion = x_euler, r_euler, torsion_euler

        if orthonormalize:
            r = SO3.orthonormalize_safe(r.reshape(-1, 3, 3)).reshape(b, n, 3, 3)
        xs.append(x)
        rs.append(r)
        torsions.append(torsion)

    return torch.stack(xs, dim=1), torch.stack(rs, dim=1), torch.stack(torsions, dim=1)


def integrate_previous_states(
    x0: torch.Tensor,
    r0: torch.Tensor,
    torsion0: torch.Tensor,
    v_local: torch.Tensor,
    omega: torch.Tensor,
    torsion_rate: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the state preceding each velocity step in a window."""
    b, w, n, _ = v_local.shape
    x, r, torsion = x0, r0, torsion0
    xs, rs, torsions = [], [], []
    for k in range(w):
        xs.append(x)
        rs.append(r)
        torsions.append(torsion)
        x = x + (r @ (v_local[:, k] * dt).unsqueeze(-1)).squeeze(-1)
        r = r @ so3_exp((omega[:, k] * dt).reshape(-1, 3)).reshape(b, n, 3, 3)
        torsion = wrap_to_pi(torsion + torsion_rate[:, k] * dt)
    return torch.stack(xs, 1), torch.stack(rs, 1), torch.stack(torsions, 1)
