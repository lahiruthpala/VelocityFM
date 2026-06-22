import torch

from velocityfm.inference.integration import integrate_velocities_to_window


def test_zero_velocity_preserves_state():
    b, w, n = 2, 4, 3
    x0 = torch.randn(b, n, 3)
    r0 = torch.eye(3).reshape(1, 1, 3, 3).repeat(b, n, 1, 1)
    t0 = torch.randn(b, n, 7)
    zeros3 = torch.zeros(b, w, n, 3)
    zeros7 = torch.zeros(b, w, n, 7)
    x, r, t = integrate_velocities_to_window(x0, r0, t0, zeros3, zeros3, zeros7, 1.0)
    assert torch.allclose(x, x0[:, None].expand_as(x))
    assert torch.allclose(r, r0[:, None].expand_as(r))
    expected_t = ((t0 + torch.pi) % (2 * torch.pi) - torch.pi)[:, None].expand_as(t)
    assert torch.allclose(t, expected_t)
