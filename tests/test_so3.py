import math

import torch

from velocityfm.geometry.so3 import SO3, wrap_to_pi


def test_wrap_to_pi_range():
    values = torch.tensor([-5 * math.pi, -math.pi, 0.0, math.pi, 5 * math.pi])
    wrapped = wrap_to_pi(values)
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped < math.pi + 1e-7)


def test_so3_exp_is_rotation_matrix():
    w = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    r = SO3.exp(w)
    identity = r.transpose(-1, -2) @ r
    assert torch.allclose(identity, torch.eye(3, dtype=torch.float64).expand_as(identity), atol=1e-8)
    assert torch.allclose(torch.linalg.det(r), torch.ones(1, dtype=torch.float64), atol=1e-8)


def test_so3_small_round_trip():
    w = torch.tensor([[0.03, -0.02, 0.01]], dtype=torch.float64)
    recovered = SO3.log(SO3.exp(w))
    assert torch.allclose(recovered, w, atol=1e-6)
