import torch

from velocityfm.flow.ar1 import sample_ar1_noise


def test_ar1_shape_and_finiteness():
    x = sample_ar1_noise((4, 16, 10, 3), 0.7, dtype=torch.float64)
    assert x.shape == (4, 16, 10, 3)
    assert torch.isfinite(x).all()


def test_ar1_empirical_lag_one_correlation():
    generator = torch.Generator().manual_seed(42)
    x = sample_ar1_noise((256, 32, 4, 2), 0.7, dtype=torch.float64, generator=generator)
    a = x[:, :-1].reshape(-1)
    b = x[:, 1:].reshape(-1)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert abs(float(corr) - 0.7) < 0.05
