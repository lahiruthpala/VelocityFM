"""Per-residue temporal self-attention used after the spatial IPA trunk."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Apply Transformer attention across time independently for each residue."""

    def __init__(
        self,
        channels: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        if self.n_layers <= 0:
            self.encoder: nn.Module | None = None
            return
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=4 * channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return x
        if x.ndim != 4:
            raise ValueError("Expected x with shape (B, W, N, C)")
        b, w, n, c = x.shape
        y = x.permute(0, 2, 1, 3).contiguous().view(b * n, w, c)
        y = self.encoder(y)
        return y.view(b, n, w, c).permute(0, 2, 1, 3).contiguous()
