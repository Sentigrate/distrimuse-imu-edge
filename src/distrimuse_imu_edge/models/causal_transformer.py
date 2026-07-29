from __future__ import annotations

import math

import torch
from torch import nn

from distrimuse_imu_edge.models.base import ConvWindowEncoder
from distrimuse_imu_edge.models.registry import register_model


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]]


@register_model("causal_context_transformer_cnn")
class CausalContextTransformerCNN(nn.Module):
    """Window CNN plus causal or bidirectional temporal Transformer."""

    def __init__(
        self,
        n_classes: int = 9,
        input_channels: int = 6,
        context_len: int = 8,
        current_index: int | None = None,
        bidirectional: bool = False,
        enc_channels: int = 128,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if current_index is None:
            current_index = context_len - 1
        if not 0 <= current_index < context_len:
            raise ValueError("current_index must be within the configured context")
        self.context_len = context_len
        self.current_index = current_index
        self.bidirectional = bidirectional
        self.window_encoder = ConvWindowEncoder(input_channels, enc_channels, width_mult=width_mult)
        self.input_proj = nn.Linear(enc_channels, d_model)
        self.pe = _PositionalEncoding(d_model, max_len=context_len + 4)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B,N,T,C), got {tuple(x.shape)}")
        b, n, t, c = x.shape
        if n != self.context_len:
            raise ValueError(f"expected {self.context_len} windows, got {n}")
        flat = x.reshape(b * n, t, c).transpose(1, 2)
        embeds = self.window_encoder(flat).reshape(b, n, -1)
        seq = self.pe(self.input_proj(embeds))
        mask = (
            None
            if self.bidirectional
            else nn.Transformer.generate_square_subsequent_mask(n, device=seq.device)
        )
        encoded = self.transformer(seq, mask=mask)
        return self.head(encoded[:, self.current_index])
