from __future__ import annotations

import torch
from torch import nn


def make_width(value: int, width_mult: float) -> int:
    return max(8, int(round(value * width_mult)))


class CurrentWindowMixin:
    """Helper for models that only consume the current window."""

    @staticmethod
    def current_window(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x[:, -1]
        if x.ndim == 3:
            return x
        raise ValueError(f"expected (B,N,T,C) or (B,T,C), got {tuple(x.shape)}")


class ContextConvMixin:
    """Helper for models that treat causal context as one longer 1D signal."""

    @staticmethod
    def context_signal(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.transpose(1, 2)
        if x.ndim != 4:
            raise ValueError(f"expected (B,N,T,C), got {tuple(x.shape)}")
        b, n, t, c = x.shape
        return x.reshape(b, n * t, c).transpose(1, 2)


class ConvWindowEncoder(nn.Module):
    def __init__(self, input_channels: int = 6, embedding_dim: int = 128, width_mult: float = 1.0) -> None:
        super().__init__()
        c1 = make_width(64, width_mult)
        c2 = make_width(128, width_mult)
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(c2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).squeeze(-1))
