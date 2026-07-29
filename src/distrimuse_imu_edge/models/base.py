from __future__ import annotations

import torch
from torch import nn


def make_width(value: int, width_mult: float) -> int:
    return max(8, int(round(value * width_mult)))


class CurrentWindowMixin:
    """Helper for models that require exactly one current window."""

    @staticmethod
    def current_window(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    f"single-window model received {x.shape[1]} windows; "
                    "use edge_window_gru or edge_window_tcn for temporal context"
                )
            return x[:, 0]
        if x.ndim == 3:
            return x
        raise ValueError(f"expected (B,N,T,C) or (B,T,C), got {tuple(x.shape)}")


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
