from __future__ import annotations

import torch
from torch import nn

from distrimuse_imu_edge.models.base import CurrentWindowMixin, make_width
from distrimuse_imu_edge.models.registry import register_model


class _ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


@register_model("edge_tcn")
class EdgeTCN(nn.Module, CurrentWindowMixin):
    """Residual dilated TCN student for one current window."""

    def __init__(self, n_classes: int = 9, input_channels: int = 6, width_mult: float = 0.5) -> None:
        super().__init__()
        c1 = make_width(48, width_mult)
        c2 = make_width(96, width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
        )
        self.tcn = nn.Sequential(
            _ResidualTCNBlock(c1, c1, dilation=1),
            _ResidualTCNBlock(c1, c2, dilation=2),
            _ResidualTCNBlock(c2, c2, dilation=4),
            _ResidualTCNBlock(c2, c2, dilation=8),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(c2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.current_window(x).transpose(1, 2)
        return self.head(self.tcn(self.stem(signal)))
