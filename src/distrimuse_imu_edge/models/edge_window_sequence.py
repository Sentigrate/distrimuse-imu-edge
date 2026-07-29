from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from distrimuse_imu_edge.models.base import ConvWindowEncoder, make_width
from distrimuse_imu_edge.models.registry import register_model


class _WindowSequenceMixin:
    current_index: int
    window_encoder: ConvWindowEncoder

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B,N,T,C), got {tuple(x.shape)}")
        _, n, _, _ = x.shape
        if not 0 <= self.current_index < n:
            raise ValueError(
                f"current_index {self.current_index} is outside input with {n} windows"
            )
        b, n, t, c = x.shape
        flat = x.reshape(b * n, t, c).transpose(1, 2)
        return self.window_encoder(flat).reshape(b, n, -1)


@register_model("edge_window_gru")
class EdgeWindowGRU(nn.Module, _WindowSequenceMixin):
    """Compact per-window CNN followed by a temporal GRU."""

    def __init__(
        self,
        n_classes: int = 9,
        input_channels: int = 6,
        width_mult: float = 0.5,
        current_index: int = 0,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embedding_dim = make_width(96, width_mult)
        hidden_dim = make_width(96, width_mult)
        self.current_index = current_index
        self.bidirectional = bidirectional
        self.window_encoder = ConvWindowEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            width_mult=width_mult,
        )
        self.temporal = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_windows(x)
        temporal, _ = self.temporal(encoded)
        return self.head(temporal[:, self.current_index])


class _SequenceConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.left_padding = 0 if bidirectional else 2 * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation if bidirectional else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding:
            x = functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class _ChannelLayerNorm(nn.Module):
    """Apply LayerNorm per sequence position without mixing future positions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _WindowTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _SequenceConv(
                in_channels,
                out_channels,
                dilation=dilation,
                bidirectional=bidirectional,
            ),
            _ChannelLayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            _SequenceConv(
                out_channels,
                out_channels,
                dilation=dilation,
                bidirectional=bidirectional,
            ),
            _ChannelLayerNorm(out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


@register_model("edge_window_tcn")
class EdgeWindowTCN(nn.Module, _WindowSequenceMixin):
    """Compact per-window CNN followed by a temporal embedding TCN."""

    def __init__(
        self,
        n_classes: int = 9,
        input_channels: int = 6,
        width_mult: float = 0.5,
        current_index: int = 0,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embedding_dim = make_width(96, width_mult)
        hidden_dim = make_width(96, width_mult)
        self.current_index = current_index
        self.bidirectional = bidirectional
        self.window_encoder = ConvWindowEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            width_mult=width_mult,
        )
        self.temporal = nn.Sequential(
            _WindowTCNBlock(
                embedding_dim,
                hidden_dim,
                dilation=1,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            _WindowTCNBlock(
                hidden_dim,
                hidden_dim,
                dilation=2,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            _WindowTCNBlock(
                hidden_dim,
                hidden_dim,
                dilation=4,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_windows(x).transpose(1, 2)
        temporal = self.temporal(encoded).transpose(1, 2)
        return self.head(temporal[:, self.current_index])
