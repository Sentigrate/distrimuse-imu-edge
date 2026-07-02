from __future__ import annotations

import torch
from torch import nn


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic quantization to supported linear layers."""
    return torch.quantization.quantize_dynamic(model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8)
