from __future__ import annotations

import torch
from torch import nn


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to supported linear and recurrent layers."""
    if torch.backends.quantized.engine == "none":
        supported = torch.backends.quantized.supported_engines
        for engine in ("x86", "fbgemm", "qnnpack", "onednn"):
            if engine in supported:
                torch.backends.quantized.engine = engine
                break
    return torch.quantization.quantize_dynamic(
        model.to("cpu").eval(),
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
