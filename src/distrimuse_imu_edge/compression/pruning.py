from __future__ import annotations

from torch import nn
from torch.nn.utils import prune


def apply_structured_pruning(model: nn.Module, *, amount: float = 0.25) -> nn.Module:
    """Apply structured channel pruning to Conv1d layers and remove masks."""
    if not 0 <= amount < 1:
        raise ValueError("amount must be in [0, 1)")
    for module in model.modules():
        if isinstance(module, nn.Conv1d) and module.out_channels > 1:
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model
