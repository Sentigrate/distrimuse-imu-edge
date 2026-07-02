from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"model already registered: {name}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[nn.Module]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}") from exc


def build_model(name: str, **kwargs) -> nn.Module:
    return get_model_class(name)(**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY)
