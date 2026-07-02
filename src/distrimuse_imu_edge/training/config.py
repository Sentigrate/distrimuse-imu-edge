from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DistillationConfig:
    temperature: float = 4.0
    alpha: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    early_stop_patience: int = 8
    output_root: Path = Path("experiments/results")
    run_name: str | None = None
    width_mult: float = 0.5
    log_every_n_batches: int = 10
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["output_root"] = str(self.output_root)
        return out


def train_config_from_mapping(payload: dict[str, Any]) -> TrainConfig:
    data = dict(payload.get("train", payload))
    distill = data.get("distillation", {}) or {}
    return TrainConfig(
        max_epochs=int(data.get("max_epochs", 30)),
        lr=float(data.get("lr", 1e-3)),
        weight_decay=float(data.get("weight_decay", 1e-4)),
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        early_stop_patience=int(data.get("early_stop_patience", 8)),
        output_root=Path(data.get("output_root", "experiments/results")).expanduser(),
        run_name=data.get("run_name"),
        width_mult=float(data.get("width_mult", 0.5)),
        log_every_n_batches=int(data.get("log_every_n_batches", 10)),
        distillation=DistillationConfig(
            temperature=float(distill.get("temperature", 4.0)),
            alpha=float(distill.get("alpha", 0.5)),
        ),
    )
