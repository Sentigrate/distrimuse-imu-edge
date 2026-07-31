from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.evaluation.metrics import (
    classification_report_payload,
    collect_predictions,
    predictions_frame,
)
from distrimuse_imu_edge.evaluation.reports import write_run_reports
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.config import TrainConfig
from distrimuse_imu_edge.training.losses import distillation_loss


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_model(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def load_transfer_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    skip_prefixes: tuple[str, ...] = ("head.",),
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load compatible checkpoint weights into ``model`` for transfer learning.

    Classifier heads are skipped by default so public pretraining checkpoints
    can initialize a DistriMuSe model with a different class count.
    """
    ckpt_path = Path(path).expanduser()
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    source_state = ckpt.get("state_dict", ckpt)
    target_state = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped_head: list[str] = []
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []

    for key, value in source_state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_head.append(key)
            continue
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append(key)
            continue
        loadable[key] = value

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint": str(ckpt_path),
        "source_model": ckpt.get("model_name"),
        "source_model_kwargs": ckpt.get("model_kwargs", {}),
        "loaded_keys": sorted(loadable),
        "loaded_count": len(loadable),
        "skipped_head": sorted(skipped_head),
        "skipped_shape": sorted(skipped_shape),
        "skipped_missing": sorted(skipped_missing),
        "target_missing_after_load": sorted(missing),
        "target_unexpected_after_load": sorted(unexpected),
    }


def _macro_f1_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor, *, n_classes: int) -> float:
    true = y_true.detach().cpu().numpy()
    pred = y_pred.detach().cpu().numpy()
    values: list[float] = []
    for cls in range(n_classes):
        tp = float(np.logical_and(true == cls, pred == cls).sum())
        fp = float(np.logical_and(true != cls, pred == cls).sum())
        fn = float(np.logical_and(true == cls, pred != cls).sum())
        denom = (2.0 * tp) + fp + fn
        if denom > 0.0:
            values.append((2.0 * tp) / denom)
    return float(np.mean(values)) if values else 0.0


def _progress_line(
    *,
    model_name: str,
    epoch: int,
    max_epochs: int,
    batch_idx: int,
    total_batches: int,
    loss: float,
    running_loss: float,
    batch_macro_f1: float,
) -> str:
    return (
        f"[{model_name}] epoch {epoch:03d}/{max_epochs:03d} "
        f"batch {batch_idx:04d}/{total_batches:04d} | "
        f"loss={loss:.4f} | avg_loss={running_loss:.4f} | batch_macro_f1={batch_macro_f1:.4f}"
    )


def _format_int_list(values: np.ndarray, *, max_items: int = 12) -> str:
    items = sorted(int(value) for value in np.unique(values))
    if len(items) <= max_items:
        return ",".join(str(value) for value in items)
    shown = ",".join(str(value) for value in items[:max_items])
    return f"{shown},+{len(items) - max_items}"


def _format_class_counts(y: torch.Tensor, *, n_classes: int) -> str:
    counts = np.bincount(y.detach().cpu().numpy().astype(np.int64), minlength=n_classes)
    return " ".join(f"{idx}:{int(count)}" for idx, count in enumerate(counts))


def _log_dataset_summary(datamodule: IMUEdgeDataModule) -> None:
    cfg = datamodule.config
    tqdm.write(
        "Dataset | "
        f"window={cfg.window_size_s:g}s | hop={cfg.hop_size_s:g}s | "
        f"past+current={cfg.context_len} | future={cfg.future_context_len} | "
        f"channels={len(cfg.sensor_cols)} ({','.join(cfg.sensor_cols)}) | classes={cfg.n_classes}"
    )
    for split_name in ("train", "val", "test"):
        dataset = datamodule.datasets[split_name]
        person_ids = dataset.person_ids
        scenario_ids = dataset.scenario_ids
        tqdm.write(
            f"Dataset {split_name:>5} | windows={len(dataset):6d} | "
            f"subjects={len(np.unique(person_ids)):2d} [{_format_int_list(person_ids)}] | "
            f"scenarios={len(np.unique(scenario_ids)):2d} [{_format_int_list(scenario_ids)}] | "
            f"class_counts={_format_class_counts(dataset.y, n_classes=cfg.n_classes)}"
        )


def _train_epoch(
    model: nn.Module,
    loader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None,
    model_name: str,
    epoch: int,
    max_epochs: int,
    n_classes: int,
    log_every_n_batches: int,
    teacher: nn.Module | None = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> float:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_items = 0
    total_batches = len(loader)
    progress = tqdm(
        loader,
        desc=f"{model_name} epoch {epoch:03d}/{max_epochs:03d}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, (x, y, _) in enumerate(progress, start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        with torch.no_grad():
            teacher_logits = teacher(x) if teacher is not None else None
        loss = distillation_loss(
            logits,
            y,
            teacher_logits=teacher_logits,
            temperature=temperature,
            alpha=alpha,
            class_weights=class_weights,
        )
        loss.backward()
        optimizer.step()
        batch_size = int(y.shape[0])
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value * batch_size
        total_items += batch_size
        running_loss = total_loss / max(1, total_items)
        pred = torch.argmax(logits.detach(), dim=-1)
        batch_macro_f1 = _macro_f1_from_tensors(y, pred, n_classes=n_classes)
        progress.set_postfix(
            {
                "loss": f"{loss_value:.4f}",
                "avg": f"{running_loss:.4f}",
                "f1": f"{batch_macro_f1:.4f}",
            }
        )
        if log_every_n_batches > 0 and (
            batch_idx == 1 or batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
        ):
            tqdm.write(
                _progress_line(
                    model_name=model_name,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    loss=loss_value,
                    running_loss=running_loss,
                    batch_macro_f1=batch_macro_f1,
                )
            )
    return total_loss / max(1, total_items)


def _evaluate_macro_f1(model: nn.Module, loader, *, device: torch.device, n_classes: int) -> float:
    y_true, y_pred, _ = collect_predictions(model, loader, device=device)
    if len(y_true) == 0:
        return 0.0
    return float(classification_report_payload(y_true=y_true, y_pred=y_pred, n_classes=n_classes, prefix="eval")["eval_macro_f1"])


def _metric_namespace(datamodule: IMUEdgeDataModule) -> str | None:
    campaign = datamodule.config.campaign.lower()
    if campaign == "wisdm19":
        return "wisdm"
    return None


def _namespace_report(payload: dict[str, Any], namespace: str | None) -> dict[str, Any]:
    if namespace is None:
        return payload
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key.startswith(("val_", "test_")):
            out[f"{namespace}_{key}"] = value
        else:
            out[key] = value
    return out


def train_model(
    *,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    datamodule: IMUEdgeDataModule,
    train_config: TrainConfig,
    output_dir: Path,
    resolved_config: dict[str, Any],
    teacher: nn.Module | None = None,
    compression: dict[str, Any] | None = None,
    profile_context: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Train ``model`` and write metrics, reports, and an efficiency profile.

    ``profile_context`` overrides the ``(context_len, future_context_len)`` used
    to build the efficiency profiling input. Pass it when the model consumes
    fewer windows than the dataloader emits, as in privileged-context
    distillation, so the reported FLOPs and latency match deployment.

    The analytic energy estimate reads its hardware profile from
    ``resolved_config["energy"]``, which ``load_runtime_config`` populates.
    A caller that omits the key gets the default profile.
    """
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
    class_weights = (
        torch.as_tensor(datamodule.class_weights, dtype=torch.float32, device=device)
        if datamodule.class_weights is not None
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict[str, float | int]] = []

    _log_dataset_summary(datamodule)
    tqdm.write(
        f"Training {model_name} on {device} | epochs={train_config.max_epochs} | "
        f"lr={train_config.lr:g} | weight_decay={train_config.weight_decay:g} | "
        f"log_every_n_batches={train_config.log_every_n_batches}"
    )
    for epoch in range(train_config.max_epochs):
        train_loss = _train_epoch(
            model,
            datamodule.train_loader(),
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            model_name=model_name,
            epoch=epoch + 1,
            max_epochs=train_config.max_epochs,
            n_classes=datamodule.config.n_classes,
            log_every_n_batches=train_config.log_every_n_batches,
            teacher=teacher,
            temperature=train_config.distillation.temperature,
            alpha=train_config.distillation.alpha,
        )
        val_f1 = _evaluate_macro_f1(model, datamodule.val_loader(), device=device, n_classes=datamodule.config.n_classes)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_macro_f1": val_f1})
        improved = val_f1 > best_val
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        status = "best" if improved else f"patience {patience}/{train_config.early_stop_patience}"
        tqdm.write(
            f"[{model_name}] epoch {epoch + 1:03d}/{train_config.max_epochs:03d} done  | "
            f"train_loss={train_loss:.4f} | val_macro_f1={val_f1:.4f} | "
            f"best_val_macro_f1={best_val:.4f} | {status}"
        )
        if not improved and patience >= train_config.early_stop_patience:
            tqdm.write(
                f"[{model_name}] early stopping after epoch {epoch + 1:03d}; "
                f"no validation macro-F1 improvement for {patience} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "state_dict": model.to("cpu").state_dict(),
        "normalizer": datamodule.normalizer.state_dict(),
        "config": resolved_config,
        "history": history,
    }
    torch.save(ckpt, output_dir / "checkpoints" / "best.ckpt")

    device = resolve_device(train_config.device)
    model = model.to(device)
    val_true, val_pred, val_prob = collect_predictions(model, datamodule.val_loader(), device=device)
    test_true, test_pred, test_prob = collect_predictions(model, datamodule.test_loader(), device=device)
    metric_namespace = _metric_namespace(datamodule)
    best_key = "best_val_macro_f1" if metric_namespace is None else f"{metric_namespace}_best_val_macro_f1"
    metrics = {
        "model": model_name,
        "model_kwargs": model_kwargs,
        "dataset": datamodule.config.campaign,
        "metric_namespace": metric_namespace or "distrimuse",
        best_key: float(best_val),
        **_namespace_report(
            classification_report_payload(
                y_true=val_true,
                y_pred=val_pred,
                y_prob=val_prob,
                n_classes=datamodule.config.n_classes,
                prefix="val",
            ),
            metric_namespace,
        ),
        **_namespace_report(
            classification_report_payload(
                y_true=test_true,
                y_pred=test_pred,
                y_prob=test_prob,
                n_classes=datamodule.config.n_classes,
                prefix="test",
            ),
            metric_namespace,
        ),
        "history": history,
    }
    profile_ctx, profile_future = profile_context or (
        datamodule.config.context_len,
        datamodule.config.future_context_len,
    )
    stats = compute_model_stats(
        model.to("cpu"),
        context_len=profile_ctx,
        future_context_len=profile_future,
        window_size_s=datamodule.config.window_size_s,
        n_channels=len(datamodule.config.sensor_cols),
        compression=compression,
        hop_size_s=datamodule.config.hop_size_s,
        energy_profile=resolved_config.get("energy"),
    )
    predictions = pd.concat(
        [
            predictions_frame(split="val", y_true=val_true, y_pred=val_pred, y_prob=val_prob, metadata=datamodule.split_metadata("val")),
            predictions_frame(split="test", y_true=test_true, y_pred=test_pred, y_prob=test_prob, metadata=datamodule.split_metadata("test")),
        ],
        ignore_index=True,
    )
    write_run_reports(
        output_dir=output_dir,
        metrics=metrics,
        model_stats=stats,
        predictions=predictions,
        resolved_config=resolved_config,
    )
    (output_dir / "reports" / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return metrics
