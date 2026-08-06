from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from distrimuse_imu_edge.cli.common import (
    default_run_name,
    effective_context_lengths_for,
    load_runtime_config,
)
from distrimuse_imu_edge.data.config import DataConfig
from distrimuse_imu_edge.compression.pruning import apply_structured_pruning
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.evaluation.metrics import (
    classification_report_payload,
    collect_predictions,
    event_classification_report_payload,
    predictions_frame,
)
from distrimuse_imu_edge.evaluation.reports import write_run_reports
from distrimuse_imu_edge.training.runner import load_checkpoint_model, resolve_device, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress and evaluate a trained IMU edge model.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Optional label override for reports.")
    parser.add_argument(
        "--method",
        choices=["structured_prune"],
        default="structured_prune",
        help=(
            # argparse %%-formats help strings, so a literal percent must be doubled.
            "Compression to apply. For int8 quantization use imu-edge-quantize: "
            "dynamic quantization was removed because it covers Linear/GRU only "
            "and this project's models hold ~99%% of their MACs in Conv1d."
        ),
    )
    parser.add_argument("--prune-amount", type=float, default=0.25)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help=(
            "Past plus current windows. Default: the value the checkpoint was trained "
            "with, else data.context_len from config. Forced to 1 for single-window models."
        ),
    )
    parser.add_argument(
        "--future-context-len",
        type=int,
        default=None,
        help=(
            "Future look-ahead windows. Default: the value the checkpoint was trained "
            "with, else data.future_context_len from config. Forced to 0 for "
            "single-window models."
        ),
    )
    return parser


def resolve_context_lengths(
    ckpt: dict[str, Any],
    *,
    data_cfg: DataConfig,
    context_len: int | None,
    future_context_len: int | None,
) -> tuple[int, int]:
    """Pick the ``(context_len, future_context_len)`` to compress and evaluate with.

    Precedence is explicit CLI value, then the context recorded in the
    checkpoint, then the config. The checkpoint takes priority over the config
    because a checkpoint is only valid for the context it was trained with:
    evaluating a model trained on 15 windows against the config's 8 raises no
    error — ``encode_windows`` only checks ``current_index < n`` — it just
    silently reports degraded metrics and a wrong efficiency profile.
    """
    trained = (ckpt.get("config") or {}).get("data") or {}
    resolved_context = (
        context_len
        if context_len is not None
        else int(trained.get("context_len", data_cfg.context_len))
    )
    resolved_future = (
        future_context_len
        if future_context_len is not None
        else int(trained.get("future_context_len", data_cfg.future_context_len))
    )
    return resolved_context, resolved_future


def check_context_matches_checkpoint(
    model_kwargs: dict[str, Any], *, context_len: int, future_context_len: int
) -> None:
    """Fail loudly when the chosen context contradicts the saved architecture.

    Window-sequence models bake ``current_index`` (which must equal
    ``context_len - 1``) and ``bidirectional`` (set when future context exists)
    into their constructor arguments. A mismatch produces a model that runs but
    classifies the wrong sequence position or silently loses half its receptive
    field, so it must be an error rather than a warning.
    """
    current_index = model_kwargs.get("current_index")
    if current_index is not None and int(current_index) != context_len - 1:
        raise SystemExit(
            f"context mismatch: checkpoint was built with current_index="
            f"{current_index}, which requires --context-len {int(current_index) + 1}, "
            f"but this run resolved --context-len {context_len}. "
            "Pass the matching value explicitly."
        )
    bidirectional = model_kwargs.get("bidirectional")
    if bidirectional is not None and bool(bidirectional) != (future_context_len > 0):
        expected = "greater than 0" if bidirectional else "0"
        raise SystemExit(
            f"context mismatch: checkpoint was built with bidirectional="
            f"{bidirectional}, so --future-context-len must be {expected}, "
            f"but this run resolved {future_context_len}."
        )


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved = load_runtime_config(args.config)
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    model, ckpt = load_checkpoint_model(args.checkpoint, map_location="cpu")
    model_name = args.model or ckpt["model_name"]
    model_kwargs = ckpt.get("model_kwargs", {})
    data_cfg.context_len, data_cfg.future_context_len = resolve_context_lengths(
        ckpt,
        data_cfg=data_cfg,
        context_len=args.context_len,
        future_context_len=args.future_context_len,
    )
    data_cfg.context_len, data_cfg.future_context_len = effective_context_lengths_for(
        model_name,
        data_cfg.context_len,
        data_cfg.future_context_len,
    )
    check_context_matches_checkpoint(
        model_kwargs,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
    )
    print(
        f"Compressing {model_name} with context_len={data_cfg.context_len} "
        f"future_context_len={data_cfg.future_context_len} (method={args.method})"
    )
    resolved["data"] = data_cfg.to_dict()
    compression = {"method": args.method, "amount": args.prune_amount}
    model = apply_structured_pruning(model, amount=args.prune_amount)

    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    run_name = args.run_name or default_run_name(
        model_name,
        width_mult=float(model_kwargs.get("width_mult", train_cfg.width_mult)),
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        suffix=args.method,
    )
    output_dir = Path(train_cfg.output_root) / run_name
    resolved["compression"] = compression
    resolved["source_checkpoint"] = str(Path(args.checkpoint).expanduser())

    if args.finetune:
        train_model(
            model=model,
            model_name=model_name,
            model_kwargs=model_kwargs,
            datamodule=dm,
            train_config=train_cfg,
            output_dir=output_dir,
            resolved_config=resolved,
            compression=compression,
        )
        print(output_dir)
        return

    device = resolve_device(train_cfg.device)
    model = model.to(device).eval()
    val_true, val_pred, val_prob = collect_predictions(model, dm.val_loader(), device=device)
    test_true, test_pred, test_prob = collect_predictions(model, dm.test_loader(), device=device)
    metrics = {
        "model": model_name,
        "model_kwargs": model_kwargs,
        **classification_report_payload(y_true=val_true, y_pred=val_pred, y_prob=val_prob, n_classes=data_cfg.n_classes, prefix="val"),
        **classification_report_payload(y_true=test_true, y_pred=test_pred, y_prob=test_prob, n_classes=data_cfg.n_classes, prefix="test"),
    }
    stats = compute_model_stats(
        model.to("cpu"),
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        window_size_s=data_cfg.window_size_s,
        n_channels=len(data_cfg.sensor_cols),
        compression=compression,
        hop_size_s=data_cfg.hop_size_s,
        energy_profile=resolved.get("energy"),
    )
    predictions = pd.concat(
        [
            predictions_frame(split="val", y_true=val_true, y_pred=val_pred, y_prob=val_prob, metadata=dm.split_metadata("val")),
            predictions_frame(split="test", y_true=test_true, y_pred=test_pred, y_prob=test_prob, metadata=dm.split_metadata("test")),
        ],
        ignore_index=True,
    )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "source_checkpoint": str(Path(args.checkpoint).expanduser()),
            "compression": compression,
        },
        output_dir / "checkpoints" / "best.ckpt",
    )
    for split in ("val", "test"):
        metrics.update(
            event_classification_report_payload(
                predictions,
                n_classes=data_cfg.n_classes,
                prefix=split,
            )
        )

    write_run_reports(
        output_dir=output_dir,
        metrics=metrics,
        model_stats=stats,
        predictions=predictions,
        resolved_config=resolved,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
