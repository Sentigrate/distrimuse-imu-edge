from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from distrimuse_imu_edge.cli.common import (
    default_run_name,
    effective_context_lengths_for,
    load_runtime_config,
)
from distrimuse_imu_edge.compression.pruning import apply_structured_pruning
from distrimuse_imu_edge.compression.quantization import apply_dynamic_quantization
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.evaluation.metrics import classification_report_payload, collect_predictions, predictions_frame
from distrimuse_imu_edge.evaluation.reports import write_run_reports
from distrimuse_imu_edge.training.runner import load_checkpoint_model, resolve_device, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress and evaluate a trained IMU edge model.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Optional label override for reports.")
    parser.add_argument("--method", choices=["dynamic_quant", "structured_prune"], required=True)
    parser.add_argument("--prune-amount", type=float, default=0.25)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved = load_runtime_config(args.config)
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    model, ckpt = load_checkpoint_model(args.checkpoint, map_location="cpu")
    model_name = args.model or ckpt["model_name"]
    model_kwargs = ckpt.get("model_kwargs", {})
    data_cfg.context_len, data_cfg.future_context_len = effective_context_lengths_for(
        model_name,
        data_cfg.context_len,
        data_cfg.future_context_len,
    )
    resolved["data"] = data_cfg.to_dict()
    compression = {"method": args.method}
    if args.method == "dynamic_quant":
        model = apply_dynamic_quantization(model)
    elif args.method == "structured_prune":
        model = apply_structured_pruning(model, amount=args.prune_amount)
        compression["amount"] = args.prune_amount

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

    if args.finetune and args.method != "dynamic_quant":
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

    device = torch.device("cpu") if args.method == "dynamic_quant" else resolve_device(train_cfg.device)
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
