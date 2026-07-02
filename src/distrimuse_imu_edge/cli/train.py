from __future__ import annotations

import argparse
from pathlib import Path

from distrimuse_imu_edge.cli.common import (
    default_run_name,
    effective_context_len_for,
    load_runtime_config,
    model_kwargs_for,
)
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.runner import load_transfer_checkpoint, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an IMU edge model.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--width-mult", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device: auto, cpu, cuda, mps, etc. Defaults to train.device from config.",
    )
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=None,
        help="Print live training loss/F1 every N batches. Use 0 to disable batch log lines.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Checkpoint used to initialize matching non-head weights before training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved = load_runtime_config(args.config)
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    if args.device is not None:
        train_cfg.device = args.device
    if args.log_every_n_batches is not None:
        train_cfg.log_every_n_batches = args.log_every_n_batches
    width_mult = args.width_mult if args.width_mult is not None else train_cfg.width_mult
    data_cfg.context_len = effective_context_len_for(args.model, data_cfg.context_len)
    model_kwargs = model_kwargs_for(args.model, data_cfg=data_cfg, width_mult=width_mult)
    model = build_model(args.model, **model_kwargs)
    transfer_report = None
    if args.init_checkpoint is not None:
        transfer_report = load_transfer_checkpoint(model, args.init_checkpoint)
    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    run_name = args.run_name or train_cfg.run_name or default_run_name(
        args.model, width_mult=width_mult, context_len=data_cfg.context_len
    )
    output_dir = Path(train_cfg.output_root) / run_name
    resolved["data"] = data_cfg.to_dict()
    resolved["train"] = train_cfg.to_dict()
    resolved["model"] = {"name": args.model, "kwargs": model_kwargs}
    if transfer_report is not None:
        resolved["init_checkpoint"] = transfer_report
    train_model(
        model=model,
        model_name=args.model,
        model_kwargs=model_kwargs,
        datamodule=dm,
        train_config=train_cfg,
        output_dir=output_dir,
        resolved_config=resolved,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
