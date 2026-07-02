from __future__ import annotations

import argparse
from pathlib import Path

from distrimuse_imu_edge.cli.common import default_run_name, load_runtime_config, model_kwargs_for
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.distillation import load_teacher
from distrimuse_imu_edge.training.runner import resolve_device, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill a teacher into a compact IMU student.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--width-mult", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved = load_runtime_config(args.config)
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    if args.temperature is not None:
        train_cfg.distillation.temperature = args.temperature
    if args.alpha is not None:
        train_cfg.distillation.alpha = args.alpha
    width_mult = args.width_mult if args.width_mult is not None else train_cfg.width_mult
    model_kwargs = model_kwargs_for(args.student, data_cfg=data_cfg, width_mult=width_mult)
    student = build_model(args.student, **model_kwargs)
    teacher = load_teacher(args.teacher_checkpoint, device=resolve_device(train_cfg.device))
    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    run_name = args.run_name or train_cfg.run_name or default_run_name(
        args.student, width_mult=width_mult, context_len=data_cfg.context_len, suffix="distilled"
    )
    output_dir = Path(train_cfg.output_root) / run_name
    resolved["train"] = train_cfg.to_dict()
    resolved["model"] = {"name": args.student, "kwargs": model_kwargs}
    resolved["teacher_checkpoint"] = str(Path(args.teacher_checkpoint).expanduser())
    train_model(
        model=student,
        model_name=args.student,
        model_kwargs=model_kwargs,
        datamodule=dm,
        train_config=train_cfg,
        output_dir=output_dir,
        resolved_config=resolved,
        teacher=teacher,
        compression={"method": "distillation", "temperature": train_cfg.distillation.temperature, "alpha": train_cfg.distillation.alpha},
    )
    print(output_dir)


if __name__ == "__main__":
    main()
