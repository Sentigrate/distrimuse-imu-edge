from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from distrimuse_imu_edge.cli.common import (
    default_run_name,
    effective_context_lengths_for,
    load_runtime_config,
    model_kwargs_for,
)
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.models.context_adapter import ContextSlicingStudent
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
    parser.add_argument("--device", default=None, help="Override train.device from config.")
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help=(
            "Past plus current windows the dataloader emits. Must match the teacher "
            "checkpoint. Default: data.context_len from config."
        ),
    )
    parser.add_argument(
        "--future-context-len",
        type=int,
        default=None,
        help=(
            "Future windows the dataloader emits, i.e. the teacher's look-ahead. "
            "Default: data.future_context_len from config."
        ),
    )
    parser.add_argument(
        "--student-context-len",
        type=int,
        default=None,
        help=(
            "Past plus current windows the student reads. Defaults to the teacher's "
            "context. Set 1 for a current-window-only student."
        ),
    )
    parser.add_argument(
        "--student-future-context-len",
        type=int,
        default=None,
        help="Future windows the student reads. Defaults to the teacher's look-ahead.",
    )
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
    if args.device is not None:
        train_cfg.device = args.device
    if args.context_len is not None:
        data_cfg.context_len = args.context_len
    if args.future_context_len is not None:
        data_cfg.future_context_len = args.future_context_len
    resolved["data"] = data_cfg.to_dict()
    width_mult = args.width_mult if args.width_mult is not None else train_cfg.width_mult

    # The dataloader emits the teacher's window sequence; the student may read a
    # narrower slice of it, which is the privileged-context distillation setup.
    student_context_len = (
        args.student_context_len if args.student_context_len is not None else data_cfg.context_len
    )
    student_future_context_len = (
        args.student_future_context_len
        if args.student_future_context_len is not None
        else data_cfg.future_context_len
    )
    student_context_len, student_future_context_len = effective_context_lengths_for(
        args.student,
        student_context_len,
        student_future_context_len,
    )
    if student_context_len > data_cfg.context_len or student_future_context_len > data_cfg.future_context_len:
        raise ValueError(
            "Student context must fit inside the teacher context: student reads "
            f"{student_context_len} past+current and {student_future_context_len} future windows, "
            f"data provides {data_cfg.context_len} and {data_cfg.future_context_len}"
        )
    student_data_cfg = replace(
        data_cfg,
        context_len=student_context_len,
        future_context_len=student_future_context_len,
    )
    model_kwargs = model_kwargs_for(args.student, data_cfg=student_data_cfg, width_mult=width_mult)
    student = build_model(args.student, **model_kwargs)
    teacher = load_teacher(args.teacher_checkpoint, device=resolve_device(train_cfg.device))
    teacher_context_len = getattr(teacher, "context_len", data_cfg.total_context_len)
    if teacher_context_len != data_cfg.total_context_len:
        raise ValueError(
            "Teacher context must match the data context for distillation: "
            f"teacher expects {teacher_context_len} windows, "
            f"data provides {data_cfg.total_context_len}"
        )

    trainee = student
    if student_data_cfg.total_context_len != data_cfg.total_context_len:
        # Align the student's current window with the teacher's, at index context_len - 1.
        trainee = ContextSlicingStudent(
            student,
            start=data_cfg.context_len - student_context_len,
            length=student_data_cfg.total_context_len,
        )

    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    run_name = args.run_name or train_cfg.run_name or default_run_name(
        args.student,
        width_mult=width_mult,
        context_len=student_context_len,
        future_context_len=student_future_context_len,
        suffix="distilled",
    )
    output_dir = Path(train_cfg.output_root) / run_name
    resolved["train"] = train_cfg.to_dict()
    resolved["model"] = {"name": args.student, "kwargs": model_kwargs}
    resolved["teacher_checkpoint"] = str(Path(args.teacher_checkpoint).expanduser())
    resolved["student_context"] = {
        "context_len": student_context_len,
        "future_context_len": student_future_context_len,
        "slice_start": getattr(trainee, "start", 0),
    }
    train_model(
        model=trainee,
        model_name=args.student,
        model_kwargs=model_kwargs,
        datamodule=dm,
        train_config=train_cfg,
        output_dir=output_dir,
        resolved_config=resolved,
        teacher=teacher,
        compression={"method": "distillation", "temperature": train_cfg.distillation.temperature, "alpha": train_cfg.distillation.alpha},
        profile_context=(student_context_len, student_future_context_len),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
