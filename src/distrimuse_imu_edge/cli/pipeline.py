"""Full pipeline: train teacher → distill students → (optionally) compress → benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from distrimuse_imu_edge.cli.common import (
    SINGLE_WINDOW_MODELS,
    default_run_name,
    load_runtime_config,
    model_kwargs_for,
)
from distrimuse_imu_edge.compression.quantization import apply_dynamic_quantization
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.aggregate import aggregate_results
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.evaluation.metrics import (
    classification_report_payload,
    collect_predictions,
    predictions_frame,
)
from distrimuse_imu_edge.evaluation.reports import write_run_reports
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.distillation import load_teacher
from distrimuse_imu_edge.training.runner import load_checkpoint_model, resolve_device, train_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full IMU edge pipeline: train teacher → distill students → "
            "(optionally) compress → aggregate benchmark."
        )
    )
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument(
        "--students",
        nargs="+",
        default=None,
        help="Student model names. Default: all non-teacher models from benchmark.models.",
    )
    p.add_argument(
        "--width-mults",
        nargs="+",
        type=float,
        default=None,
        help="Width multipliers to sweep. Default: benchmark.width_multipliers.",
    )
    p.add_argument("--teacher-epochs", type=int, default=None, help="Override max_epochs for teacher.")
    p.add_argument("--student-epochs", type=int, default=None, help="Override max_epochs for students.")
    p.add_argument(
        "--compress",
        action="store_true",
        help="Apply dynamic quantization to each distilled checkpoint after training.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its checkpoint already exists.",
    )
    return p


def _divider(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def _run_teacher(
    *,
    dm: IMUEdgeDataModule,
    data_cfg,
    train_cfg,
    resolved_base: dict,
    output_root: Path,
    teacher_epochs: int | None,
    skip_existing: bool,
) -> Path:
    wm = 1.0
    run_name = default_run_name(
        "teacher_causal_cnn",
        width_mult=wm,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
    )
    ckpt = output_root / run_name / "checkpoints" / "best.ckpt"

    if ckpt.exists() and skip_existing:
        print(f"[skip] teacher — {ckpt}")
        return ckpt

    _divider(f"Training teacher: {run_name}")
    cfg = deepcopy(train_cfg)
    if teacher_epochs is not None:
        cfg.max_epochs = teacher_epochs
    mkw = model_kwargs_for("teacher_causal_cnn", data_cfg=data_cfg, width_mult=wm)
    model = build_model("teacher_causal_cnn", **mkw)
    resolved = {**deepcopy(resolved_base), "model": {"name": "teacher_causal_cnn", "kwargs": mkw}}
    train_model(
        model=model,
        model_name="teacher_causal_cnn",
        model_kwargs=mkw,
        datamodule=dm,
        train_config=cfg,
        output_dir=output_root / run_name,
        resolved_config=resolved,
    )
    return ckpt


def _run_distill(
    *,
    student: str,
    wm: float,
    dm: IMUEdgeDataModule,
    data_cfg,
    train_cfg,
    resolved_base: dict,
    output_root: Path,
    teacher_ckpt: Path,
    device,
    student_epochs: int | None,
    skip_existing: bool,
    step: str,
) -> Path:
    run_name = default_run_name(
        student,
        width_mult=wm,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        suffix="distilled",
    )
    ckpt = output_root / run_name / "checkpoints" / "best.ckpt"

    if ckpt.exists() and skip_existing:
        print(f"[skip] {run_name}")
        return ckpt

    _divider(f"[{step}] Distilling {student} wm={wm}: {run_name}")
    cfg = deepcopy(train_cfg)
    if student_epochs is not None:
        cfg.max_epochs = student_epochs
    mkw = model_kwargs_for(student, data_cfg=data_cfg, width_mult=wm)
    student_model = build_model(student, **mkw)
    teacher = load_teacher(str(teacher_ckpt), device=device)
    compression = {
        "method": "distillation",
        "temperature": cfg.distillation.temperature,
        "alpha": cfg.distillation.alpha,
    }
    resolved = {
        **deepcopy(resolved_base),
        "model": {"name": student, "kwargs": mkw},
        "teacher_checkpoint": str(teacher_ckpt),
    }
    train_model(
        model=student_model,
        model_name=student,
        model_kwargs=mkw,
        datamodule=dm,
        train_config=cfg,
        output_dir=output_root / run_name,
        resolved_config=resolved,
        teacher=teacher,
        compression=compression,
    )
    return ckpt


def _run_quantize(
    *,
    student_name: str,
    wm: float,
    source_ckpt: Path,
    dm: IMUEdgeDataModule,
    data_cfg,
    resolved_base: dict,
    output_root: Path,
    skip_existing: bool,
) -> None:
    if not source_ckpt.exists():
        print(f"[warn] missing checkpoint for quantization: {source_ckpt}")
        return

    run_name = default_run_name(
        student_name,
        width_mult=wm,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        suffix="dynamic_quant",
    )
    out = output_root / run_name
    if (out / "checkpoints" / "best.ckpt").exists() and skip_existing:
        print(f"[skip] {run_name}")
        return

    print(f"  Quantizing {student_name} wm={wm} → {run_name}")
    model, ckpt_data = load_checkpoint_model(str(source_ckpt), map_location="cpu")
    model_kwargs = ckpt_data.get("model_kwargs", {})
    model = apply_dynamic_quantization(model).eval()
    compression = {"method": "dynamic_quant"}

    cpu = torch.device("cpu")
    val_true, val_pred, val_prob = collect_predictions(model, dm.val_loader(), device=cpu)
    test_true, test_pred, test_prob = collect_predictions(model, dm.test_loader(), device=cpu)

    metrics = {
        "model": student_name,
        "model_kwargs": model_kwargs,
        **classification_report_payload(
            y_true=val_true, y_pred=val_pred, y_prob=val_prob,
            n_classes=data_cfg.n_classes, prefix="val",
        ),
        **classification_report_payload(
            y_true=test_true, y_pred=test_pred, y_prob=test_prob,
            n_classes=data_cfg.n_classes, prefix="test",
        ),
    }
    stats = compute_model_stats(
        model,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        window_size_s=data_cfg.window_size_s,
        n_channels=len(data_cfg.sensor_cols),
        compression=compression,
    )
    predictions = pd.concat(
        [
            predictions_frame(
                split="val",
                y_true=val_true,
                y_pred=val_pred,
                y_prob=val_prob,
                metadata=dm.split_metadata("val"),
            ),
            predictions_frame(
                split="test",
                y_true=test_true,
                y_pred=test_pred,
                y_prob=test_prob,
                metadata=dm.split_metadata("test"),
            ),
        ],
        ignore_index=True,
    )
    resolved = {
        **deepcopy(resolved_base),
        "compression": compression,
        "source_checkpoint": str(source_ckpt),
    }
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": student_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "compression": compression,
        },
        out / "checkpoints" / "best.ckpt",
    )
    write_run_reports(
        output_dir=out,
        metrics=metrics,
        model_stats=stats,
        predictions=predictions,
        resolved_config=resolved,
    )
    print(f"    → {out}")


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved_base = load_runtime_config(args.config)

    bm = resolved_base.get("raw", {}).get("benchmark", {})
    all_models = bm.get(
        "models",
        ["teacher_causal_cnn", "edge_window_gru", "edge_window_tcn"],
    )
    students = args.students or [m for m in all_models if m != "teacher_causal_cnn"]
    invalid_students = sorted(set(students).intersection(SINGLE_WINDOW_MODELS))
    if invalid_students:
        raise ValueError(
            "The distillation pipeline requires context-capable students; "
            f"train single-window models separately: {invalid_students}"
        )
    width_mults = args.width_mults or bm.get("width_multipliers", [0.25, 0.5, 1.0])
    output_root = Path(train_cfg.output_root)
    device = resolve_device(train_cfg.device)

    print("Setting up data module...")
    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()

    # Step 1: train teacher
    teacher_ckpt = _run_teacher(
        dm=dm,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resolved_base=resolved_base,
        output_root=output_root,
        teacher_epochs=args.teacher_epochs,
        skip_existing=args.skip_existing,
    )

    # Step 2: distill each student × width_mult
    total = len(students) * len(width_mults)
    distilled: list[tuple[str, float, Path]] = []
    for i, student in enumerate(students):
        for j, wm in enumerate(width_mults):
            step = f"{i * len(width_mults) + j + 1}/{total}"
            ckpt = _run_distill(
                student=student,
                wm=wm,
                dm=dm,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                resolved_base=resolved_base,
                output_root=output_root,
                teacher_ckpt=teacher_ckpt,
                device=device,
                student_epochs=args.student_epochs,
                skip_existing=args.skip_existing,
                step=step,
            )
            distilled.append((student, wm, ckpt))

    # Step 3: compress (optional)
    if args.compress:
        _divider("Applying dynamic quantization to distilled models")
        for student_name, wm, ckpt in distilled:
            _run_quantize(
                student_name=student_name,
                wm=wm,
                source_ckpt=ckpt,
                dm=dm,
                data_cfg=data_cfg,
                resolved_base=resolved_base,
                output_root=output_root,
                skip_existing=args.skip_existing,
            )

    # Step 4: aggregate + plots
    _divider("Benchmark summary")
    df = aggregate_results(output_root)
    print(df.to_string(index=False))
    print(f"\nSummary CSV : {output_root / 'benchmark_summary.csv'}")
    print(f"Plots       : {output_root}")


if __name__ == "__main__":
    main()
