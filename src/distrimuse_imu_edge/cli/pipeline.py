"""Full pipeline: train teacher → distill students → (optionally) compress → benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path


from distrimuse_imu_edge.cli.common import (
    SINGLE_WINDOW_MODELS,
    default_run_name,
    load_runtime_config,
    model_kwargs_for,
)
from distrimuse_imu_edge.cli.quantize import quantize_checkpoint
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.aggregate import aggregate_results
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
        help="Statically quantize each distilled checkpoint to int8 ONNX after training.",
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
    train_cfg,
    output_root: Path,
    skip_existing: bool,
) -> None:
    """Quantize one distilled checkpoint to int8 ONNX.

    Delegates to the same ``quantize_checkpoint`` the ``imu-edge-quantize`` CLI
    uses, so the pipeline and the standalone command cannot diverge.
    """
    if not source_ckpt.exists():
        print(f"[warn] missing checkpoint for quantization: {source_ckpt}")
        return

    run_name = default_run_name(
        student_name,
        width_mult=wm,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        suffix="int8",
    )
    out = output_root / run_name
    # The deployable artifact is the ONNX graph, not a torch checkpoint.
    if (out / "onnx" / "model_int8.onnx").exists() and skip_existing:
        print(f"[skip] {run_name}")
        return

    print(f"  Quantizing {student_name} wm={wm} -> {run_name}")
    model, ckpt_data = load_checkpoint_model(str(source_ckpt), map_location="cpu")
    quantize_checkpoint(
        model=model,
        model_name=student_name,
        model_kwargs=ckpt_data.get("model_kwargs", {}),
        dm=dm,
        data_cfg=data_cfg,
        resolved=deepcopy(resolved_base),
        output_dir=out,
        source_checkpoint=str(source_ckpt),
        seed=train_cfg.seed,
    )
    print(f"    -> {out}")


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
        _divider("Quantizing distilled models to int8 ONNX")
        for student_name, wm, ckpt in distilled:
            _run_quantize(
                train_cfg=train_cfg,
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
