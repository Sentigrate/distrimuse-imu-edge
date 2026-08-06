"""Export a trained checkpoint to int8 ONNX and evaluate it against float32.

Runs the full static post-training-quantization path: export the float32 graph,
calibrate on the training split, quantize to int8, then evaluate the quantized
model on val and test using the same metrics, confusion matrices, and per-subject
reports as any other run. Writes a standard run directory plus a
``quantization_comparison.json`` holding the float32-versus-int8 deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from distrimuse_imu_edge.cli.common import (
    effective_context_lengths_for,
    load_runtime_config,
)
from distrimuse_imu_edge.cli.compress import (
    check_context_matches_checkpoint,
    resolve_context_lengths,
)
from distrimuse_imu_edge.compression.onnx_int8 import (
    DEFAULT_CALIBRATION_BATCHES,
    OnnxModule,
    build_calibration_reader,
    export_onnx,
    int8_mac_fraction_from_onnx,
    onnx_op_counts,
    quantize_onnx_static,
)
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.evaluation.metrics import (
    classification_report_payload,
    event_classification_report_payload,
    collect_predictions,
    predictions_frame,
)
from distrimuse_imu_edge.evaluation.reports import write_run_reports
from distrimuse_imu_edge.training.runner import load_checkpoint_model, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statically quantize a trained checkpoint to int8 ONNX and evaluate it."
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Optional label override for reports.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Default: the value the checkpoint was trained with, else data.context_len.",
    )
    parser.add_argument(
        "--future-context-len",
        type=int,
        default=None,
        help="Default: the value the checkpoint was trained with, else data.future_context_len.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=DEFAULT_CALIBRATION_BATCHES,
        help="Training batches used to record activation ranges.",
    )
    parser.add_argument(
        "--quant-format",
        choices=["QOperator", "QDQ"],
        default="QOperator",
        help=(
            "QOperator emits fused QLinearConv nodes and shrinks the file. QDQ adds "
            "Quantize/Dequantize pairs which, on models this small, cost more bytes "
            "than int8 weights save."
        ),
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use one weight scale per tensor instead of per output channel.",
    )
    return parser


def _evaluate(model, dm: IMUEdgeDataModule, *, n_classes: int) -> dict:
    device = torch.device("cpu")
    val = collect_predictions(model, dm.val_loader(), device=device)
    test = collect_predictions(model, dm.test_loader(), device=device)
    payload = {
        **classification_report_payload(
            y_true=val[0], y_pred=val[1], y_prob=val[2], n_classes=n_classes, prefix="val"
        ),
        **classification_report_payload(
            y_true=test[0], y_pred=test[1], y_prob=test[2], n_classes=n_classes, prefix="test"
        ),
    }
    return {"payload": payload, "val": val, "test": test}


def _comparison(float32: dict, int8: dict, *, n_classes: int) -> dict:
    """Build the float32-versus-int8 delta table.

    Per-class deltas are included because quantization damage concentrates in
    rare classes, and macro-F1 weights every class equally — so a single small
    class can move the headline number more than the quantization did.
    """
    f_payload, q_payload = float32["payload"], int8["payload"]
    out: dict = {}
    for split in ("val", "test"):
        key = f"{split}_macro_f1"
        out[key] = {
            "float32": f_payload[key],
            "int8": q_payload[key],
            "delta": q_payload[key] - f_payload[key],
        }
    per_class_key = "test_per_class_f1"
    out["test_per_class_f1"] = {
        name: {
            "float32": f_payload[per_class_key][name],
            "int8": q_payload[per_class_key][name],
            "delta": q_payload[per_class_key][name] - f_payload[per_class_key][name],
        }
        for name in f_payload[per_class_key]
    }
    agreement = float(
        (float32["test"][1] == int8["test"][1]).mean() if len(float32["test"][1]) else 0.0
    )
    out["test_prediction_agreement"] = agreement
    return out


def quantize_checkpoint(
    *,
    model,
    model_name: str,
    model_kwargs: dict,
    dm: IMUEdgeDataModule,
    data_cfg,
    resolved: dict,
    output_dir: Path,
    source_checkpoint: str,
    seed: int,
    calibration_batches: int = DEFAULT_CALIBRATION_BATCHES,
    quant_format: str = "QOperator",
    per_channel: bool = True,
) -> dict:
    """Quantize ``model`` to int8 ONNX, evaluate it, and write a run directory.

    Shared by ``imu-edge-quantize`` and the pipeline's ``--compress`` step so the
    two cannot drift. The caller owns context resolution and the datamodule,
    because the pipeline already has both and rebuilding them would re-read the
    window cache.

    Returns:
        The float32-versus-int8 comparison payload.
    """
    total_context = data_cfg.context_len + data_cfg.future_context_len
    onnx_dir = output_dir / "onnx"
    fp32_path = export_onnx(
        model,
        path=onnx_dir / "model_fp32.onnx",
        context_len=total_context,
        window_size_s=data_cfg.window_size_s,
        n_channels=len(data_cfg.sensor_cols),
    )
    print(f"  exported float32 ONNX: {fp32_path.stat().st_size / 1024:.1f} KiB")

    # Calibration reads the input distribution, so it must come from train.
    #
    # The training loader shuffles, so which batches calibration sees decides the
    # activation ranges and therefore the quantized model. Unseeded, repeat runs
    # of the same checkpoint moved test macro-F1 by up to ~0.006 — the same order
    # as some of the effects being measured. Seed so a run is reproducible.
    set_seed(seed)
    print(f"  calibrating on {calibration_batches} training batches (seed={seed})")
    int8_path = quantize_onnx_static(
        fp32_path,
        onnx_dir / "model_int8.onnx",
        calibration_reader=build_calibration_reader(
            dm.train_loader(), limit=calibration_batches
        ),
        per_channel=per_channel,
        quant_format=quant_format,
    )
    op_counts = onnx_op_counts(int8_path)
    int8_fraction = int8_mac_fraction_from_onnx(int8_path)
    print(
        f"  quantized int8 ONNX: {int8_path.stat().st_size / 1024:.1f} KiB | "
        f"int8 MAC-op share={int8_fraction:.2f} | QLinearConv={op_counts.get('QLinearConv', 0)}"
    )

    print("  evaluating float32 and int8")
    float32_eval = _evaluate(model, dm, n_classes=data_cfg.n_classes)
    quantized = OnnxModule(int8_path)
    int8_eval = _evaluate(quantized, dm, n_classes=data_cfg.n_classes)

    compression = {
        "method": "onnx_static_int8",
        "quant_format": quant_format,
        "per_channel": per_channel,
        "calibration_batches": calibration_batches,
        "calibration_split": "train",
        "onnx_fp32_kib": round(fp32_path.stat().st_size / 1024, 2),
        "onnx_int8_kib": round(int8_path.stat().st_size / 1024, 2),
        "onnx_op_counts": op_counts,
        "int8_mac_op_fraction": int8_fraction,
    }
    # Profiled on the float32 module: torchinfo cannot trace an ONNX session, so
    # params, MACs, and latency describe the source model. The int8 artifact's
    # real size and op mix are recorded in `compression` above.
    #
    # The int8 MAC share must be passed explicitly for the same reason: tracing
    # the float32 source would report 0.0 and credit the deployed graph with
    # float32 energy, even though every convolution in it is a QLinearConv.
    stats = compute_model_stats(
        model,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
        window_size_s=data_cfg.window_size_s,
        n_channels=len(data_cfg.sensor_cols),
        compression=compression,
        hop_size_s=data_cfg.hop_size_s,
        energy_profile=resolved.get("energy"),
        int8_mac_fraction=int8_fraction,
    )

    metrics = {
        "model": model_name,
        "model_kwargs": model_kwargs,
        "dataset": data_cfg.campaign,
        "source_checkpoint": source_checkpoint,
        **int8_eval["payload"],
    }
    predictions = pd.concat(
        [
            predictions_frame(
                split="val",
                y_true=int8_eval["val"][0],
                y_pred=int8_eval["val"][1],
                y_prob=int8_eval["val"][2],
                metadata=dm.split_metadata("val"),
            ),
            predictions_frame(
                split="test",
                y_true=int8_eval["test"][0],
                y_pred=int8_eval["test"][1],
                y_prob=int8_eval["test"][2],
                metadata=dm.split_metadata("test"),
            ),
        ],
        ignore_index=True,
    )
    resolved["data"] = data_cfg.to_dict()
    resolved["compression"] = compression

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

    comparison = _comparison(float32_eval, int8_eval, n_classes=data_cfg.n_classes)
    comparison["compression"] = compression
    (output_dir / "reports" / "quantization_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    test_delta = comparison["test_macro_f1"]
    print(
        f"  test macro-F1: float32={test_delta['float32']:.4f} "
        f"int8={test_delta['int8']:.4f} delta={test_delta['delta']:+.4f} | "
        f"prediction agreement={comparison['test_prediction_agreement']:.4f}"
    )
    return comparison


def main() -> None:
    args = build_parser().parse_args()
    data_cfg, train_cfg, resolved = load_runtime_config(args.config)
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
        model_name, data_cfg.context_len, data_cfg.future_context_len
    )
    check_context_matches_checkpoint(
        model_kwargs,
        context_len=data_cfg.context_len,
        future_context_len=data_cfg.future_context_len,
    )

    run_name = args.run_name or f"{Path(args.checkpoint).parents[1].name}_int8"
    output_dir = Path(train_cfg.output_root) / run_name
    print(
        f"Quantizing {model_name} -> {run_name} | context_len={data_cfg.context_len} "
        f"future={data_cfg.future_context_len} | format={args.quant_format}"
    )

    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    quantize_checkpoint(
        model=model,
        model_name=model_name,
        model_kwargs=model_kwargs,
        dm=dm,
        data_cfg=data_cfg,
        resolved=resolved,
        output_dir=output_dir,
        source_checkpoint=str(Path(args.checkpoint).expanduser()),
        seed=train_cfg.seed,
        calibration_batches=args.calibration_batches,
        quant_format=args.quant_format,
        per_channel=not args.per_tensor,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
