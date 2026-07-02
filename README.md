# distrimuse-imu-edge

Standalone IMU-only edge-model experiments for Distrimuse.

This repo depends on `../distrimuse-core` for shared labels, metrics, storage,
and tracking helpers. It does not import `distrimuse-early-fusion`; IMU
windowing, model definitions, training, distillation, compression, and
benchmark reporting live here.

## V1 Scope

- Fixed canonical participant split.
- Causal context: `context_len=8`, meaning 7 previous windows plus the current
  window.
- PyTorch-side edge metrics: macro F1, per-class F1, confusion matrix arrays,
  parameter counts, serialized size, FLOPs/MACs estimate, and CPU latency.
- Baselines: teacher causal transformer, compact CNN/TCN students, knowledge
  distillation, dynamic quantization, and structured pruning helpers.

## Commands

```bash
uv sync

uv run imu-edge-train --config configs/benchmark.yaml --model teacher_causal_cnn
uv run imu-edge-distill --config configs/benchmark.yaml --teacher-checkpoint path/to/best.ckpt --student edge_tcn
uv run imu-edge-compress --config configs/benchmark.yaml --checkpoint path/to/best.ckpt --model edge_tcn --method dynamic_quant
uv run imu-edge-benchmark --results-dir experiments/results

# WISDM-19 public pretraining → DistriMuSe fine-tuning
uv run imu-edge-prepare-wisdm --root cache/public/wisdm19
uv run imu-edge-train --config configs/pretrain_wisdm19.yaml --model edge_tcn --run-name edge_tcn_wisdm19_pretrain
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn --init-checkpoint experiments/results/edge_tcn_wisdm19_pretrain/checkpoints/best.ckpt --run-name edge_tcn_wisdm19_finetune

# Full scratch vs WISDM-pretrained experiment sweep
scripts/run_wisdm_pretraining_experiments.sh
```

Each run writes:

```text
experiments/results/{run_name}/
├── checkpoints/best.ckpt
├── reports/metrics.json
├── reports/model_stats.json
├── reports/predictions.parquet
├── reports/config.resolved.yaml
└── plots/
```

## Data

The data loader accepts either local split parquet files or processed
participant/session parquets through `distrimuse-core`:

- Local split files: `data.split_dir/{train,val,test}.parquet`
- Processed cache/S3 layout:
  `processed/{campaign}/imu/{processing_version}/person_{id}/scenario_{id}/data.parquet`
- WISDM-19 public pretraining splits:
  `cache/public/wisdm19/splits/watch_accel_gyro/{train,val,test}.parquet`

If `data.manifest_path` is set, window timestamps and labels come from the
manifest. Otherwise, windows are created by majority vote over the configured
label column.

## Configuration

Defaults live in `configs/`. `configs/benchmark.yaml` composes the practical
V1 benchmark defaults:

- `window_size_s=2.0`
- `hop_size_s=0.5`
- `context_len=8`
- `task_col=big_movement`
- train/val/test participants from the canonical early-fusion split

## Tests

```bash
uv run pytest
```
