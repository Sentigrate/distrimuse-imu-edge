# distrimuse-imu-edge

IMU-only edge-model experiments for the Distrimuse activity-recognition pipeline.

This repository trains, distills, compresses, and benchmarks PyTorch models that
use only wearable IMU channels. The goal is to find models that are accurate
enough for Distrimuse activity recognition while staying small and fast enough
for edge deployment.

## What This Repo Does

- Builds fixed-length IMU windows from processed Distrimuse parquet data.
- Trains IMU classifiers for the configured task, currently `big_movement`.
- Supports causal context models that see the current window plus previous
  windows, matching streaming inference constraints.
- Distills a larger teacher into compact student models.
- Applies compression helpers such as dynamic quantization and structured
  pruning.
- Writes comparable benchmark reports: macro F1, per-class F1, confusion
  matrices, prediction parquet files, parameter counts, serialized model size,
  FLOPs/MACs estimates, and CPU latency.

This repository does not do radar fusion, synthetic-data generation, or firmware
integration. Those live in separate Distrimuse repositories.

## Repository Layout

```text
configs/                         Experiment and data configs
scripts/                         Reproducible experiment sweeps
src/distrimuse_imu_edge/
  cli/                           Command-line entry points
  data/                          Split loading, windowing, sequence datasets
  models/                        Teacher, baseline, and edge architectures
  training/                      Supervised training and distillation
  compression/                   Quantization and pruning helpers
  evaluation/                    Metrics, model stats, plots, aggregation
tests/                           Unit and smoke tests
experiments/results/             Generated outputs, ignored except .gitkeep
cache/                           Generated window/data cache, ignored
```

## Requirements

- Python 3.11 or newer.
- `uv`.
- A sibling checkout of `distrimuse-core`, because `pyproject.toml` installs it
  as an editable path dependency:

```text
distrimuse/
  distrimuse-core/
  distrimuse-imu-edge/
```

Install dependencies from the repository root:

```bash
uv sync
```

## Data

The default benchmark config expects prebuilt local split files:

```text
../distrimuse-early-fusion/cache/datasets/v35/imu/
  train.parquet
  val.parquet
  test.parquet
```

Each split parquet should contain:

- `person_id`
- `scenario_id` if multiple recordings are present
- IMU channels from `data.sensor_cols`, by default
  `acc_x`, `acc_y`, `acc_z`, `gyr_x`, `gyr_y`, `gyr_z`
- the task label column from `data.task_col`, by default `big_movement`
- an optional timestamp column, one of `imu_s`, `timestamp_dt`, `timestamp`,
  `Timestamp`, `timestamp_s`, `time`, or `Time`

Labels must be integer class IDs in `[0, data.n_classes)`. Negative labels are
skipped during window creation. If no timestamp column is present, the windowing
code assumes the Distrimuse IMU sampling rate from `distrimuse-core`.

Instead of `data.split_dir`, you can set `data.processing_version` and
`data.cache_dir`. In that mode the loader reads or downloads processed sessions
with this layout:

```text
processed/{campaign}/imu/{processing_version}/person_{id}/scenario_{id}/data.parquet
```

The loader creates 3-second windows every 1 second by default, resamples each
window to the expected sample count, assigns labels by majority vote, and fits
channel normalization on the training windows only. `data.context_len` includes
the current window (for example, `8` means seven past windows plus the current
window). `data.future_context_len` optionally adds look-ahead windows; its
default is `0`. Single-window baselines such as `cnn_har` and `tinierhar` force
both settings to `1` and `0`, respectively.

## Common Workflows

Run commands from the repository root.

### 1. Verify the Code

```bash
uv run pytest
```

### 2. Train One Model

```bash
uv run imu-edge-train \
  --config configs/benchmark.yaml \
  --model edge_tcn \
  --run-name edge_tcn_baseline
```

Useful overrides:

```bash
uv run imu-edge-train --config configs/benchmark.yaml --model edge_cnn --width-mult 0.25
uv run imu-edge-train --config configs/benchmark.yaml --model teacher_causal_cnn --max-epochs 5
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn --device cpu
```

Compare Edge TCN context modes:

```bash
# Current window only
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn \
  --context-len 1 --future-context-len 0 --run-name edge_tcn_current

# Seven past windows plus current
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn \
  --context-len 8 --future-context-len 0 --run-name edge_tcn_past7_current

# Seven past windows, current, and seven future windows
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn \
  --context-len 8 --future-context-len 7 --run-name edge_tcn_past7_current_future7
```

Future context is non-causal: with the default one-second hop, seven future
windows add seven seconds of decision delay.

### 3. Run the Teacher-Student Benchmark Pipeline

This is the easiest way to reproduce the intended V1 experiment:

```bash
uv run imu-edge-pipeline --config configs/benchmark.yaml --compress
```

The pipeline trains the teacher, distills configured students across configured
width multipliers, optionally dynamic-quantizes the distilled checkpoints, then
aggregates the benchmark table and plots.

For a quick smoke run:

```bash
uv run imu-edge-pipeline \
  --config configs/benchmark.yaml \
  --students edge_tcn \
  --width-mults 0.25 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --compress
```

### 4. Distill or Compress Manually

```bash
uv run imu-edge-distill \
  --config configs/benchmark.yaml \
  --teacher-checkpoint experiments/results/teacher_causal_cnn_wm1_ctx8/checkpoints/best.ckpt \
  --student edge_tcn \
  --width-mult 0.5

uv run imu-edge-compress \
  --config configs/benchmark.yaml \
  --checkpoint experiments/results/edge_tcn_wm0.5_ctx8_distilled/checkpoints/best.ckpt \
  --method dynamic_quant

uv run imu-edge-benchmark --results-dir experiments/results
```

### 5. Train With Synthetic Augmentation

Synthetic data is produced in the sibling `distrimuse-synthetic-data`
repository. Once that repo has exported flat split files, use:

```bash
uv run imu-edge-train \
  --config configs/synthetic_augmented.yaml \
  --model edge_tcn \
  --run-name edge_tcn_synthetic_augmented
```

`configs/synthetic_augmented.yaml` points to:

```text
../distrimuse-synthetic-data/cache/imu_edge_synthetic_augmented/v35
```

That directory should contain `train.parquet`, `val.parquet`, and
`test.parquet`; only the training split should contain synthetic recordings.

### 6. WISDM-19 Pretraining

Prepare public WISDM-19 watch accel/gyro splits:

```bash
uv run imu-edge-prepare-wisdm --root cache/public/wisdm19
```

Pretrain and fine-tune:

```bash
uv run imu-edge-train \
  --config configs/pretrain_wisdm19.yaml \
  --model edge_tcn \
  --run-name edge_tcn_wisdm19_pretrain

uv run imu-edge-train \
  --config configs/benchmark.yaml \
  --model edge_tcn \
  --init-checkpoint experiments/results/edge_tcn_wisdm19_pretrain/checkpoints/best.ckpt \
  --run-name edge_tcn_wisdm19_finetune
```

The full scratch-vs-pretrained sweep is also wrapped in:

```bash
scripts/run_wisdm_pretraining_experiments.sh
```

## Models

Registered model IDs:

- `teacher_causal_cnn`: larger causal teacher used for distillation.
- `causal_context_transformer_cnn`: causal context model with attention over
  recent windows.
- `edge_cnn`: compact CNN student.
- `edge_tcn`: compact temporal convolution student.
- `cnn_har`: single-window CNN-HAR baseline.
- `tinierhar`: single-window TinierHAR baseline.

The default benchmark uses `teacher_causal_cnn`, `edge_cnn`, and `edge_tcn`
with width multipliers `0.25`, `0.5`, and `1.0`.

## Outputs

Each run writes to `experiments/results/{run_name}/`:

```text
checkpoints/best.ckpt
reports/metrics.json
reports/model_stats.json
reports/predictions.parquet
reports/config.resolved.yaml
reports/test_per_subject_metrics.{json,csv}
confusion_matrices/test_all_subjects.html
confusion_matrices/test_subject_{person_id}.html
confusion_matrices/test_subjects_overview.html
plots/prediction_timeline_subject_{person_id}.html
plots/index.html
```

The aggregate benchmark command writes:

```text
experiments/results/benchmark_summary.csv
experiments/results/f1_bar.html
experiments/results/f1_vs_gflops.html
experiments/results/f1_vs_latency.html
experiments/results/f1_vs_size.html
```

Generated caches, checkpoints, metrics, plots, and benchmark outputs are ignored
by Git.

## Configuration Notes

Main knobs live under `data`, `train`, and `benchmark` in `configs/*.yaml`.

Important defaults in `configs/benchmark.yaml`:

- `data.window_size_s: 3.0`
- `data.hop_size_s: 1.0`
- `data.context_len: 8`
- `data.future_context_len: 0`
- `data.task_col: big_movement`
- `data.n_classes: 9`
- `train.max_epochs: 30`
- `train.output_root: experiments/results`
- `benchmark.models: [teacher_causal_cnn, edge_cnn, edge_tcn]`

If you change the data source or windowing parameters and want to rebuild cached
windows, either set `data.reuse_window_cache: false` or delete the relevant
files under `cache/windows/`.
