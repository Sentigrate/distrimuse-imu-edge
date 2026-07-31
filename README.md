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
  windows, matching streaming inference constraints, plus explicit
  bidirectional look-ahead experiments.
- Distills a larger teacher into compact student models.
- Quantizes trained models to int8 via export-time static post-training
  quantization, and applies structured pruning.
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
  compression/                   Int8 ONNX quantization and pruning helpers
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
default is `0`. Single-window models (`edge_cnn`, `edge_tcn`, `cnn_har`, and
`tinierhar`) force both settings to `1` and `0`, respectively. Use
`edge_window_gru` or `edge_window_tcn` for temporal context.

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
  --model edge_window_tcn \
  --run-name edge_window_tcn_baseline
```

Useful overrides:

```bash
uv run imu-edge-train --config configs/benchmark.yaml --model edge_cnn --width-mult 0.25
uv run imu-edge-train --config configs/benchmark.yaml --model teacher_causal_cnn --max-epochs 5
uv run imu-edge-train --config configs/benchmark.yaml --model edge_tcn --device cpu
```

Compare context modes with a per-window encoder and temporal TCN:

```bash
# Current window only
uv run imu-edge-train --config configs/benchmark.yaml --model edge_window_tcn \
  --context-len 1 --future-context-len 0 --run-name edge_window_tcn_current

# Seven past windows plus current
uv run imu-edge-train --config configs/benchmark.yaml --model edge_window_tcn \
  --context-len 8 --future-context-len 0 --run-name edge_window_tcn_past7_current

# Seven past windows, current, and seven future windows
uv run imu-edge-train --config configs/benchmark.yaml --model edge_window_tcn \
  --context-len 8 --future-context-len 7 \
  --run-name edge_window_tcn_past7_current_future7
```

Future context is non-causal: with the default one-second hop, seven future
windows add seven seconds of decision delay. The teacher and window-sequence
students always classify the explicit current position. They use causal
aggregation without future context and bidirectional aggregation when future
context is configured.

### 3. Run the Teacher-Student Benchmark Pipeline

This is the easiest way to reproduce the intended V1 experiment:

```bash
uv run imu-edge-pipeline --config configs/benchmark.yaml --compress
```

The pipeline trains the teacher, distills configured students across configured
width multipliers, optionally quantizes the distilled checkpoints to int8 ONNX,
then aggregates the benchmark table and plots.

For a quick smoke run:

```bash
uv run imu-edge-pipeline \
  --config configs/benchmark.yaml \
  --students edge_window_tcn \
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
  --student edge_window_tcn \
  --width-mult 0.5

uv run imu-edge-quantize \
  --config configs/benchmark.yaml \
  --checkpoint experiments/results/edge_window_tcn_wm0.5_ctx8_distilled/checkpoints/best.ckpt

uv run imu-edge-compress \
  --config configs/benchmark.yaml \
  --checkpoint experiments/results/edge_window_tcn_wm0.5_ctx8_distilled/checkpoints/best.ckpt \
  --method structured_prune --prune-amount 0.25

uv run imu-edge-benchmark --results-dir experiments/results
```

### 5. Train With Synthetic Augmentation

Synthetic data is produced in the sibling `distrimuse-synthetic-data`
repository. Once that repo has exported flat split files, use:

```bash
uv run imu-edge-train \
  --config configs/synthetic_augmented.yaml \
  --model edge_window_tcn \
  --run-name edge_window_tcn_synthetic_augmented
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
  --model edge_window_tcn \
  --run-name edge_window_tcn_wisdm19_pretrain

uv run imu-edge-train \
  --config configs/benchmark.yaml \
  --model edge_window_tcn \
  --init-checkpoint experiments/results/edge_window_tcn_wisdm19_pretrain/checkpoints/best.ckpt \
  --run-name edge_window_tcn_wisdm19_finetune
```

The full scratch-vs-pretrained sweep is also wrapped in:

```bash
scripts/run_wisdm_pretraining_experiments.sh
```

## Models

Registered model IDs:

- `teacher_causal_cnn`: larger per-window CNN plus temporal Transformer teacher.
- `causal_context_transformer_cnn`: configurable causal/bidirectional
  per-window Transformer.
- `edge_window_gru`: compact per-window CNN plus temporal GRU student.
- `edge_window_tcn`: compact per-window CNN plus temporal embedding-TCN student.
- `edge_cnn`: compact single-window CNN student.
- `edge_tcn`: compact single-window temporal convolution student.
- `cnn_har`: single-window CNN-HAR baseline.
- `tinierhar`: single-window TinierHAR baseline.

The default benchmark uses `teacher_causal_cnn`, `edge_window_gru`, and
`edge_window_tcn` with width multipliers `0.25`, `0.5`, and `1.0`.

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

## Energy Estimates

`reports/model_stats.json` carries an `energy` block alongside the parameter,
size, MAC, and latency fields:

```json
"energy": {
  "energy_per_inference_mj": 18.447176,
  "avg_power_mw": 18.447952,
  "est_battery_life_h": 36.589,
  "est_battery_life_days": 1.525,
  "active_time_per_inference_ms": 922.358781,
  "duty_cycle": 0.922359,
  "real_time_feasible": true,
  "hop_size_s": 1.0,
  "numeric_format": "float32",
  "int8_mac_fraction": 0.0,
  "assumptions": { "name": "nrf52840_m4f_64mhz", "...": "..." }
}
```

The model is the standard duty-cycle estimate used in embedded engineering:

```text
t_active     = (macs_int8 / mpc_int8 + macs_f32 / mpc_f32) / f_clock
E_inference  = P_active * t_active
P_avg        = P_active * duty + P_sleep * (1 - duty)    duty = t_active / hop
battery_life = capacity_mAh * V / P_avg
```

`hop_size_s` comes from `data.hop_size_s`, since one prediction is emitted per
hop.

`int8_mac_fraction` is **measured from the traced layers**, not inferred from the
compression label. int8 kernels sustain roughly 4x the MACs per cycle that
float32 kernels do on a scalar FPU, so the split matters — but partial
quantisation is the normal case, and a label does not say how much of the
arithmetic was actually converted. It is also why PyTorch dynamic quantisation
was removed from this repo: it converts `Linear`/`GRU` and not `Conv1d`, so
`edge_window_tcn` — about 99% Conv1d MACs — came out at roughly `0.0` int8 and a
`1.8%` smaller state dict. Real int8 for a convolutional model has to come from
export-time quantisation, which is what `imu-edge-quantize` does; `compute_model_stats`
accepts an explicit `int8_mac_fraction` so the exported artifact's measured share
is credited rather than the traced float32 source's.

### Read this before quoting the numbers

- **It does not rank models differently from `gmacs`.** Every other term is
  constant across models, so energy is exactly proportional to MACs. The value
  is unit translation — "1.5 days on a coin cell" instead of "0.0295 GMACs" —
  not a new comparison axis.
- **It is an assumption, not a measurement.** Profile values are
  order-of-magnitude datasheet-class figures for a part *class*, not readings
  from a board. The full profile is echoed into `assumptions` so any reader can
  audit it.
- **It covers inference only.** Continuous 104 Hz IMU sampling, sensor
  front-end power, and radio traffic are excluded, and on a real wearable those
  terms are often larger. `est_battery_life_h` is not a device-level battery
  projection.
- **It ignores memory movement.** `P_active` is one constant, so a
  memory-bound and a compute-bound model with equal MACs get equal energy. The
  models that handle this properly (Yang, Chen & Sze, CVPR 2017;
  Accelergy/Timeloop) weight per-level memory accesses explicitly.

For a figure that survives review, measure on the target with a power analyser
(Joulescope, Otii Arc, Nordic PPK2) or use the MLPerf Tiny energy harness.

### Selecting a profile

```yaml
energy:
  profile: nrf52840_m4f_64mhz
```

Bundled profiles: `nrf52840_m4f_64mhz` (default), `stm32l4_m4f_80mhz`,
`stm32u5_m33_160mhz`, `ethos_u55_64_200mhz`. Any numeric field can be
overridden once you have measured your own hardware:

```yaml
energy:
  profile: nrf52840_m4f_64mhz
  p_active_mw: 17.5
  battery_capacity_mah: 100.0
  battery_voltage_v: 3.7
```

An overridden profile is reported as `<base>+overrides` so it can never be
mistaken for a stock profile. Unknown profile names and unknown field names
fail at config-load time. See
[energy.py](src/distrimuse_imu_edge/evaluation/energy.py) for profile
definitions and the reasoning behind each value.

`benchmark_summary.csv` surfaces `avg_power_mw`, `est_battery_life_h`,
`duty_cycle`, and `energy_profile`. Runs recorded before energy reporting
existed aggregate with nulls in those columns.

## Configuration Notes

Main knobs live under `data`, `train`, `energy`, and `benchmark` in
`configs/*.yaml`.

Important defaults in `configs/benchmark.yaml`:

- `data.window_size_s: 3.0`
- `data.hop_size_s: 1.0`
- `data.context_len: 8`
- `data.future_context_len: 0`
- `data.task_col: big_movement`
- `data.n_classes: 9`
- `train.max_epochs: 30`
- `train.output_root: experiments/results`
- `energy.profile: nrf52840_m4f_64mhz`
- `benchmark.models: [teacher_causal_cnn, edge_window_gru, edge_window_tcn]`

If you change the data source or windowing parameters and want to rebuild cached
windows, either set `data.reuse_window_cache: false` or delete the relevant
files under `cache/windows/`.
