# Edge Window TCN compact temporal-context study

**Model:** `edge_window_tcn` · **Dataset:** `dc-extern-2026-01` · **Task:** 9-class big-movement recognition

**Windowing:** 3 s windows at 104 Hz, 1 s hop · **Main model width:** `0.25` · **Evaluation:** five held-out test subjects

## Executive summary

The complete width-0.25 ablation shows that temporal context is useful even in a model small enough for the 0.1 MB target.

- All three compact models contain **16,569 parameters** and have a **0.084 MB** state dict. Changing context reuses the same weights, so it changes compute and latency but not model size.
- Adding seven past windows raised test macro-F1 from **0.512 to 0.552** (`+0.039`) without look-ahead. The gain appeared in 18 of 24 subject-scenario recordings.
- Adding seven future windows raised macro-F1 from **0.552 to 0.694** (`+0.142`). The past 7 + current + future 7 model beat the past-only model in 22 of 24 recordings and the current-only model in all 24.
- More context increases execution cost rather than storage: current-only, past 7 + current, and past 7 + current + future 7 inference require `0.0010`, `0.0083`, and `0.0155` GFLOPs, with measured median CPU latencies of `0.267`, `0.983`, and `1.564 ms`.
- For a strict sub-0.1 MB online model, **width 0.25 with seven past windows + current** is the best measured causal choice. For offline or delayed inference, **width 0.25 with past, current, and future context** is clearly strongest.
- As a secondary comparison, width 0.25 essentially matched width 0.5 for past 7 + current + future 7 input (`0.694` versus `0.690` macro-F1) while using **73.7% fewer parameters** and **72.0% fewer FLOPs**; however, its causal past-context score was `0.061` lower.

![Width-0.25 test performance across context modes](edge_window_tcn_context_report_assets/performance-comparison.svg)

## Experimental design

The compact CNN encodes every 3 s window independently. A temporal TCN then operates over the sequence of window embeddings, and the classifier reads the embedding at the explicit current position.

| Mode | CLI context | Windows per prediction | Unique temporal span | Added look-ahead | Current index | Temporal TCN |
|---|---:|---:|---:|---:|---:|---|
| Current only | `context_len=1`, `future=0` | 1 | 3 s | 0 s | 0 | Causal |
| Past 7 + current | `context_len=8`, `future=0` | 8 | 10 s | 0 s | 7 | Causal |
| Past 7 + current + future 7 | `context_len=8`, `future=7` | 15 | 17 s | 7 s | 7 | Non-causal |

The temporal span is shorter than `number of windows × 3 s` because adjacent windows overlap by 2 s. Added look-ahead is measured relative to the end of the current window.

All main runs used the same data split and training settings:

| Setting | Value |
|---|---|
| Train / validation / test windows | 40,008 / 6,361 / 10,628 |
| Train / validation / test subjects | 18 / 3 / 5 |
| Test subjects | 8, 15, 24, 26, 27 |
| Sensor input | Accelerometer + gyroscope, 6 channels |
| Width multiplier | `0.25` |
| Optimizer settings | Learning rate `1e-3`, weight decay `1e-4` |
| Maximum epochs / early-stop patience | 30 / 8 |
| Random seed | 42 |

The width-0.5 runs use the same split, optimizer settings, and seed and are retained later as a matched capacity comparison.

## Architecture: how Edge Window TCN works

`edge_window_tcn` separates **within-window feature extraction** from **between-window temporal reasoning**. Rather than concatenating all context into one long raw signal, it first converts every window into an embedding using a shared encoder and then models relationships between those embeddings.

```mermaid
flowchart LR
    A["Input sequence<br/>B × N × 312 × 6"] --> B["Flatten windows<br/>(B·N) × 6 × 312"]
    B --> C["Shared CNN window encoder"]
    C --> D["Window embeddings<br/>B × N × D"]
    D --> E{"Future context?"}
    E -- "No" --> F["Causal embedding TCN<br/>dilations 1, 2, 4"]
    E -- "Yes" --> G["Non-causal embedding TCN<br/>dilations 1, 2, 4"]
    F --> H["Select current position<br/>index = context_len − 1"]
    G --> H
    H --> I["LayerNorm + dropout<br/>linear classifier"]
    I --> J["9 class logits"]
```

### 1. Sequence construction

For a target window at dataset index `i`, the loader creates:

```text
[batch, total windows, samples per window, sensor channels]
```

Here the shape is `[B, N, 312, 6]`, where `312 = 3 s × 104 Hz` and:

```text
N = context_len + future_context_len
```

`context_len` includes the current window. Therefore, `context_len=8` selects `i−7 ... i`; adding `future_context_len=7` extends the sequence through `i+7`. Sequences never cross person or scenario boundaries. Missing positions at a recording boundary are zero-padded, while the target always remains the label of window `i`.

### 2. Shared per-window CNN encoder

The batch and window dimensions are combined temporarily, so every window passes through one shared encoder as `[B·N, 6, 312]`.

| Layer | Operation | Width 0.5 | Width 0.25 |
|---|---|---:|---:|
| 1 | 1D convolution, kernel 7 + batch norm + ReLU | 32 | 16 |
| 2 | 1D convolution, kernel 5 + batch norm + ReLU | 32 | 16 |
| 3 | Max-pool, stride 2 | 32 | 16 |
| 4 | 1D convolution, kernel 5 + batch norm + ReLU | 64 | 32 |
| 5 | Adaptive average pool over the 3 s time axis | 64 | 32 |
| 6 | Linear projection | 48 | 24 |

The result is reshaped to `[B, N, D]`: one embedding per window, with `D=48` at width 0.5 and `D=24` at width 0.25. Encoder weights are shared across positions, so adding windows increases compute but not parameter count.

### 3. What the width multiplier controls

The width multiplier scales the internal channel counts:

```python
scaled_width = max(8, round(base_width * width_mult))
```

For `edge_window_tcn`, the base CNN widths are 64 and 128, while the embedding and temporal-TCN widths are 96. The multiplier does **not** change the layer count, kernels, dilations, context length, sampling rate, or output classes.

| Width multiplier | CNN channels | Embedding / TCN width | Window encoder params | Temporal TCN params | Head params | Total params |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 64 → 128 | 96 | 77,280 | 167,616 | 1,065 | 245,961 |
| 0.5 | 32 → 64 | 48 | 20,208 | 42,336 | 537 | 63,081 |
| 0.25 | 16 → 32 | 24 | 5,496 | 10,800 | 273 | 16,569 |

Halving the width approximately quarters the dominant convolutional parameter count because convolution weights scale with both input and output channels. The reduction is not exactly fourfold because the fixed six sensor inputs, nine output classes, biases, and normalization parameters do not all scale quadratically.

Width 1.0 is the unscaled architecture. It would contain 245,961 parameters, occupy approximately 1.004 MB, and require about 0.2079 GFLOPs for a 15-window past 7 + current + future 7 input. It is not a candidate for the current 0.1 MB target.

### 4. Temporal TCN over window embeddings

The embeddings are transposed to `[B, D, N]` and passed through three residual TCN blocks. Every block contains two kernel-size-3 convolutions, per-position layer normalization, ReLU, dropout, and a residual shortcut. Block dilations are `1`, `2`, and `4`, so the network combines nearby and more distant window embeddings without a recurrent loop.

The stacked theoretical receptive field is 29 window positions: up to 28 previous positions in causal mode, or up to 14 positions on either side in non-causal mode. The longest experiment supplies 15 windows, so the temporal stack can connect the selected current position to the complete supplied sequence.

The padding mode controls what information is legal:

- **No future windows:** left-only padding makes the temporal convolutions causal. The current representation can use current and earlier embeddings, never later ones.
- **Future windows present:** symmetric padding makes the temporal convolutions non-causal. The current representation can combine embeddings from both sides.

“Non-causal” describes the information flow, not a second set of model weights. The causal and past 7 + current + future 7 variants therefore have the same parameter count.

### 5. Explicit current-position classification

After temporal processing, the model selects:

```python
current_index = context_len - 1
current_embedding = temporal[:, current_index]
```

This gives index 0 for current-only and index 7 for both seven-past experiments. In the 15-window past 7 + current + future 7 sequence, the classifier reads the actual current position rather than the final future window.

The selected vector passes through layer normalization, 10% dropout, and a linear `D → 9` head. Training uses class-weighted cross-entropy. During evaluation, softmax produces class probabilities and the highest-probability class becomes the prediction.

With `N=1`, the same model has no neighboring embeddings to exploit. This makes current-only a controlled baseline for measuring the value of past and future context.

## Main width-0.25 test results

| Context mode | Best validation macro-F1 | Test macro-F1 | Accuracy | Weighted F1 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Current only | 0.533 | 0.512 | 0.700 | 0.724 | 0.729 |
| Past 7 + current | 0.612 | 0.552 | 0.731 | 0.749 | 0.792 |
| Past 7 + current + future 7 | **0.667** | **0.694** | **0.812** | **0.818** | **0.887** |

### Effect of adding past context

Relative to current-only classification, seven past windows added:

- `+0.039` macro-F1
- `+3.05` percentage points accuracy
- `+0.025` weighted F1

Past context helps the compact model, but the gain is modest compared with the earlier width-0.5 result. It is still deployment-friendly: the model gets a 10 s historical view while remaining causal and incurring no algorithmic look-ahead.

### Effect of adding future context

Relative to the compact causal past-context model, seven future windows added:

- `+0.142` macro-F1
- `+8.15` percentage points accuracy
- `+0.069` weighted F1

Relative to current-only, the complete past 7 + current + future 7 gain is `+0.181` macro-F1 and `+11.20` percentage points accuracy. The future sequence is especially valuable for ambiguous transitions because it reveals the state reached after the current action. This benefit requires waiting for seven future hops, or 7 s.

### Paired robustness check

A descriptive paired bootstrap over 24 aligned subject-scenario test recordings produced:

| Comparison | Macro-F1 difference | 95% cluster-bootstrap interval | Recording wins |
|---|---:|---:|---:|
| Past 7 + current − current | +0.039 | [0.014, 0.067] | 18 / 24 |
| Past 7 + current + future 7 − current | +0.181 | [0.137, 0.224] | 24 / 24 |
| Past 7 + current + future 7 − past 7 + current | +0.142 | [0.108, 0.180] | 22 / 24 |

The past-context gain is positive but less universal than the future-context gain. Past 7 + current + future 7 wins consistently enough that the result is not driven by one subject or scenario. The intervals remain descriptive rather than a substitute for repeated-seed experiments: recordings from the same subject are correlated, and adjacent windows overlap.

## Consistency across test subjects

![Per-subject macro-F1 for the width-0.25 context ablation](edge_window_tcn_context_report_assets/per-subject-comparison.svg)

| Subject | Current | Past 7 + current | Past 7 + current + future 7 |
|---:|---:|---:|---:|
| 8 | 0.400 | 0.526 | **0.631** |
| 15 | **0.511** | 0.505 | 0.731 |
| 24 | 0.498 | 0.486 | **0.558** |
| 26 | 0.641 | 0.741 | **0.790** |
| 27 | 0.476 | 0.523 | **0.710** |

Past context improves subjects 8, 26, and 27, while subjects 15 and 24 decline slightly. Adding future 7 then improves every subject over past 7 + current. Subject 26 remains easiest and subject 24 remains hardest, so temporal context reduces but does not remove subject variability.

## Per-class behavior

![Per-class F1 for the width-0.25 context ablation](edge_window_tcn_context_report_assets/per-class-f1.svg)

| Class | Test support | Current | Past 7 + current | Past 7 + current + future 7 |
|---|---:|---:|---:|---:|
| Not Moving | 4,201 | 0.811 | 0.817 | **0.842** |
| Walk | 2,874 | 0.844 | 0.844 | **0.871** |
| Sit Down | 294 | 0.577 | 0.615 | **0.712** |
| Lay Down | 255 | 0.197 | 0.241 | **0.490** |
| Turn | 873 | 0.542 | 0.645 | **0.764** |
| Sit Up | 228 | 0.209 | 0.337 | **0.601** |
| Stand Up | 429 | 0.432 | 0.507 | **0.737** |
| Falling | 86 | 0.347 | 0.251 | **0.386** |
| Hand | 1,388 | 0.652 | 0.708 | **0.839** |

Past context improves eight of nine class F1 scores; Falling is the exception (`0.347 → 0.251`). Adding future 7 improves all nine classes over past 7 + current. The largest future-context gains are on temporal transition classes: Lay Down, Sit Up, Stand Up, and Hand.

Falling remains the weakest class with past 7 + current + future 7 and has only 86 test windows. Its non-monotonic result across context modes is a warning against overinterpreting one small class from one seed. It still needs more examples, targeted sampling or loss work, and inspection of the saved confusion matrices.

## Model size, FLOPs, and latency

![Macro-F1 versus measured CPU latency for width 0.25](edge_window_tcn_context_report_assets/accuracy-latency-tradeoff.svg)

| Context mode | Parameters | State-dict size | GFLOPs / inference | CPU median | CPU p95 |
|---|---:|---:|---:|---:|---:|
| Current only | 16,569 | 0.084 MB | 0.0010 | 0.267 ms | 0.276 ms |
| Past 7 + current | 16,569 | 0.084 MB | 0.0083 | 0.983 ms | 1.055 ms |
| Past 7 + current + future 7 | 16,569 | 0.084 MB | 0.0155 | 1.564 ms | 1.645 ms |

GFLOPs are reported using the project’s standard `gflops` profiler field. One GFLOP is one billion floating-point operations.

Parameter count and storage remain constant because the same CNN encoder and temporal TCN weights are reused at every position. Compute grows approximately with the number of input windows:

- Past 7 + current uses `8.08×` the FLOPs of current-only.
- Past 7 + current + future 7 uses `15.14×` the FLOPs of current-only and `1.87×` those of past 7 + current.
- Measured median CPU latency grows by `3.68×` for past 7 + current and `5.86×` for past 7 + current + future 7 relative to current-only.

Latency grows more slowly than FLOPs because fixed framework and operator overhead dominate very small batch-size-one inference. The timings use five warm-up passes and 30 timed passes on the run host; they are useful for comparison, not a target-device guarantee.

End-to-end response time differs from model execution time. Every mode first needs the 3 s current window. The past 7 + current + future 7 model then requires an additional 7 s of future data, making its approximately 1.56 ms neural-network latency negligible beside its algorithmic look-ahead delay.

## Training behavior and checkpoint selection

![Validation macro-F1 for the width-0.25 runs](edge_window_tcn_context_report_assets/validation-training-curves.svg)

| Context mode | Best epoch | Best validation macro-F1 | Final epoch | Final train loss | Final validation macro-F1 | Stopping condition |
|---|---:|---:|---:|---:|---:|---|
| Current only | 13 | 0.533 | 21 | 0.895 | 0.527 | Early stop |
| Past 7 + current | 9 | 0.612 | 17 | 0.415 | 0.599 | Early stop |
| Past 7 + current + future 7 | 26 | 0.667 | 30 | 0.209 | 0.643 | Maximum epochs |

The saved best checkpoint was restored before test evaluation, so test results use the selected checkpoint rather than the final epoch. The past 7 + current + future 7 run continued improving much longer than the causal runs and fit the training set more strongly.

Mean confidence exceeds accuracy by 2.9, 6.1, and 7.4 percentage points for current-only, past 7 + current, and past 7 + current + future 7 input. Confidence becomes more optimistic as context grows. If probabilities drive alarms or thresholds, calibration should be measured with reliability diagrams, expected calibration error, and held-out temperature scaling.

## Comparison with width 0.5

The matched width-0.5 results are useful for separating the effect of temporal context from model capacity. They are retained here as a secondary comparison; the report’s main figures and conclusions above use width 0.25.

![Macro-F1, median CPU latency, and GFLOPs for widths 0.25 and 0.5](edge_window_tcn_context_report_assets/width-latency-comparison.svg)

Color identifies the context mode, while squares represent width 0.5 and circles represent width 0.25; bubble area and the adjacent labels show GFLOPs. The paired points make the context-dependent width effect visible: the compact past 7 + current model is slower and less accurate on this host, whereas the two past 7 + current + future 7 models have nearly identical latency and macro-F1 despite the compact model’s much lower FLOPs.

![Width-0.25 and width-0.5 results across matched contexts](edge_window_tcn_context_report_assets/width-comparison.svg)

| Context mode | Macro-F1, wm=0.5 | Macro-F1, wm=0.25 | Compact − wider | GFLOPs, wm=0.5 | GFLOPs, wm=0.25 | FLOPs reduction |
|---|---:|---:|---:|---:|---:|---:|
| Current only | 0.488 | **0.512** | +0.025 | 0.0036 | 0.0010 | 71.9% |
| Past 7 + current | **0.613** | 0.552 | −0.061 | 0.0295 | 0.0083 | 72.0% |
| Past 7 + current + future 7 | 0.690 | **0.694** | +0.004 | 0.0553 | 0.0155 | 72.0% |

Across all contexts, width 0.25 reduces parameters from 63,081 to 16,569 (`−73.7%`) and state-dict size from 0.271 to 0.084 MB (`−69.0%`). Its effect on accuracy is not uniform:

- **Current only:** the compact model is `+0.025` macro-F1 higher, but the paired interval includes zero.
- **Past 7 + current:** the compact model is `−0.061` lower, with a paired interval fully below zero.
- **Past 7 + current + future 7:** the compact and wider models are effectively tied.

| Compact width 0.25 − width 0.5 | Macro-F1 difference | 95% cluster-bootstrap interval | Compact recording wins |
|---|---:|---:|---:|
| Current only | +0.025 | [−0.002, 0.050] | 14 / 24 |
| Past 7 + current | −0.061 | [−0.082, −0.040] | 5 / 24 |
| Past 7 + current + future 7 | +0.004 | [−0.026, 0.038] | 10 / 24 |

Past 7 + current + future 7 apparently supplies enough temporal evidence that the smaller representation is sufficient. The causal past-only case appears more capacity-sensitive: width 0.5 can use the historical sequence substantially better. Because every result is from one random seed, repeated runs are still needed before treating this interaction as a stable architectural property.

Width reduction also does not guarantee lower host CPU latency. At width 0.25, current-only latency rises slightly (`0.249 → 0.267 ms`), past 7 + current rises (`0.699 → 0.983 ms`), and past 7 + current + future 7 latency is unchanged (`1.562 → 1.564 ms`). The compact model lowers memory and arithmetic requirements, but runtime kernels, tensor shapes, and fixed overhead determine whether that theoretical saving appears on a particular processor.

## Architecture validation

The earlier raw-context `edge_tcn` flattened temporal context into the signal path and did not benefit from additional windows:

| Architecture | Current | Past 7 + current | Past 7 + current + future 7 |
|---|---:|---:|---:|
| Raw-context `edge_tcn` | 0.512 | 0.411 | 0.411 |
| Window-encoder `edge_window_tcn`, wm=0.25 | **0.512** | **0.552** | **0.694** |

The window-encoder architecture matches the raw model for current-only input and improves by `+0.141` with past 7 + current and `+0.283` with past 7 + current + future 7. This supports the design: encode every window consistently, model relationships between window embeddings, and classify the explicit current position.

## Recommendation

**Strict sub-0.1 MB online deployment:** use **Past 7 + current at width 0.25**. It remains causal, fits in 0.084 MB, and improves macro-F1 over compact current-only from `0.512` to `0.552`.

**Accuracy-first causal deployment:** if 0.271 MB is acceptable, the width-0.5 past-context model is stronger (`0.613` macro-F1) and remains below 1 ms median CPU inference on the measurement host.

**Offline or delayed inference:** use **Past 7 + current + future 7 at width 0.25** when a 7 s delay is acceptable. It fits in 0.084 MB and reaches `0.694` macro-F1, statistically indistinguishable from the wider past 7 + current + future 7 model while using approximately 72% fewer FLOPs.

**Lowest-cost current-only inference:** width 0.25 uses only `0.0010` GFLOPs and `0.267 ms` median CPU time, but its `0.512` macro-F1 leaves substantial accuracy on the table.

Before selecting a production configuration:

1. Repeat the key runs with at least 3–5 seeds and report paired means, standard deviations, and confidence intervals.
2. Benchmark exported models on the actual edge hardware and runtime.
3. Sweep shorter causal histories, such as 2 and 4 total windows, to find the accuracy-latency knee below eight windows.
4. For delayed inference, sweep 1 and 3 future windows to test whether most of the future-context gain is available below 7 s delay.
5. Investigate rare transition classes using the saved confusion matrices and subject timelines.
6. Evaluate probability calibration if confidence values will be consumed downstream.

## Reproduction commands

```bash
# Compact current only
uv run imu-edge-train --config configs/benchmark.yaml \
  --model edge_window_tcn \
  --width-mult 0.25 \
  --context-len 1 \
  --future-context-len 0 \
  --run-name edge_window_tcn_wm025_current

# Compact seven past windows + current
uv run imu-edge-train --config configs/benchmark.yaml \
  --model edge_window_tcn \
  --width-mult 0.25 \
  --context-len 8 \
  --future-context-len 0 \
  --run-name edge_window_tcn_wm025_past7_current

# Compact seven past windows + current + seven future windows
uv run imu-edge-train --config configs/benchmark.yaml \
  --model edge_window_tcn \
  --width-mult 0.25 \
  --context-len 8 \
  --future-context-len 7 \
  --run-name edge_window_tcn_wm025_past7_current_future7
```

The matched width-0.5 runs use the same commands with `--width-mult 0.5` and the existing run names `edge_window_tcn_current`, `edge_window_tcn_past7_current`, and `edge_window_tcn_past7_current_future7`.

## Detailed artifacts

### Main width-0.25 runs

| Run | Metrics and predictions | Confusion matrices | Prediction timelines |
|---|---|---|---|
| Current only | [reports](edge_window_tcn_wm025_current/reports/) | [all subjects](edge_window_tcn_wm025_current/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_wm025_current/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_wm025_current/plots/index.html) |
| Past 7 + current | [reports](edge_window_tcn_wm025_past7_current/reports/) | [all subjects](edge_window_tcn_wm025_past7_current/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_wm025_past7_current/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_wm025_past7_current/plots/index.html) |
| Past 7 + current + future 7 | [reports](edge_window_tcn_wm025_centered_scratch/reports/) | [all subjects](edge_window_tcn_wm025_centered_scratch/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_wm025_centered_scratch/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_wm025_centered_scratch/plots/index.html) |

### Width-0.5 reference runs

| Run | Metrics and predictions | Confusion matrices | Prediction timelines |
|---|---|---|---|
| Current only | [reports](edge_window_tcn_current/reports/) | [all subjects](edge_window_tcn_current/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_current/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_current/plots/index.html) |
| Past 7 + current | [reports](edge_window_tcn_past7_current/reports/) | [all subjects](edge_window_tcn_past7_current/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_past7_current/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_past7_current/plots/index.html) |
| Past 7 + current + future 7 | [reports](edge_window_tcn_past7_current_future7/reports/) | [all subjects](edge_window_tcn_past7_current_future7/confusion_matrices/test_all_subjects.html) · [per-subject overview](edge_window_tcn_past7_current_future7/confusion_matrices/test_subjects_overview.html) | [timeline index](edge_window_tcn_past7_current_future7/plots/index.html) |
