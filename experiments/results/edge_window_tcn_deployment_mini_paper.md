# Edge Window TCN: temporal context, static int8, and embedding-cached inference

## Abstract

This note summarises the complete compact `edge_window_tcn` context study: a
single current window, seven past windows plus the current window, and seven
past windows plus the current and seven future windows. Each configuration is
evaluated in float32 and with static ONNX int8 post-training quantisation.

The main result is that the **15-window, int8, embedding-cached** design is the
strongest deployment candidate for an application that can tolerate a 7 s
look-ahead: it retains a test macro-F1 of **0.6930** (float32: 0.6935) while
reducing the actual ONNX Runtime-profiled activation-plus-cache working set
from **146.40 KiB** for normal int8 execution to **11.31 KiB** with cached
embeddings. The cache path is implemented in `distrimuse-ds-shared` and takes
**0.129 ms** median host ONNX Runtime time per hop (20 warm-ups, 100 timed
calls). The corresponding float32 cache measures 45.47 KiB and 0.152 ms.

![Overview of the 15-window Edge Window TCN architecture](edge_window_tcn_context_report_assets/edge-window-tcn-architecture-overview-intro.png)

**Figure 1 — Edge Window TCN overview.** Fifteen overlapping 3 s IMU windows
(seven past, the current target window, and seven future) are sampled every
second, giving a 17 s unique temporal span. One shared CNN encodes every raw
window independently into a 24-D embedding, so adding context reuses the same
weights instead of concatenating a longer raw signal. The non-causal dilated
TCN then mixes these embeddings across time and the classifier explicitly
selects the middle/current token (`t`, index 7) for the 9-class prediction.
The seven future windows are look-ahead: they improve the prediction but mean
the decision for `t` is available 7 s later.

## Model and temporal contexts

`edge_window_tcn` separates local sensor feature extraction from temporal
reasoning. Every 3 s IMU window (312 samples at 104 Hz; 6 channels) passes
through the **same compact CNN**. At width 0.25, that encoder produces one
24-dimensional embedding per window. A three-block residual TCN (kernel 3,
dilations 1, 2, 4) then processes the sequence of embeddings. The classifier
always reads the explicit current token, rather than the newest token.

| Context | Input windows for one prediction | Temporal span | Decision delay | Temporal TCN mode |
|---|---:|---:|---:|---|
| Current only | 1 | 3 s | 0 s | Causal |
| Past 7 + current | 8 | 10 s | 0 s | Causal |
| Past 7 + current + future 7 | 15 | 17 s | 7 s | Non-causal / symmetric padding |

Adjacent windows overlap because the hop is 1 s. In the final row, a prediction
for window `t` is emitted only after windows through `t+7` have arrived. The
look-ahead is a property of the model’s future context, not a side effect of
caching.

All three width-0.25 models have 16,569 learned parameters and a 0.084 MB
float32 state dict: context changes the input sequence and execution cost, not
the learned weight count. Exported ONNX artifact sizes differ slightly by
context because the exported graph contains input-shape-dependent bookkeeping;
they should not be interpreted as different learned model capacities.

## Deployment trade-offs

![Test macro-F1 versus latency, activation memory, and ONNX artifact size](edge_window_tcn_context_report_assets/edge-window-tcn-deployment-tradeoffs.svg)

Figure description: colour denotes temporal context; circles are normal
float32, squares are normal static-int8, down-triangles are cached float32,
and up-triangles are cached static-int8. Every point is measured from the
published shared ONNX artifacts. Cached points use only real, non-padded
contexts; normal points include the configuration's zero-padded boundaries.
Panel B includes the resident float32 embedding ring buffer as well as actual
ONNX Runtime-profiled node input/output activations. Latencies use 20 warm-up
calls and 100 timed calls on one held-out test input. Cached deployment stores
a split encoder and temporal graph pair; the table reports the exact combined
and split artifact sizes.

| Context | Precision / scheduling | Test macro-F1 | Median host ONNX Runtime latency | Peak activation + cache | ONNX artifact size |
|---|---|---:|---:|---:|---:|
| Current only | float32, normal | 0.5123 (full) | 0.058 ms | 44.06 KiB | 63.14 KiB |
| Current only | float32, cached | 0.5123 (valid stream) | 0.056 ms | 44.16 KiB | 51.66 KiB |
| Current only | int8, normal | 0.4756 (full) | 0.087 ms | 9.90 KiB | 60.62 KiB |
| Current only | int8, cached | 0.4756 (valid stream) | 0.047 ms | 9.99 KiB | 44.46 KiB |
| Past 7 + current | float32, normal | 0.5517 (full) | 0.158 ms | 317.06 KiB | 95.43 KiB |
| Past 7 + current | float32, cached | 0.5515 (valid stream) | 0.103 ms | 44.81 KiB | 92.60 KiB |
| Past 7 + current | int8, normal | 0.5282 (full) | 0.172 ms | 78.15 KiB | 74.85 KiB |
| Past 7 + current | int8, cached | 0.5279 (valid stream) | 0.134 ms | 10.65 KiB | 70.75 KiB |
| Past 7 + current + future 7 | float32, normal | **0.6935** (full) | 0.272 ms | 590.06 KiB | 77.63 KiB |
| Past 7 + current + future 7 | float32, cached | 0.6925 (valid stream) | 0.152 ms | 45.47 KiB | 74.80 KiB |
| Past 7 + current + future 7 | int8, normal | **0.6930** (full) | 0.253 ms | 146.40 KiB | 50.19 KiB |
| Past 7 + current + future 7 | int8, cached | **0.6921** (valid stream) | **0.129 ms** | **11.31 KiB** | **45.47 KiB** |

The strongest accuracy result is the 15-window model. Adding seven past
windows raised macro-F1 from 0.5123 to 0.5517; adding seven future windows
then raised it to 0.6935. Crucially, static int8 changes that final result by
only −0.0006 macro-F1, whereas its effect is appreciably larger for the causal
models (−0.0366 for current-only and −0.0235 for past-plus-current).

For the normal, non-cached schedule, more context raises peak activation memory
linearly because all raw windows are sent through the CNN together. The
15-window float32 run therefore measures 590.06 KiB at peak, which is larger
than a 256 KiB target RAM budget. Normal int8 measures 146.40 KiB, still more
than half of that budget.

## Why caching embeddings changes the memory story

![Embedding-cache execution schedule and memory reduction](edge_window_tcn_context_report_assets/edge-window-tcn-embedding-cache-explainer.svg)

Figure description: the top timeline is the complete context used to classify
the middle/current window `t`. In naive inference, every 1 s hop re-runs the
CNN on all 15 windows. In streaming inference, the encoder processes only the
newly arrived raw window. Its 24-D output is appended to a fixed-size buffer,
the oldest embedding is discarded, and the unchanged TCN runs over that
15-embedding buffer to classify the middle token.

This works because the CNN encoder is shared across window positions and
contains no cross-window operation: a window’s embedding depends only on that
window. Consecutive contexts overlap by 14 of 15 windows, so 14 encoder passes
per hop are redundant in the normal schedule. The cache changes scheduling,
not the mathematical model for float32: the existing streaming-equivalence
tests verify identical logits for the equivalent window range (up to
floating-point reassociation). The separately calibrated int8 split graphs are
validated class-for-class on the held-out valid stream.

| 15-window future-context configuration | Normal schedule | Embedding-cached schedule | Change |
|---|---:|---:|---:|
| CNN encodes per 1 s hop | 15 windows | 1 arriving window | 15× less repeated CNN work |
| Float32 peak activation + cache | 590.06 KiB | **45.47 KiB** | **13.0× lower** |
| Float32 median host ONNX Runtime latency | 0.272 ms | **0.152 ms** | 1.79× faster |
| Float32 macro-F1 | 0.6935 (full) | 0.6925 (valid stream) | boundary windows excluded |
| Int8 peak activation + cache | 146.40 KiB | **11.31 KiB** | **12.9× lower** |
| Int8 macro-F1 | 0.6930 (full) | 0.6921 (valid stream) | boundary windows excluded |
| Int8 cached latency | 0.253 ms | **0.129 ms** | 1.96× faster |

\*The cache intentionally makes no prediction where a real device lacks needed
history or look-ahead. Thus, the cached F1 uses 10,292 real contexts rather
than 10,628 zero-padded test targets. Float32 cached logits are equivalent to
the combined graph; the int8 cached and combined graphs agree on the predicted
class for all 10,292 valid final-model contexts.

Caching does **not** remove the 7 s look-ahead: window `t` still cannot be
classified until `t+7` has arrived. It also does not cache the TCN itself; the
TCN runs over the full embedding buffer at every hop. It removes the dominant,
repeated CNN work and keeps the CNN activation peak constant at one window.

## Executable IMU streaming reference in `distrimuse-ds-shared`

This caching approach is not only a training-repository prototype. The shared
repository ships an ONNX-only IMU inference implementation under
[`models/imu/`](https://github.com/Sentigrate/distrimuse-ds-shared/tree/main/models/imu).
It exposes the normal path as `OnnxModel` and the cached path as
[`StreamingOnnxModel`](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/pipeline.py#L382-L498).
The latter mirrors the PyTorch `StreamingWindowPredictor` used to make the
measurements in this report, but it uses ONNX Runtime and therefore does not
need PyTorch in the deployment package.

### What the two code paths do differently

| Step | Normal stream (`OnnxModel`) | Cached stream (`StreamingOnnxModel`) |
|---|---|---|
| Published artifact | One combined ONNX graph | Two ONNX graphs: shared CNN encoder + temporal TCN/head |
| Per-hop input | Rebuild a raw `(1, N, 312, 6)` context (`N=15` for the final model) | Accept one newly arrived `(312, 6)` raw window |
| CNN work per hop | Combined graph encodes all `N` windows again | Encoder graph runs once, then its 24-D output is appended to a `deque(maxlen=N)` |
| Temporal work per hop | Happens inside the combined graph | Stack the cached embeddings as `(1, 24, N)` and run the unchanged temporal graph |
| State | No learned/cached runtime state | One ring buffer per sensor/person stream; reset at every person/scenario boundary |
| Output timing | Uses zero-padding at recording boundaries and a trailing flush | Emits only after `N` real windows fill the cache; no fictional zero-history or end-of-recording future windows |
| Prediction / F1 | Reference output | Float32: same logits for every mutually valid context. Int8: separately calibrated split graphs; validated class-for-class on the held-out valid stream. |

The normal live-stream loop reconstructs the complete context and invokes the
combined model on every hop:

```python
ctx = build_context_for_target(windows, target_idx, context_len, current_index=7)
probs = model.probabilities(ctx[None])
```

The cached implementation performs the schedule change directly in
`StreamingOnnxModel.push()`:

```python
embedding = encoder_onnx(new_window.T[None])   # one new 24-D vector
buffer.append(embedding)                        # evicts the oldest at capacity 15
if buffer_is_full:
    logits = temporal_onnx(stack(buffer).T[None])  # (1, 24, 15)
```

The real code is available in the shared repository’s
[`OnnxModel.logits`](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/pipeline.py#L362-L379),
[`StreamingOnnxModel.push`](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/pipeline.py#L461-L491),
and stream drivers for the
[normal](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/inference.py#L685-L747)
and [cached](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/inference.py#L750-L835)
paths. The cache driver explicitly resets the buffer when a person/scenario
group changes, preventing context leakage across recordings.

### Running the implementation

From the root of `distrimuse-ds-shared`, the normal and cached live-stream
simulations use the same prepared IMU input. The float32 pair demonstrates
bit-level-equivalent scheduling (within floating-point reassociation):

```bash
# Normal: rebuild and re-encode the 15-window context on each hop.
uv run python models/imu/inference.py \
  --input-npz models/imu/data/sample_windows.npz \
  --model past7_future7_fp32 --stream

# Cached: encode only the newly arrived window and reuse the 14 overlaps.
uv run python models/imu/inference.py \
  --input-npz models/imu/data/sample_windows.npz \
  --model past7_future7_fp32 --stream --streaming-cache
```

The same public path is available for the final compressed model:

```bash
uv run python models/imu/inference.py \
  --input-npz models/imu/data/sample_windows.npz \
  --model past7_future7_int8 --stream --streaming-cache
```

The entrypoint selects the implementation through
[`--streaming-cache`](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/inference.py#L183-L195)
and fails clearly unless `--stream` is present and the selected model has its
two published streaming graphs. The mapping from model variant to encoder and
temporal graphs is declared in
[`models/config/imu_shared_config.yaml`](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/config/imu_shared_config.yaml#L88-L150),
not hard-coded in application logic.

### Correctness and int8 validation

The shared repository has an
[exact-equivalence test](https://github.com/Sentigrate/distrimuse-ds-shared/blob/main/models/imu/tests/test_prepared_inference.py#L139-L191): it feeds the same stream to the combined ONNX model and the split cached ONNX model for all three published float32 contexts, then checks the logits with `allclose(atol=1e-4)` after the cache has warmed up. This validates the claim used throughout this mini-paper: caching changes the execution schedule, not the model’s prediction.

The shared repository now publishes split encoder/temporal ONNX pairs for
**all six** model variants, including `current_int8`, `past7_int8`, and the
final `past7_future7_int8`. The three int8 pairs are calibrated separately on
the training split and have a dedicated streaming smoke test. They therefore
must not be described as bit-identical logits to the combined int8 graph. An
end-to-end replay of the held-out stream provides the practical check: for the
final 15-window model, normal and cached int8 agreed on the predicted class for
**10,292 / 10,292** valid (non-boundary-padded) test predictions, and both
obtained **0.6921 macro-F1** on that same subset. The official zero-padded
full-test value remains **0.6930**. The difference in sample count/value is
expected because real streaming does not invent the first 7 history windows or
the final 7 look-ahead windows of each recording.

## Recommendation

There are two sensible operating points.

1. **Causal / no decision delay:** use **past 7 + current, float32 cached**.
   It reaches 0.5515 macro-F1 on valid streamed test contexts with no
   look-ahead, 44.81 KiB measured activation-plus-cache peak, and 0.134 ms
   median host ONNX Runtime latency.

2. **Best accuracy per peak-memory budget:** use **past 7 + current + future
   7, int8 with embedding caching**. It reaches 0.6921 macro-F1 on 10,292
   valid streamed test contexts with the lowest measured working set among the
   highest-accuracy choices (11.31 KiB) and 0.129 ms median host latency. This
   is the preferred final design when a 7 s decision delay is acceptable. It
   runs as `past7_future7_int8 --stream --streaming-cache` in
   `distrimuse-ds-shared`. Host measurements still are not a claim about a
   particular microcontroller; repeat the same benchmark on that device.

## Measurement notes and reproducibility

- All F1 values are recomputed from `cache/windows/test_e08d1a9b655d553c.npz`,
  whose people exactly match `configs/split.yaml`'s configured test set:
  8, 15, 24, 26, and 27. Normal inference reports every zero-padded target;
  cached inference reports only genuine contexts available to a live stream.
- Latencies are median host ONNX Runtime times for 100 calls after 20 warm-ups,
  using a held-out test input. They compare scheduling choices on this host,
  not a microcontroller guarantee.
- Peak memory is measured from ONNX Runtime profiling of the concrete node
  input/output type-shapes. It is the largest node-local activation footprint,
  plus the resident 24-D float32 embedding ring buffer for cached inference.
  Model weights, quantization scales, and process-wide allocator reservation
  are intentionally excluded, so this remains comparable to the paper's
  activation-memory metric rather than a noisy RSS measurement.
- The source of truth is
  `edge_window_tcn_context_report_assets/shared_onnx_streaming_benchmark.json`.
  Recompute it and regenerate the figures with:

  ```bash
  uv run python scripts/benchmark_shared_onnx_streaming.py
  uv run python scripts/render_edge_window_tcn_deployment_summary.py
  ```
