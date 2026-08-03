# Deployment hardware

Target platform for the Distrimuse IMU edge models. Written so that model-side
decisions in this repository (width, context length, quantization format,
normalization constants) can be checked against a concrete device instead of a
generic "microcontroller".

Two classes of number appear below. **Datasheet** figures are quoted from the
vendor. **Estimate** figures are analytic calculations from this repository's
model definitions under a stated assumption — they are budgeting numbers, not
measurements, and nothing here has yet been run on silicon.

## Summary

| | Part | Role |
|---|---|---|
| Compute / radio | Nordic **nRF54L15** | Application core, BLE link, model inference |
| Sensor | ST **LSM6DSOX** | 3-axis accelerometer + 3-axis gyroscope (the 6 channels in `data.sensor_cols`) |

This is the tag's *next* hardware revision, not the platform the current
recordings were captured on.

## nRF54L15 (compute)

| Property | Value | Source |
|---|---|---|
| Application core | Arm Cortex-M33 @ 128 MHz, with DSP extension and single-precision FPU | datasheet |
| Coprocessor | 128 MHz RISC-V VPR ("FLPR"), for SoftPeripherals | datasheet |
| Non-volatile memory | 1.5 MB RRAM | datasheet |
| RAM | 256 KB | datasheet |
| NN accelerator | **none** | — |
| Benchmark | 503 CoreMark, 193 CoreMark/mA @ 3 V | datasheet |
| Sleep power | 0.7 µA – 2.9 µA @ 3 V | datasheet |
| Radio power | 3.4 mA RX / 4.8 mA TX @ 0 dBm, 3 V | datasheet |
| Sensor interfaces | 1× high-speed SPI/UART + 4× SPI/TWIM/UART | datasheet |
| Security | TrustZone, CRACEN crypto engine, instruction/data cache | datasheet |

### What matters for inference

- **Cortex-M33 + DSP extension means CMSIS-NN is a first-class target.** This is
  the single most important difference from the thesis platform. The thesis was
  stuck on a Cortex-R4F, which CMSIS-NN does not support; it had to disable
  vectorization (`tir.disable_vectorize=1`), fell back to plain C for fully
  connected layers, and saw TVM emit 16-bit weights instead of int8. On an M33
  the `arm_convolve_*_s8` / `arm_depthwise_conv_s8` kernels and their SMLAD-based
  dual-MAC inner loops are directly usable. int8 should therefore deliver its
  full throughput advantage here, which it did not on the IWR6843.
- **No Helium/MVE.** That is Cortex-M55/M85. Per-cycle int8 throughput is
  M4F-class (~2 MAC/cycle sustained via SMLAD); the gain over an nRF52840 is
  clock rate (128 vs 64 MHz), not width.
- **No NPU.** Compression is the only lever; there is no accelerator to fall
  back on.
- **256 KB RAM is shared** with the BLE stack, Zephyr, the sensor ring buffer,
  and application logic — not all of it is available to the model. Weights live
  in RRAM (1.5 MB, execute-in-place), so weight storage is not the binding
  constraint; activation working set is.

## LSM6DSOX (sensing)

| Property | Value | Source |
|---|---|---|
| Channels | 3-axis accel + 3-axis gyro | datasheet |
| Accel full scale | ±2 / ±4 / ±8 / ±16 g | datasheet |
| Gyro full scale | ±125 / ±250 / ±500 / ±1000 / ±2000 dps | datasheet |
| Current | 0.55 mA in combo high-performance mode | datasheet |
| FIFO | Smart FIFO, up to 9 KB, with dynamic batching | datasheet |
| Interface | SPI / I²C / MIPI I3C, plus auxiliary SPI | datasheet |
| Supply | 1.71–3.6 V analog, independent 1.62 V IO | datasheet |
| On-sensor ML | Machine Learning Core — up to 8 decision trees; 16-program finite state machine | datasheet / AN5259 |
| Package | 2.5 × 3 × 0.83 mm LGA | datasheet |

### What matters for the pipeline

- **104 Hz is a native LSM6DSOX ODR.** The models in this repository are trained
  at 104 Hz (`export_onnx(..., fs=104)`, 3 s window = 312 samples), so the sensor
  can be configured to produce exactly the training rate with no resampling on
  device. Keep it that way — a mismatch between deployed ODR and training ODR
  silently changes the temporal scale the convolutions learned.
- **Full-scale settings are part of the model contract.** Raw LSB-to-g and
  LSB-to-dps conversion depends on the configured full scale. Whatever the
  training recordings used must be recorded and reproduced in firmware,
  *together with* the per-channel normalization statistics the datamodule fits on
  the training split. Those constants must be baked into the firmware; they are
  currently only implicit in the cached windows.
- **The Smart FIFO removes most of the sampling wakeup cost.** At 104 Hz × 6
  channels × 2 bytes ≈ 1.25 KB/s, the 9 KB FIFO holds roughly 7 s of data, so the
  M33 can sleep through a full hop and wake once per second to drain it.
- **The on-sensor MLC is not a path for these models.** It runs small decision
  trees over hand-computed features, not dilated convolutions. It is worth
  keeping in mind as a wake-up gate (e.g. an on-sensor motion/no-motion tree that
  keeps the M33 asleep during long `Not Moving` stretches) rather than as an
  inference target.

## Comparison with the thesis target

The thesis (Deruytter, *Optimizing Radar-Based Machine Learning Algorithms for
Deployment on Edge Devices in Patient Monitoring*) deployed on a TI IWR6843.
That is a different device class, and the differences move in our favour on
every axis that matters for a small int8 CNN:

| | IWR6843 (thesis) | nRF54L15 (this project) |
|---|---|---|
| Inference core | Cortex-R4F @ 200 MHz | Cortex-M33 @ 128 MHz |
| CMSIS-NN support | **no** (falls back to plain C, no SIMD) | **yes** |
| RAM | 1.75 MB total, but 768 KB L3 shared with the radar processing chain; only 29.3 KB of DATA RAM free | 256 KB, shared with BLE stack + application |
| Weight storage | 512 KB PROG RAM / external 2 MB QSPI flash | 1.5 MB on-chip RRAM (XIP) |
| Sensor | on-die FMCW radar (3 TX / 4 RX) | external LSM6DSOX over SPI |
| Sensor preprocessing | 2× FFT on HWA/DSP, then Time-Doppler map | none — raw 6-channel windows |
| Active power | ~1.5–2 W | mA-class; 193 CoreMark/mA @ 3 V |
| Real-time deadline | 2.5 s radar window | 1 s hop (`data.hop_size_s`) |

Two consequences worth stating plainly:

1. **The thesis's toolchain pain was largely Cortex-R specific.** Disabled
   vectorization, int16 weight inflation, and the CMSIS-NN/USMP incompatibility
   all trace back to the R4F not being a CMSIS-NN target. Those specific problems
   should not reappear on an M33.
2. **Our real-time deadline is tighter (1 s vs 2.5 s) but our model is far
   smaller.** The thesis's recommended model was 32.5 KB / 11.8 MMACs / 333 ms.
   `edge_window_tcn` at width 0.25 is 16,569 parameters (~16 KB int8) and ~1–15
   MMACs depending on context handling — see below.

## Budget: what fits

Model: `edge_window_tcn`, `width_mult=0.25`, 3 s windows at 104 Hz, 1 s hop,
9 classes. Parameter breakdown is measured from the model definition; MAC counts
and timings are **estimates** under 2 MAC/cycle int8 and 0.5 MAC/cycle float32 at
128 MHz.

### Parameters (measured)

| Layer type | Count | Parameters | Share |
|---|---:|---:|---:|
| `Conv1d` | 9 | 15,088 | 91.1% |
| `Linear` | 2 | 1,017 | 6.1% |
| `LayerNorm` | 7 | 336 | 2.0% |
| `BatchNorm1d` | 3 | 128 | 0.8% |
| **Total** | | **16,569** | |

int8 weights ≈ 16.2 KiB, float32 ≈ 64.7 KiB. Either fits RRAM trivially.

### Compute — and the one deployment optimization that changes everything

The per-window CNN encoder costs **1,009,152 MACs per window** (conv1 209,664 +
conv2 399,360 + conv3 399,360 + projection 768). The temporal TCN over window
embeddings costs only 10,368 MACs *per position*.

So almost all the compute is the encoder — and because every window is encoded
**independently**, a streaming deployment only has to encode the one new window
per hop and keep the other embeddings in a ring buffer:

| Context mode | Windows | Re-encode all windows | Cache embeddings | Saving |
|---|---:|---:|---:|---:|
| Current only | 1 | 1.02 MMAC | 1.02 MMAC | — |
| Past 7 + current | 8 | 8.16 MMAC | **1.09 MMAC** | 7.5× |
| Past 7 + current + future 7 | 15 | 15.29 MMAC | **1.17 MMAC** | 13.1× |

Estimated inference time on the M33 @ 128 MHz:

| Context mode | int8, re-encode | int8, cached | float32, cached | Duty cycle (int8, cached, 1 s hop) |
|---|---:|---:|---:|---:|
| Current only | 3.98 ms | 3.98 ms | 15.9 ms | 0.40% |
| Past 7 + current | 31.9 ms | **4.27 ms** | 17.1 ms | 0.43% |
| Past 7 + current + future 7 | 59.7 ms | **4.55 ms** | 18.2 ms | 0.46% |

Read that last column again: **with embedding caching, all three context modes
cost essentially the same on device.** The 8× and 15× FLOP figures in
`experiments/results/edge_window_tcn_context_comparison.md` are host-side
whole-sequence forward passes; they are the right number for a training-time
comparison and the wrong number for a streaming firmware budget. The real cost of
future context is the **7 s of decision latency**, not compute.

Even the pessimistic re-encode-everything case (59.7 ms) is comfortably inside
the 1 s hop and well under the thesis's 333 ms.

### RAM working set

`compute_model_stats` now reports `peak_activation_kib_fp32` /
`peak_activation_kib_int8_est` for every run — the largest per-layer
input+output activation size, i.e. the ping-pong buffer definition Table 4.1 of
the reference thesis uses. Measured for `edge_window_tcn` at width 0.25:

| Context mode | Peak activation, fp32 (measured) | Peak activation, int8 (naive ÷4) |
|---|---:|---:|
| Current only | 39.0 KiB | 9.75 KiB |
| Past 7 + current | 312.0 KiB | 78.0 KiB |
| Past 7 + current + future 7 | 585.0 KiB | 146.25 KiB |

These numbers scale linearly with the number of windows (`312 = 39 × 8`,
`585 = 39 × 15`) because they trace the same **naive re-encode-all-windows**
input shape used for the `macs`/`gflops` fields — the encode-windows reshape
puts all `N` windows through the shared CNN encoder as one batch, so the
biggest layer's ping-pong buffer holds `N` windows' worth of activations at
once. This is the right number for the current benchmark pipeline's
comparisons, and the wrong number for a streaming firmware budget, for the same
reason the naive MACs figures above are the wrong number for firmware: a
device that encodes one new window per hop and reuses cached embeddings for the
rest never holds more than *one* window's worth of encoder activations at a
time, regardless of context length. That hand-derived streaming estimate:

| Item | Size |
|---|---:|
| Encoder activation ping-pong peak, one window (16×312 in + 16×312 out) | 9.8 KiB |
| Raw window ring buffer (312 × 6, int16) | 3.7 KiB |
| Embedding ring buffer (15 × 24, int8) | 360 B |
| Temporal TCN activation peak (24×15 × 2) | 720 B |
| **Total** | **≈ 15 KiB** |

Both readings agree at context length 1, where there is only one window to
begin with (9.75 KiB measured int8 vs. 9.8 KiB hand-derived encoder peak) — the
gap only opens once `N > 1`, and it opens for the batching reason stated above,
not because the two calculations disagree about the model.

Against 256 KB of total RAM, these two readings tell different stories, which
is exactly why the distinction matters:

- **Streaming, either precision (≈ 15 KiB fp32 / ≈ 4 KiB int8):** nowhere near
  a constraint, with room to spare for the BLE stack and everything else Zephyr
  needs resident.
- **Naive full-batch, int8 (up to 146.25 KiB at the widest context):** still
  fits, but now consumes over half the chip's RAM for activations alone before
  the radio stack gets anything.
- **Naive full-batch, float32 (up to 585 KiB at the widest context):**
  **does not fit at all** — more than double the chip's total RAM. The
  streaming design is not an optimization here; for the past 7 + current +
  future 7 configuration in float32, it is the difference between fitting on
  this part and not.

Model weights are excluded from this budget either way: they live in RRAM
(execute-in-place, same role as the thesis's QSPI XIP flash for the IWR6843),
not in the 256 KB RAM this table is about. **Memory is a real constraint for
the naive deployment path, and not a constraint at all for the streaming
one** — the opposite framing from the thesis, where 29.3 KB of free DATA RAM
forced workspace buffers into higher-latency L3 RAM regardless of how the
model was structured. That is the strongest concrete argument in this document
for implementing the streaming design rather than treating it as a future
optimization.

## Toolchain

Neither the artifact this repository currently produces nor its runtime targets
this hardware. `imu-edge-quantize` emits a QOperator int8 ONNX graph and
evaluates it with **ONNX Runtime on the host** — the right tool for answering
"what does int8 cost in accuracy", and not a deployable artifact for bare-metal
Cortex-M33. The gap is stated in
[`compression/onnx_int8.py`](src/distrimuse_imu_edge/compression/onnx_int8.py)
and in the context report; it is a known open item, not an oversight.

Candidate paths, in rough order of preference for this target:

1. **TFLite Micro under Zephyr / nRF Connect SDK, with CMSIS-NN kernels.**
   Nordic's officially supported ML runtime. Needs an int8 `.tflite`, so the
   conversion route matters: `ai-edge-torch` converts PyTorch directly to TFLite
   with int8 PTQ and avoids the ONNX→TF hop entirely; `onnx2tf` works from the
   existing float32 ONNX. Note that a **QDQ** export converts to TFLite far more
   cleanly than QOperator, so a deployment build may want the QDQ artifact even
   though QOperator is the right choice for file size — `--quant-format QDQ` is
   already exposed on `imu-edge-quantize`.
2. **microTVM, mirroring the thesis.** The existing QOperator ONNX is a direct
   input: `tvmc compile --target="cmsis-nn, c" --target-c-device=cortex-m33
   --executor=aot --runtime=crt --executor-aot-unpacked-api=1 --interface-api=c`.
   Because the M33 *is* a CMSIS-NN target, the two workarounds the thesis needed
   (disabled vectorization, int16 weight fallback) should not be required. The
   USMP-vs-CMSIS-NN incompatibility it hit may still apply.
3. **ExecuTorch with the ARM/Cortex-M backend.** Same PyTorch source, no ONNX
   hop, but the least mature of the three for this part.
4. **Edge Impulse.** Lowest friction, Nordic-partnered, at the cost of owning
   less of the pipeline.

Whichever is chosen, the deployed artifact must be re-evaluated against the same
test split — a converted graph is not the graph that was measured.

## Known model-side blockers for this target

Concrete, actionable, and all in this repository rather than in the hardware:

1. **`LayerNorm` does not fold and has no int8 MCU kernel.** The temporal TCN uses
   six `_ChannelLayerNorm` layers plus one in the head. `BatchNorm1d` folds into
   the preceding convolution's weights at export and costs nothing at inference;
   `LayerNorm` requires a runtime mean/variance pass, and neither CMSIS-NN nor
   TFLite Micro has an int8 kernel for it, so it would fall back to float and
   break an otherwise all-int8 graph. The encoder already uses `BatchNorm1d`
   correctly. Switching `_ChannelLayerNorm` to `BatchNorm1d` is safe for
   causality — at inference BatchNorm uses running statistics, so it does not mix
   information across sequence positions — and needs a retrain to confirm no
   accuracy cost.
2. **`apply_structured_pruning` cannot reduce size or MACs.** It calls
   `prune.ln_structured` then `prune.remove`, which bakes a zero mask in without
   changing tensor shapes: parameters, state-dict size, and MACs come out
   bit-identical. The thesis used `torch-pruning`
   (`GroupMagnitudeImportance(p=2)` + `MagnitudePruner`), which physically removes
   channels and rewires the following layer — and got a real 3× speedup and 2×
   workspace reduction at 50% sparsity. Adopting `torch-pruning` is the fix.
3. ~~The energy profile does not describe this device.~~ **Done.**
   `nrf54l15_m33_128mhz` is now the default profile in
   [`evaluation/energy.py`](src/distrimuse_imu_edge/evaluation/energy.py) and in
   `configs/benchmark.yaml`, replacing the `nrf52840_m4f_64mhz` entry it grew out
   of. Values and their derivations:

   | Field | Value | Derivation |
   |---|---:|---|
   | `f_clock_hz` | 128e6 | datasheet |
   | `macs_per_cycle_int8` | 2.0 | CMSIS-NN SMLAD dual-MAC; M33 DSP extension, no Helium |
   | `macs_per_cycle_float32` | 0.5 | scalar single-precision FPU |
   | `p_active_mw` | 7.8 | 503 CoreMark ÷ 193 CoreMark/mA = 2.61 mA, × 3.0 V |
   | `p_sleep_mw` | 0.009 | 2.9 µA (System ON worst case) × 3.0 V |
   | `battery_capacity_mah` / `battery_voltage_v` | 225 / 3.0 | CR2032, unchanged across profiles for comparability |

   Runs recorded before this change carry `energy_profile:
   nrf52840_m4f_64mhz` and are not comparable; group by `energy_profile` before
   comparing. They still load, because `config.resolved.yaml` stores the whole
   profile rather than just its name.

   The `p_active_mw` figure is the weakest link: CoreMark/mA is a CPU-efficiency
   benchmark, and a convolution inner loop hammering RRAM and SRAM will not draw
   the same current as CoreMark. Treat 7.8 mW as a floor and override it after
   measuring.
4. ~~Peak activation memory was not tracked at all.~~ **Done.**
   `compute_model_stats` now reports `peak_activation_kib_fp32` and a naive
   `peak_activation_kib_int8_est`, following the thesis's own "max input+output
   over all layers" definition. See the RAM working set table above — this is
   also what turned up the 585 KiB float32 figure that exceeds this chip's
   256 KB of RAM for the naive full-batch path, which is now the sharpest
   concrete reason to implement item 6 below rather than defer it.
5. **Depthwise separable convolutions are not used, and would help less here than
   in the thesis.** The thesis got ~8–9× fewer operations per layer because it was
   replacing 2D 3×3 convolutions. In 1D the reduction is capped near the kernel
   size: 3.81× for conv2 (16→16, k5), 4.32× for conv3 (16→32, k5). Applying it to
   conv2 and conv3 while leaving conv1 standard cuts the encoder from 1,009,152 to
   407,616 MACs — a real **2.48×**, but not the thesis's 8–9×, and depthwise
   kernels are memory-bound so wall-clock gain will be smaller than the MAC gain.
   Given that the cached-embedding budget is already ~4.3 ms against a 1 s hop,
   this is an energy optimization, not a feasibility one.
6. **Streaming inference is not implemented anywhere.** The embedding cache
   described above is a firmware-side design, and the accuracy equivalence
   between "re-encode all windows" and "reuse cached embeddings" should be
   verified numerically before it is relied upon. It should hold exactly, because
   the encoder is deterministic and position-independent, but int8 requantization
   of stored embeddings is a plausible source of drift.

## What still has to be measured

Everything in the estimate columns. In priority order:

1. Latency and cycle count of the exported int8 graph on an nRF54L15 DK, per
   context mode, with and without embedding caching.
2. Actual RAM high-water mark from the Zephyr build, against the ~15 KiB estimate.
3. Board power with a Nordic PPK2 or Otii Arc, split into sensor, inference, and
   BLE terms. The analytic energy model in this repository covers inference only
   and explicitly excludes sampling and radio, which on a wearable are often the
   larger terms.
4. Test macro-F1 of the *converted* artifact (TFLite or TVM), not just the ONNX
   one, on the same held-out subjects.

## Sources

- [nRF54L15 product page](https://www.nordicsemi.com/Products/nRF54L15) —
  Nordic Semiconductor
- [nRF54L15 / nRF54L10 / nRF54L05 datasheet](https://docs.nordicsemi.com/bundle/ps_nrf54l15/page/keyfeatures.html) —
  Nordic Semiconductor
- [LSM6DSOX product page](https://www.st.com/en/mems-and-sensors/lsm6dsox.html) and
  [datasheet](https://www.st.com/resource/en/datasheet/lsm6dsox.pdf) —
  STMicroelectronics
- [AN5259: LSM6DSOX machine learning core](https://www.st.com/resource/en/application_note/an5259-lsm6dsox-machine-learning-core-stmicroelectronics.pdf) —
  STMicroelectronics
- E. Deruytter, *Optimizing Radar-Based Machine Learning Algorithms for
  Deployment on Edge Devices in Patient Monitoring*, KU Leuven Bruges, 2025–2026
  — `../docs/thesis_Elias_deruytter.pdf` (under embargo until 30/09/2028)
