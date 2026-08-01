# edge_window_tcn: real vs. real+synthetic, across temporal context

Six runs of `edge_window_tcn`, three temporal-context configurations each trained on
real-only data (`configs/benchmark.yaml`) and on real + synthetic-augmented data
(`configs/synthetic_augmented.yaml`, `distrimuse-synthetic-data`'s v35 export with
diffusion-generated SitDown/LayDown/SitUp/Falling bouts, splice-generated everything else).
30 max epochs, early-stop patience 8, seed 42, single run each (no multi-seed averaging).

- **current**: `context_len=1, future_context_len=0` — no temporal context at all
- **past7_current**: `context_len=8, future_context_len=0` — 7 past windows + current
- **past7_current_future7**: `context_len=8, future_context_len=7` — 7 past + current + 7 future

## Headline: test macro-F1

| context | real-only | real+synthetic | Δ |
|---|---|---|---|
| current | 0.5298 | 0.5493 | +1.95pp |
| past7_current | 0.5511 | 0.6126 | **+6.15pp** |
| past7_current_future7 | 0.6655 | 0.6953 | +2.98pp |

Two things stand out:

1. **Synthetic augmentation helps in all three configurations**, but the size of the gain
   isn't monotonic with context — it's biggest for `past7_current`, not for the
   context-richest `past7_current_future7`. The richer-context real-only baseline is
   already strong (0.6655) and has less room to improve.
2. **More temporal context beats more data.** Going from `current` to
   `past7_current_future7` on real data alone (0.5298 → 0.6655, +13.6pp) is a bigger jump
   than synthetic augmentation gives at any single context level. Context and synthetic
   data are complementary, not substitutes — the best result here is the combination
   (`past7_current_future7` + synthetic, 0.6953).

Model size and params are identical across all six runs (63,081 params, 0.271MB) — context
length doesn't change model capacity for this architecture, only inference latency
(0.23ms for `current` vs. ~0.97–1.0ms once past/future context is added).

## Per-class test F1

### current (context_len=1, future_context_len=0)

| class | real-only | real+synthetic | Δ |
|---|---|---|---|
| Not Moving | 0.8193 | 0.8473 | +2.80pp |
| Walk | 0.8317 | 0.8619 | +3.02pp |
| Sit Down | 0.5039 | 0.5946 | +9.07pp |
| Lay Down | 0.2726 | 0.2122 | **−6.04pp** |
| Turn | 0.5994 | 0.6175 | +1.80pp |
| Sit Up | 0.2936 | 0.2816 | −1.20pp |
| Stand Up | 0.4711 | 0.4991 | +2.80pp |
| Falling | 0.2440 | 0.2841 | +4.00pp |
| Hand | 0.7321 | 0.7453 | +1.32pp |

### past7_current (context_len=8, future_context_len=0)

| class | real-only | real+synthetic | Δ |
|---|---|---|---|
| Not Moving | 0.7982 | 0.8209 | +2.27pp |
| Walk | 0.8262 | 0.8585 | +3.23pp |
| Sit Down | 0.6276 | 0.6417 | +1.41pp |
| Lay Down | 0.2794 | 0.3113 | +3.18pp |
| Turn | 0.6101 | 0.7056 | +9.54pp |
| Sit Up | 0.2924 | 0.3831 | +9.07pp |
| Stand Up | 0.5199 | 0.6535 | +13.36pp |
| Falling | 0.2622 | 0.3472 | +8.50pp |
| Hand | 0.7434 | 0.7915 | +4.81pp |

### past7_current_future7 (context_len=8, future_context_len=7)

| class | real-only | real+synthetic | Δ |
|---|---|---|---|
| Not Moving | 0.8360 | 0.8637 | +2.77pp |
| Walk | 0.8485 | 0.8907 | +4.22pp |
| Sit Down | 0.7500 | 0.7581 | +0.81pp |
| Lay Down | 0.4299 | 0.4416 | +1.17pp |
| Turn | 0.7097 | 0.7584 | +4.86pp |
| Sit Up | 0.5725 | 0.5172 | **−5.53pp** |
| Stand Up | 0.7213 | 0.7114 | −0.99pp |
| Falling | 0.3522 | 0.4839 | **+13.16pp** |
| Hand | 0.7695 | 0.8331 | +6.36pp |

## Falling: the class the diffusion model specifically targets

Falling F1 improves with synthetic augmentation in **every** context configuration
(+4.0pp, +8.5pp, +13.2pp) — and the improvement grows with context, peaking in the
best overall model. This is a meaningfully different result from an earlier splice-only
(no diffusion) augmentation run, where Falling F1 actually got *worse* (−3.7pp) — splicing
alone just resamples the same ~103 real Falling bouts, while the diffusion model generates
genuinely new bout signals for this and the other rare classes (SitDown, LayDown, SitUp).

## Caveats

- **Not a uniform win at the class level.** LayDown regresses under `current`, and SitUp /
  StandUp both regress under `past7_current_future7` despite the model improving overall.
  Synthetic augmentation shifts the decision boundary; it doesn't strictly dominate on
  every class in every context configuration.
- **Single seed, no averaging.** Each cell above is one training run (seed 42). The
  macro-F1 deltas are large enough to be credible directionally, but per-class deltas in
  the 1–3pp range for already-small classes should be treated as noisy until re-run across
  seeds.
- **Early stopping.** Most runs stopped before 30 epochs on the validation macro-F1
  plateau (patience 8) — absolute numbers may still move with more epochs/tuning, though
  the real-vs-synthetic comparison within each context level used identical training
  budgets.

## int8 quantization of the real+synthetic models

Statically quantized each of the three real+synthetic checkpoints to int8 ONNX
(`imu-edge-quantize`, per-channel `QOperator`, calibrated on that run's own train split —
so the `past7_current`/`past7_current_future7` calibration sets include the synthetic
subjects too). Model size is the exported ONNX file, not just the raw parameter count.

| context | fp32 test F1 | int8 test F1 | Δ | fp32 size | int8 size | shrink | pred. agreement |
|---|---|---|---|---|---|---|---|
| current | 0.5493 | 0.5063 | **−4.30pp** | 163.4 KiB | 88.5 KiB | 1.85× | 0.870 |
| past7_current | 0.6126 | 0.6147 | **+0.21pp** | 276.7 KiB | 123.0 KiB | 2.25× | 0.959 |
| past7_current_future7 | 0.6953 | 0.6792 | −1.62pp | 258.9 KiB | 98.3 KiB | 2.63× | 0.934 |

Quantization is essentially free for both context-aware models (`past7_current` even ticks
up slightly — within noise) but costs the context-less `current` model meaningfully
(−4.3pp), concentrated in classes it was already weakest on (Falling −9.5pp, Stand Up
−7.2pp, Turn −5.7pp). With no temporal context to lean on, its decision boundaries are
already tight; int8 rounding pushes more predictions across them.

The practical upshot: `past7_current_future7` quantized to int8 (0.679 test F1, 98.3 KiB)
still beats `current` at full float32 (0.549, 163.4 KiB) by 13pp while being smaller —
context dominates both data and numerical precision here.

### Per-class int8 deltas (test)

**current**: Not Moving −1.2pp, Walk −4.9pp, Sit Down −5.1pp, Lay Down −2.5pp, Turn −5.7pp,
Sit Up −0.2pp, Stand Up −7.2pp, **Falling −9.5pp**, Hand −2.5pp — every class regresses.

**past7_current**: mostly flat to slightly positive (Not Moving +0.9pp, Turn +0.7pp,
Falling +1.4pp); only Sit Up (−1.1pp) and Stand Up (−0.2pp) dip, both within noise.

**past7_current_future7**: Lay Down −3.6pp and Turn −3.2pp take the brunt; Hand −4.2pp;
Stand Up and Falling actually improve slightly (+0.3pp, +0.6pp).

## Reproduction

```bash
cd distrimuse-synthetic-data
uv run python -m distrimuse_synthetic reshape-v35-cache
uv run python -m distrimuse_synthetic build-index
uv run python -m distrimuse_synthetic train-ddpm --out-dir cache/diffusion_checkpoints
uv run python -m distrimuse_synthetic generate --num-subjects 20 --duration-s 600 \
    --rare-class-source diffusion --diffusion-checkpoint cache/diffusion_checkpoints/all.ckpt
uv run python -m distrimuse_synthetic export-imu-edge-split

cd ../distrimuse-imu-edge
for ctx in "1 0 current" "8 0 past7_current" "8 7 past7_current_future7"; do
    read -r context_len future_len tag <<< "$ctx"
    uv run imu-edge-train --config configs/benchmark.yaml --model edge_window_tcn \
        --context-len "$context_len" --future-context-len "$future_len" \
        --run-name "edge_window_tcn_${tag}_scratch_v35_hand"
    uv run imu-edge-train --config configs/synthetic_augmented.yaml --model edge_window_tcn \
        --context-len "$context_len" --future-context-len "$future_len" \
        --run-name "edge_window_tcn_${tag}_diffusion_augmented_v35_hand"
done
uv run imu-edge-benchmark --results-dir experiments/results

# int8 quantization of each real+synthetic checkpoint (run-name defaults to
# "<checkpoint's run dir>_int8"; --config must match what the checkpoint was trained with)
uv run imu-edge-quantize --config configs/synthetic_augmented.yaml \
    --checkpoint experiments/results/edge_window_tcn_current_diffusion_augmented_v35_hand/checkpoints/best.ckpt
uv run imu-edge-quantize --config configs/synthetic_augmented.yaml \
    --checkpoint experiments/results/edge_window_tcn_diffusion_augmented_v35_hand/checkpoints/best.ckpt
uv run imu-edge-quantize --config configs/synthetic_augmented.yaml \
    --checkpoint experiments/results/edge_window_tcn_past7_current_future7_diffusion_augmented_v35_hand/checkpoints/best.ckpt
```
