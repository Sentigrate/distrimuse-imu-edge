"""
Manual demo: streaming inference on a real trained checkpoint, simulating
a continuous session of incoming windows, plus the real peak-memory gain.

Not part of the test suite — this is for eyeballing behavior and getting
real numbers, not an automated check (see tests/test_streaming_equivalence.py
for that).
"""

import torch
from torchinfo import summary as ti_summary

from distrimuse_imu_edge.training.runner import load_checkpoint_model
from distrimuse_imu_edge.inference.streaming import StreamingWindowPredictor

CLASS_NAMES = [
    "Not Moving",
    "Walk",
    "Sit Down",
    "Lay Down",
    "Turn",
    "Sit Up",
    "Stand Up",
    "Falling",
    "Hand",
]

model, ckpt = load_checkpoint_model(
    "experiments/results/edge_window_tcn_wm025_past_future/checkpoints/best.ckpt",
    map_location="cpu",
)
model = model.eval()

data_cfg = ckpt["config"]["data"]
context_len = data_cfg["context_len"]
future_context_len = data_cfg["future_context_len"]
total_context_len = context_len + future_context_len
window_size_s = data_cfg["window_size_s"]
fs = 104  # sampling rate, matches compute_model_stats' default
T = int(round(window_size_s * fs))
C = len(data_cfg["sensor_cols"])

print(
    f"Model expects {total_context_len} windows of context "
    f"({context_len} past+current, {future_context_len} future), "
    f"each window = {T} samples x {C} channels\n"
)

# --- Simulate a real streaming session: 30 windows arriving one at a time ---
n_windows = 30
torch.manual_seed(42)
session = torch.randn(n_windows, T, C)  # stand-in for real IMU samples

predictor = StreamingWindowPredictor(model, total_context_len=total_context_len)

print("Simulating incoming windows (hop-by-hop):")
for i in range(n_windows):
    logits = predictor.push(session[i])
    if logits is None:
        print(
            f"  hop {i:2d}: buffer filling ({len(predictor._buffer)}/{total_context_len})"
        )
    else:
        pred_class = CLASS_NAMES[logits.argmax().item()]
        delay = total_context_len - 1 - predictor.current_index
        predicted_for = i - delay
        print(f"  hop {i:2d}: prediction for window {predicted_for:2d} -> {pred_class}")

# --- Real peak-memory comparison, on the ACTUAL trained model ---
print("\nPeak activation memory, real checkpoint:")


def peak_kib(result):
    leaves = [l for l in result.summary_list if getattr(l, "is_leaf_layer", False)]
    peak = 0
    for layer in leaves:

        def elems(shape):
            if not shape:
                return None
            if isinstance(shape[0], (list, tuple)):
                total = 0
                for s in shape:
                    e = elems(s)
                    if e is None:
                        return None
                    total += e
                return total
            if any(not isinstance(d, int) or d < 0 for d in shape):
                return None
            total = 1
            for d in shape:
                total *= d
            return total

        in_e, out_e = elems(layer.input_size), elems(layer.output_size)
        if in_e is None or out_e is None:
            continue
        peak = max(peak, (in_e + out_e) * 4)
    return round(peak / 1024, 2)


batched_input = torch.zeros(1, total_context_len, T, C)
batched_result = ti_summary(model, input_data=batched_input, verbose=0)
print(
    f"  batched (current approach, all {total_context_len} windows): {peak_kib(batched_result)} KiB"
)

encoder_one_window = torch.zeros(1, C, T)
encoder_result = ti_summary(
    model.window_encoder, input_data=encoder_one_window, verbose=0
)
print(
    f"  streaming (1 new window, encoder only):         {peak_kib(encoder_result)} KiB"
)
