#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-experiments/results}"
WISDM_ROOT="${WISDM_ROOT:-cache/public/wisdm19}"
WISDM_SPLIT_DIR="$WISDM_ROOT/splits/watch_accel_gyro"
MODELS="${MODELS:-edge_cnn edge_tcn}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
WIDTH_MULT="${WIDTH_MULT:-}"
ALIGN_TOLERANCE_S="${ALIGN_TOLERANCE_S:-0.05}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

checkpoint_exists() {
  [[ -f "$RESULTS_DIR/$1/checkpoints/best.ckpt" ]]
}

run_train() {
  local config="$1"
  local model="$2"
  local run_name="$3"
  shift 3

  if [[ "$SKIP_EXISTING" == "1" ]] && checkpoint_exists "$run_name"; then
    printf '\n==> skip existing run: %s\n' "$run_name"
    return
  fi

  local args=(
    uv run imu-edge-train
    --config "$config"
    --model "$model"
    --run-name "$run_name"
  )

  if [[ -n "$MAX_EPOCHS" ]]; then
    args+=(--max-epochs "$MAX_EPOCHS")
  fi
  if [[ -n "$WIDTH_MULT" ]]; then
    args+=(--width-mult "$WIDTH_MULT")
  fi

  args+=("$@")
  run "${args[@]}"
}

prepare_args=(
  uv run imu-edge-prepare-wisdm
  --root "$WISDM_ROOT"
  --align-tolerance-s "$ALIGN_TOLERANCE_S"
)

if [[ -n "${WISDM_SOURCE_ZIP:-}" ]]; then
  prepare_args+=(--source-zip "$WISDM_SOURCE_ZIP")
fi
if [[ -n "${WISDM_EXTRACTED_DIR:-}" ]]; then
  prepare_args+=(--extracted-dir "$WISDM_EXTRACTED_DIR")
fi
if [[ "${FORCE_DOWNLOAD:-0}" == "1" ]]; then
  prepare_args+=(--force-download)
fi

if [[ "${SKIP_WISDM_PREPARE:-0}" == "1" ]]; then
  printf '\n==> skip WISDM preparation\n'
elif [[ "${FORCE_WISDM_PREPARE:-0}" != "1" \
  && -f "$WISDM_SPLIT_DIR/train.parquet" \
  && -f "$WISDM_SPLIT_DIR/val.parquet" \
  && -f "$WISDM_SPLIT_DIR/test.parquet" ]]; then
  printf '\n==> skip WISDM preparation; found %s/{train,val,test}.parquet\n' "$WISDM_SPLIT_DIR"
else
  run "${prepare_args[@]}"
fi

for model in $MODELS; do
  run_train "configs/benchmark.yaml" "$model" "${model}_scratch_v35"
done

for model in $MODELS; do
  run_train "configs/pretrain_wisdm19.yaml" "$model" "${model}_wisdm19_pretrain"
done

for model in $MODELS; do
  pretrain_ckpt="$RESULTS_DIR/${model}_wisdm19_pretrain/checkpoints/best.ckpt"
  if [[ ! -f "$pretrain_ckpt" ]]; then
    printf '\nMissing pretraining checkpoint: %s\n' "$pretrain_ckpt" >&2
    exit 1
  fi
  run_train \
    "configs/benchmark.yaml" \
    "$model" \
    "${model}_wisdm19_finetune_v35" \
    --init-checkpoint "$pretrain_ckpt"
done

run uv run imu-edge-benchmark --results-dir "$RESULTS_DIR"

printf '\nDone. Summary: %s/benchmark_summary.csv\n' "$RESULTS_DIR"
