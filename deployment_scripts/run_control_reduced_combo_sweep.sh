#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Thor/Workspace/jyjeong/Isaac-GR00T}"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-thor_measurements/control_reduced_combo_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-outputs/libero_long_hf_ckpt}"
DATASET_PATH="${DATASET_PATH:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_panda}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_10_thor_onnx/dit_model_bf16.trt}"
BENCH_PYTHON="${BENCH_PYTHON:-.venv/bin/python}"

TEXT_LENGTHS="${TEXT_LENGTHS:-8 16 32 64}"
VIEW_CONFIG_PRESETS="${VIEW_CONFIG_PRESETS:-libero_dual_view}"
IMAGE_SIZES="${IMAGE_SIZES:-orig}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
WARMUP="${WARMUP:-5}"
POWER_INTERVAL_MS="${POWER_INTERVAL_MS:-2}"
INFERENCE_MODES="${INFERENCE_MODES:-tensorrt}"
FREQ_SETTLE_S="${FREQ_SETTLE_S:-1.0}"
CONFIG_PRESETS="${CONFIG_PRESETS:-control_reduced_combo_with_default}"

mkdir -p "$OUT_DIR"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output dir: $OUT_DIR"
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Dataset path: $DATASET_PATH"
echo "[INFO] TRT engine: $TRT_ENGINE_PATH"
echo "[INFO] Benchmark python: $BENCH_PYTHON"
echo "[INFO] Text lengths: $TEXT_LENGTHS"
echo "[INFO] View config presets: $VIEW_CONFIG_PRESETS"
echo "[INFO] Image sizes: $IMAGE_SIZES"
echo "[INFO] Denoising steps: $DENOISING_STEPS"
echo "[INFO] Inference modes: $INFERENCE_MODES"
echo "[INFO] Config presets: $CONFIG_PRESETS"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --embodiment_tag "$EMBODIMENT_TAG"
  --trt_engine_path "$TRT_ENGINE_PATH"
  --num_iterations "$NUM_ITERATIONS"
  --warmup "$WARMUP"
  --power_interval_ms "$POWER_INTERVAL_MS"
  --freq_settle_s "$FREQ_SETTLE_S"
  --out_dir "$OUT_DIR"
)

for mode in $INFERENCE_MODES; do
  COMMON_ARGS+=(--inference_modes "$mode")
done

for preset in $CONFIG_PRESETS; do
  COMMON_ARGS+=(--config_presets "$preset")
done

for preset in $VIEW_CONFIG_PRESETS; do
  COMMON_ARGS+=(--view_config_presets "$preset")
done

TEXT_LENGTH_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTHS; do
  TEXT_LENGTH_ARGS+=("$text_len")
done

IMAGE_SIZE_ARGS=(--image_sizes)
for image_size in $IMAGE_SIZES; do
  IMAGE_SIZE_ARGS+=("$image_size")
done

DENOISE_ARGS=(--denoising_steps)
for steps in $DENOISING_STEPS; do
  DENOISE_ARGS+=("$steps")
done

PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
  "${COMMON_ARGS[@]}" \
  "${TEXT_LENGTH_ARGS[@]}" \
  "${DENOISE_ARGS[@]}" \
  "${IMAGE_SIZE_ARGS[@]}"

echo "[INFO] Control reduced combo sweep completed."
echo "[INFO] Output: $OUT_DIR"
