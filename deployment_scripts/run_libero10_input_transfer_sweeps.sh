#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Thor/Workspace/jyjeong/Isaac-GR00T}"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-thor_measurements/input_transfer_libero10_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-outputs/libero_long_hf_ckpt}"
DATASET_PATH="${DATASET_PATH:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_panda}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_10_thor_onnx/dit_model_bf16.trt}"
BENCH_PYTHON="${BENCH_PYTHON:-.venv/bin/python}"

TEXT_LENGTHS="${TEXT_LENGTHS:-8 16 32 64}"
IMAGE_SIZES="${IMAGE_SIZES:-224x224 256x256 320x320}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
WARMUP="${WARMUP:-5}"
POWER_INTERVAL_MS="${POWER_INTERVAL_MS:-2}"
INFERENCE_MODES="${INFERENCE_MODES:-tensorrt}"

SPATIAL_OUT="${OUT_ROOT}/spatial_candidates_to_libero10"
LIBERO10_OUT="${OUT_ROOT}/libero10_candidates_to_libero10"

mkdir -p "$SPATIAL_OUT" "$LIBERO10_OUT"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Dataset path: $DATASET_PATH"
echo "[INFO] TRT engine: $TRT_ENGINE_PATH"
echo "[INFO] Benchmark python: $BENCH_PYTHON"
echo "[INFO] Text lengths: $TEXT_LENGTHS"
echo "[INFO] Image sizes: $IMAGE_SIZES"
echo "[INFO] Inference modes: $INFERENCE_MODES"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --embodiment_tag "$EMBODIMENT_TAG"
  --trt_engine_path "$TRT_ENGINE_PATH"
  --num_iterations "$NUM_ITERATIONS"
  --warmup "$WARMUP"
  --power_interval_ms "$POWER_INTERVAL_MS"
)

for mode in $INFERENCE_MODES; do
  COMMON_ARGS+=(--inference_modes "$mode")
done

TEXT_LENGTH_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTHS; do
  TEXT_LENGTH_ARGS+=("$text_len")
done

IMAGE_SIZE_ARGS=(--image_sizes)
for image_size in $IMAGE_SIZES; do
  IMAGE_SIZE_ARGS+=("$image_size")
done

echo "[INFO] Running transfer sweep 1/2: LIBERO-spatial best candidates -> LIBERO-10 input sweep"
PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
  "${COMMON_ARGS[@]}" \
  "${TEXT_LENGTH_ARGS[@]}" \
  "${IMAGE_SIZE_ARGS[@]}" \
  --config_presets libero_spatial_candidates \
  --out_dir "$SPATIAL_OUT"

echo "[INFO] Running transfer sweep 2/2: LIBERO-10 best candidates -> LIBERO-10 input sweep"
PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
  "${COMMON_ARGS[@]}" \
  "${TEXT_LENGTH_ARGS[@]}" \
  "${IMAGE_SIZE_ARGS[@]}" \
  --config_presets libero10_candidates \
  --out_dir "$LIBERO10_OUT"

echo "[INFO] All transfer sweeps completed."
echo "[INFO] Spatial candidates output: $SPATIAL_OUT"
echo "[INFO] LIBERO-10 candidates output: $LIBERO10_OUT"
