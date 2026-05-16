#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Thor/Workspace/jyjeong/Isaac-GR00T}"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-thor_measurements/control_text_and_view_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-outputs/libero_long_hf_ckpt}"
DATASET_PATH="${DATASET_PATH:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_panda}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_10_thor_onnx/dit_model_bf16.trt}"
BENCH_PYTHON="${BENCH_PYTHON:-.venv/bin/python}"

TEXT_LENGTHS_TEXT_ONLY="${TEXT_LENGTHS_TEXT_ONLY:-8 64 128 256}"
TEXT_LENGTH_VIEW_ONLY="${TEXT_LENGTH_VIEW_ONLY:-64}"
IMAGE_SIZES="${IMAGE_SIZES:-orig}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
WARMUP="${WARMUP:-5}"
REPEAT_RUNS="${REPEAT_RUNS:-3}"
POWER_INTERVAL_MS="${POWER_INTERVAL_MS:-2}"
INFERENCE_MODES="${INFERENCE_MODES:-tensorrt}"
FREQ_SETTLE_S="${FREQ_SETTLE_S:-1.0}"
CONFIG_PRESETS="${CONFIG_PRESETS:-control_reduced_combo}"
USE_FIXED_GRID="${USE_FIXED_GRID:-0}"
INCLUDE_ALL_DEFAULT="${INCLUDE_ALL_DEFAULT:-1}"
CPU_FREQS="${CPU_FREQS:-1.836GHz 2.052GHz 2.160GHz 2.376GHz 2.484GHz 2.601GHz}"
GPU_FREQS="${GPU_FREQS:-900MHz 1.107GHz 1.206GHz 1.305GHz 1.503GHz 1.575GHz}"
EMC_FREQS="${EMC_FREQS:-665.6MHz 2.75GHz 3.2GHz 4.266GHz}"
SHUFFLE_CONDITIONS="${SHUFFLE_CONDITIONS:-0}"
VIEW_CONFIGS_TEXT_ONLY="${VIEW_CONFIGS_TEXT_ONLY:-both_views|image,wrist_image}"
VIEW_CONFIG_PRESETS_VIEW_ONLY="${VIEW_CONFIG_PRESETS_VIEW_ONLY:-up_to_three_views}"
VIEW_CONFIGS_VIEW_ONLY="${VIEW_CONFIGS_VIEW_ONLY:-}"

TEXT_OUT="${OUT_ROOT}/text_only_both_views"
VIEW_OUT="${OUT_ROOT}/viewcount_only"
mkdir -p "$TEXT_OUT" "$VIEW_OUT"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Dataset path: $DATASET_PATH"
echo "[INFO] TRT engine: $TRT_ENGINE_PATH"
echo "[INFO] Benchmark python: $BENCH_PYTHON"
echo "[INFO] Use fixed grid: $USE_FIXED_GRID"
if [[ "$USE_FIXED_GRID" == "1" ]]; then
  echo "[INFO] Include all-default point: $INCLUDE_ALL_DEFAULT"
  echo "[INFO] CPU freqs: $CPU_FREQS"
  echo "[INFO] GPU freqs: $GPU_FREQS"
  echo "[INFO] EMC freqs: $EMC_FREQS"
else
  echo "[INFO] Config presets: $CONFIG_PRESETS"
fi
echo "[INFO] Text-only lengths: $TEXT_LENGTHS_TEXT_ONLY"
echo "[INFO] View-count text length: $TEXT_LENGTH_VIEW_ONLY"
echo "[INFO] Denoising steps: $DENOISING_STEPS"
echo "[INFO] Repeat runs: $REPEAT_RUNS"
echo "[INFO] Shuffle conditions: $SHUFFLE_CONDITIONS"
echo "[INFO] Text-only view configs: $VIEW_CONFIGS_TEXT_ONLY"
echo "[INFO] View-count presets: $VIEW_CONFIG_PRESETS_VIEW_ONLY"
echo "[INFO] View-count explicit configs: ${VIEW_CONFIGS_VIEW_ONLY:-<none>}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --embodiment_tag "$EMBODIMENT_TAG"
  --trt_engine_path "$TRT_ENGINE_PATH"
  --num_iterations "$NUM_ITERATIONS"
  --warmup "$WARMUP"
  --repeat_runs "$REPEAT_RUNS"
  --power_interval_ms "$POWER_INTERVAL_MS"
  --freq_settle_s "$FREQ_SETTLE_S"
)
if [[ "$SHUFFLE_CONDITIONS" == "1" ]]; then
  COMMON_ARGS+=(--shuffle_conditions)
fi

MODE_ARGS=()
for mode in $INFERENCE_MODES; do
  MODE_ARGS+=(--inference_modes "$mode")
done

CONFIG_ARGS=()
if [[ "$USE_FIXED_GRID" == "1" ]]; then
  read -r -a CPU_FREQ_ARRAY <<< "$CPU_FREQS"
  read -r -a GPU_FREQ_ARRAY <<< "$GPU_FREQS"
  read -r -a EMC_FREQ_ARRAY <<< "$EMC_FREQS"

  if [[ "${#CPU_FREQ_ARRAY[@]}" -eq 0 || "${#GPU_FREQ_ARRAY[@]}" -eq 0 || "${#EMC_FREQ_ARRAY[@]}" -eq 0 ]]; then
    echo "[ERROR] USE_FIXED_GRID=1 requires non-empty CPU_FREQS, GPU_FREQS, and EMC_FREQS"
    exit 1
  fi

  FIXED_COMBO_COUNT=$(( ${#CPU_FREQ_ARRAY[@]} * ${#GPU_FREQ_ARRAY[@]} * ${#EMC_FREQ_ARRAY[@]} ))
  CONFIG_ARGS=(--config)
  if [[ "$INCLUDE_ALL_DEFAULT" == "1" ]]; then
    CONFIG_ARGS+=("all_default|default|default|default")
  fi
  for cpu_freq in "${CPU_FREQ_ARRAY[@]}"; do
    for gpu_freq in "${GPU_FREQ_ARRAY[@]}"; do
      for emc_freq in "${EMC_FREQ_ARRAY[@]}"; do
        CONFIG_ARGS+=(
          "cpu_${cpu_freq}_gpu_${gpu_freq}_emc_${emc_freq}|${cpu_freq}|${gpu_freq}|${emc_freq}"
        )
      done
    done
  done
  TOTAL_CONFIG_COUNT=$FIXED_COMBO_COUNT
  if [[ "$INCLUDE_ALL_DEFAULT" == "1" ]]; then
    TOTAL_CONFIG_COUNT=$(( TOTAL_CONFIG_COUNT + 1 ))
  fi
  echo "[INFO] Fixed-only combo count: $FIXED_COMBO_COUNT"
  echo "[INFO] Total configs passed to benchmark: $TOTAL_CONFIG_COUNT"
else
  for preset in $CONFIG_PRESETS; do
    CONFIG_ARGS+=(--config_presets "$preset")
  done
fi

IMAGE_SIZE_ARGS=(--image_sizes)
for image_size in $IMAGE_SIZES; do
  IMAGE_SIZE_ARGS+=("$image_size")
done

DENOISE_ARGS=(--denoising_steps)
for steps in $DENOISING_STEPS; do
  DENOISE_ARGS+=("$steps")
done

TEXT_VIEW_ARGS=(--view_config)
for view_config in $VIEW_CONFIGS_TEXT_ONLY; do
  TEXT_VIEW_ARGS+=("$view_config")
done

VIEW_ONLY_VIEW_ARGS=()
if [[ -n "$VIEW_CONFIG_PRESETS_VIEW_ONLY" ]]; then
  for preset in $VIEW_CONFIG_PRESETS_VIEW_ONLY; do
    VIEW_ONLY_VIEW_ARGS+=(--view_config_presets "$preset")
  done
fi
if [[ -n "$VIEW_CONFIGS_VIEW_ONLY" ]]; then
  VIEW_ONLY_VIEW_ARGS+=(--view_config)
  for view_config in $VIEW_CONFIGS_VIEW_ONLY; do
    VIEW_ONLY_VIEW_ARGS+=("$view_config")
  done
fi

echo "[INFO] Running control sweep 1/2: text-only with both_views"
TEXT_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTHS_TEXT_ONLY; do
  TEXT_ARGS+=("$text_len")
done

PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${CONFIG_ARGS[@]}" \
  "${DENOISE_ARGS[@]}" \
  "${TEXT_VIEW_ARGS[@]}" \
  "${TEXT_ARGS[@]}" \
  "${IMAGE_SIZE_ARGS[@]}" \
  --out_dir "$TEXT_OUT"

echo "[INFO] Running control sweep 2/2: 1-view vs 2-views at fixed text length"
VIEW_TEXT_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTH_VIEW_ONLY; do
  VIEW_TEXT_ARGS+=("$text_len")
done

PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${CONFIG_ARGS[@]}" \
  "${DENOISE_ARGS[@]}" \
  "${VIEW_ONLY_VIEW_ARGS[@]}" \
  "${VIEW_TEXT_ARGS[@]}" \
  "${IMAGE_SIZE_ARGS[@]}" \
  --out_dir "$VIEW_OUT"

echo "[INFO] Control text/view sweeps completed."
echo "[INFO] Text-only output: $TEXT_OUT"
echo "[INFO] View-count-only output: $VIEW_OUT"
