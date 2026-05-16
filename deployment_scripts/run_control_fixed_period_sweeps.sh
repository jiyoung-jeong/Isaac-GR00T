#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Thor/Workspace/jyjeong/Isaac-GR00T}"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-thor_measurements/control_fixed_period_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-outputs/libero_long_hf_ckpt}"
DATASET_PATH="${DATASET_PATH:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_panda}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_10_thor_onnx/dit_model_bf16.trt}"
BENCH_PYTHON="${BENCH_PYTHON:-.venv/bin/python}"

TEXT_LENGTHS_TEXT_ONLY="${TEXT_LENGTHS_TEXT_ONLY:-8 64 128 256}"
TEXT_LENGTH_VIEW_ONLY="${TEXT_LENGTH_VIEW_ONLY:-64}"
IMAGE_SIZES="${IMAGE_SIZES:-orig}"
DENOISING_STEPS="${DENOISING_STEPS:-1 2 4 8}"
PERIOD_BY_DENOISING="${PERIOD_BY_DENOISING:-1:70,2:85,4:110,8:155}"
FIXED_PERIOD_BODY="${FIXED_PERIOD_BODY:-components}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
WARMUP="${WARMUP:-5}"
REPEAT_RUNS="${REPEAT_RUNS:-1}"
POWER_INTERVAL_MS="${POWER_INTERVAL_MS:-2}"
INFERENCE_MODES="${INFERENCE_MODES:-tensorrt}"
FREQ_SETTLE_S="${FREQ_SETTLE_S:-1.0}"
CPU_FREQS="${CPU_FREQS:-1.836GHz 2.052GHz 2.160GHz 2.376GHz 2.484GHz 2.601GHz}"
GPU_FREQS="${GPU_FREQS:-900MHz 1.107GHz 1.206GHz 1.305GHz 1.503GHz 1.575GHz}"
EMC_FREQS="${EMC_FREQS:-665.6MHz 2.75GHz 3.2GHz 4.266GHz}"
SHUFFLE_CONDITIONS="${SHUFFLE_CONDITIONS:-1}"
RUN_DEFAULT="${RUN_DEFAULT:-1}"
RUN_LOCKED_SWEEP="${RUN_LOCKED_SWEEP:-1}"
VIEW_CONFIGS_TEXT_ONLY="${VIEW_CONFIGS_TEXT_ONLY:-both_views|image,wrist_image}"
VIEW_CONFIGS_VIEW_ONLY="${VIEW_CONFIGS_VIEW_ONLY:-image_only|image wrist_only|wrist_image both_views|image,wrist_image}"

DEFAULT_ROOT="${OUT_ROOT}/default_dvfs"
LOCKED_ROOT="${OUT_ROOT}/locked_sweep"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Dataset path: $DATASET_PATH"
echo "[INFO] TRT engine: $TRT_ENGINE_PATH"
echo "[INFO] Benchmark python: $BENCH_PYTHON"
echo "[INFO] Fixed period by denoising: $PERIOD_BY_DENOISING"
echo "[INFO] Fixed period body: $FIXED_PERIOD_BODY"
echo "[INFO] Text-only lengths: $TEXT_LENGTHS_TEXT_ONLY"
echo "[INFO] View-count text length: $TEXT_LENGTH_VIEW_ONLY"
echo "[INFO] View configs: $VIEW_CONFIGS_VIEW_ONLY"
echo "[INFO] Denoising steps: $DENOISING_STEPS"
echo "[INFO] Repeat runs: $REPEAT_RUNS"
echo "[INFO] Num iterations: $NUM_ITERATIONS"
echo "[INFO] Run default DVFS baseline: $RUN_DEFAULT"
echo "[INFO] Run locked frequency sweep: $RUN_LOCKED_SWEEP"

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
  --fixed_period
  --fixed_period_body "$FIXED_PERIOD_BODY"
  --period_by_denoising "$PERIOD_BY_DENOISING"
)
if [[ "$SHUFFLE_CONDITIONS" == "1" ]]; then
  COMMON_ARGS+=(--shuffle_conditions)
fi

MODE_ARGS=()
for mode in $INFERENCE_MODES; do
  MODE_ARGS+=(--inference_modes "$mode")
done

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

VIEW_ONLY_VIEW_ARGS=(--view_config)
for view_config in $VIEW_CONFIGS_VIEW_ONLY; do
  VIEW_ONLY_VIEW_ARGS+=("$view_config")
done

TEXT_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTHS_TEXT_ONLY; do
  TEXT_ARGS+=("$text_len")
done

VIEW_TEXT_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTH_VIEW_ONLY; do
  VIEW_TEXT_ARGS+=("$text_len")
done

run_pair() {
  local label="$1"
  local root="$2"
  shift 2
  local config_args=("$@")
  local text_out="${root}/text_only_both_views"
  local view_out="${root}/viewcount_only"
  mkdir -p "$text_out" "$view_out"

  echo "[INFO] ${label}: text-only fixed-period sweep"
  PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${config_args[@]}" \
    "${DENOISE_ARGS[@]}" \
    "${TEXT_VIEW_ARGS[@]}" \
    "${TEXT_ARGS[@]}" \
    "${IMAGE_SIZE_ARGS[@]}" \
    --out_dir "$text_out"

  echo "[INFO] ${label}: view-count fixed-period sweep"
  PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${config_args[@]}" \
    "${DENOISE_ARGS[@]}" \
    "${VIEW_ONLY_VIEW_ARGS[@]}" \
    "${VIEW_TEXT_ARGS[@]}" \
    "${IMAGE_SIZE_ARGS[@]}" \
    --out_dir "$view_out"
}

if [[ "$RUN_DEFAULT" == "1" ]]; then
  DEFAULT_CONFIG_ARGS=(--config "default_dvfs|default|default|default")
  run_pair "default DVFS baseline" "$DEFAULT_ROOT" "${DEFAULT_CONFIG_ARGS[@]}"
fi

if [[ "$RUN_LOCKED_SWEEP" == "1" ]]; then
  read -r -a CPU_FREQ_ARRAY <<< "$CPU_FREQS"
  read -r -a GPU_FREQ_ARRAY <<< "$GPU_FREQS"
  read -r -a EMC_FREQ_ARRAY <<< "$EMC_FREQS"
  LOCKED_CONFIG_ARGS=(--config)
  for cpu_freq in "${CPU_FREQ_ARRAY[@]}"; do
    for gpu_freq in "${GPU_FREQ_ARRAY[@]}"; do
      for emc_freq in "${EMC_FREQ_ARRAY[@]}"; do
        LOCKED_CONFIG_ARGS+=(
          "cpu_${cpu_freq}_gpu_${gpu_freq}_emc_${emc_freq}|${cpu_freq}|${gpu_freq}|${emc_freq}"
        )
      done
    done
  done
  echo "[INFO] Locked config count: $(( ${#CPU_FREQ_ARRAY[@]} * ${#GPU_FREQ_ARRAY[@]} * ${#EMC_FREQ_ARRAY[@]} ))"
  run_pair "locked frequency sweep" "$LOCKED_ROOT" "${LOCKED_CONFIG_ARGS[@]}"
fi

echo "[INFO] Fixed-period control sweeps completed."
echo "[INFO] Output root: $OUT_ROOT"
