#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Thor/Workspace/jyjeong/Isaac-GR00T}"
cd "$REPO_ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-thor_measurements/control_fixed_period_deadline_grid_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-outputs/libero_long_hf_ckpt}"
DATASET_PATH="${DATASET_PATH:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_panda}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_10_thor_onnx/dit_model_bf16.trt}"
BENCH_PYTHON="${BENCH_PYTHON:-.venv/bin/python}"

TEXT_LENGTHS="${TEXT_LENGTHS:-8 64 128 256}"
VIEW_CONFIGS="${VIEW_CONFIGS:-image_only|image wrist_only|wrist_image both_views|image,wrist_image}"
DENOISING_STEPS="${DENOISING_STEPS:-1 2 4 8}"
DEADLINE_GRID_BY_DENOISING="${DEADLINE_GRID_BY_DENOISING:-1:70,75,80,85;2:85,90,95,100;4:110,115,120,125;8:155,160,165,170}"
FIXED_PERIOD_BODY="${FIXED_PERIOD_BODY:-components}"
IMAGE_SIZES="${IMAGE_SIZES:-orig}"

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
RUN_DEFAULT="${RUN_DEFAULT:-0}"
RUN_LOCKED_SWEEP="${RUN_LOCKED_SWEEP:-1}"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Deadline grid: $DEADLINE_GRID_BY_DENOISING"
echo "[INFO] Text lengths: $TEXT_LENGTHS"
echo "[INFO] View configs: $VIEW_CONFIGS"
echo "[INFO] Denoising steps: $DENOISING_STEPS"
echo "[INFO] Fixed period body: $FIXED_PERIOD_BODY"
echo "[INFO] Num iterations: $NUM_ITERATIONS"
echo "[INFO] Run default DVFS baseline: $RUN_DEFAULT"
echo "[INFO] Run locked frequency sweep: $RUN_LOCKED_SWEEP"

deadline_list_for_denoising() {
  local target="$1"
  local spec="$DEADLINE_GRID_BY_DENOISING"
  local item
  IFS=';' read -r -a items <<< "$spec"
  for item in "${items[@]}"; do
    local step="${item%%:*}"
    local vals="${item#*:}"
    if [[ "$step" == "$target" ]]; then
      echo "${vals//,/ }"
      return 0
    fi
  done
  echo "[ERROR] Missing deadline list for denoising step $target in DEADLINE_GRID_BY_DENOISING=$spec" >&2
  return 1
}

COMMON_BASE_ARGS=(
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
)
if [[ "$SHUFFLE_CONDITIONS" == "1" ]]; then
  COMMON_BASE_ARGS+=(--shuffle_conditions)
fi

MODE_ARGS=()
for mode in $INFERENCE_MODES; do
  MODE_ARGS+=(--inference_modes "$mode")
done

IMAGE_SIZE_ARGS=(--image_sizes)
for image_size in $IMAGE_SIZES; do
  IMAGE_SIZE_ARGS+=("$image_size")
done

TEXT_ARGS=(--text_lengths)
for text_len in $TEXT_LENGTHS; do
  TEXT_ARGS+=("$text_len")
done

VIEW_ARGS=(--view_config)
for view_config in $VIEW_CONFIGS; do
  VIEW_ARGS+=("$view_config")
done

run_grid_case() {
  local label="$1"
  local root="$2"
  local denoise="$3"
  local deadline_ms="$4"
  shift 4
  local config_args=("$@")
  local out_dir="${root}/d${denoise}_p${deadline_ms}ms/text_view_fullfactor"
  mkdir -p "$out_dir"

  echo "[INFO] ${label}: d=${denoise}, deadline=${deadline_ms} ms"
  PYTHONPATH=. "$BENCH_PYTHON" scripts/deployment/benchmark_input_sweep.py \
    "${COMMON_BASE_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${config_args[@]}" \
    --denoising_steps "$denoise" \
    --fixed_period_ms "$deadline_ms" \
    "${VIEW_ARGS[@]}" \
    "${TEXT_ARGS[@]}" \
    "${IMAGE_SIZE_ARGS[@]}" \
    --out_dir "$out_dir"
}

DEFAULT_CONFIG_ARGS=(--config "default_dvfs|default|default|default")

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

TEXT_COUNT=$(wc -w <<< "$TEXT_LENGTHS")
VIEW_COUNT=$(wc -w <<< "$VIEW_CONFIGS")
FREQ_COUNT=$(( ${#CPU_FREQ_ARRAY[@]} * ${#GPU_FREQ_ARRAY[@]} * ${#EMC_FREQ_ARRAY[@]} ))
DEADLINE_COUNT=0
for denoise in $DENOISING_STEPS; do
  read -r -a deadlines <<< "$(deadline_list_for_denoising "$denoise")"
  DEADLINE_COUNT=$(( DEADLINE_COUNT + ${#deadlines[@]} ))
done
LOCKED_ROWS=$(( DEADLINE_COUNT * TEXT_COUNT * VIEW_COUNT * FREQ_COUNT * REPEAT_RUNS ))
DEFAULT_ROWS=$(( DEADLINE_COUNT * TEXT_COUNT * VIEW_COUNT * REPEAT_RUNS ))
echo "[INFO] Locked config count per condition: $FREQ_COUNT"
echo "[INFO] Planned locked rows: $LOCKED_ROWS"
echo "[INFO] Planned default rows: $DEFAULT_ROWS"

for denoise in $DENOISING_STEPS; do
  read -r -a deadlines <<< "$(deadline_list_for_denoising "$denoise")"
  for deadline_ms in "${deadlines[@]}"; do
    if [[ "$RUN_DEFAULT" == "1" ]]; then
      run_grid_case "default DVFS baseline" "$OUT_ROOT/default_dvfs" "$denoise" "$deadline_ms" "${DEFAULT_CONFIG_ARGS[@]}"
    fi
    if [[ "$RUN_LOCKED_SWEEP" == "1" ]]; then
      run_grid_case "locked frequency sweep" "$OUT_ROOT/locked_sweep" "$denoise" "$deadline_ms" "${LOCKED_CONFIG_ARGS[@]}"
    fi
  done
done

echo "[INFO] Fixed-period deadline-grid control sweeps completed."
echo "[INFO] Output root: $OUT_ROOT"
