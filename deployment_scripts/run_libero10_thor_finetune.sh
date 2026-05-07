#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_REPO="${DATASET_REPO:-IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot}"
DATASET_DIR="${DATASET_DIR:-examples/LIBERO/libero_10_no_noops_1.0.0_lerobot}"
MODALITY_JSON="${MODALITY_JSON:-examples/LIBERO/modality.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero_10_thor}"

if [[ "$DATASET_DIR" != /* ]]; then
    DATASET_DIR="$REPO_ROOT/$DATASET_DIR"
fi
if [[ "$MODALITY_JSON" != /* ]]; then
    MODALITY_JSON="$REPO_ROOT/$MODALITY_JSON"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi

NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.8}"

DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"
HF_CLI="${HF_CLI:-huggingface-cli}"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] NUM_GPUS=$NUM_GPUS MAX_STEPS=$MAX_STEPS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE SAVE_STEPS=$SAVE_STEPS"
echo "[INFO] USE_WANDB=$USE_WANDB WANDB_MODE=$WANDB_MODE STATE_DROPOUT_PROB=$STATE_DROPOUT_PROB"

dataset_complete() {
    local episodes_file="$DATASET_DIR/meta/episodes.jsonl"
    local rgb_dir="$DATASET_DIR/videos/chunk-000/observation.images.image"
    local wrist_dir="$DATASET_DIR/videos/chunk-000/observation.images.wrist_image"

    if [ ! -f "$episodes_file" ]; then
        return 1
    fi

    local episode_count expected_last
    episode_count="$(wc -l < "$episodes_file")"
    if [ -z "$episode_count" ] || [ "$episode_count" -le 0 ]; then
        return 1
    fi
    expected_last=$((episode_count - 1))

    test -f "$rgb_dir/episode_$(printf '%06d' "$expected_last").mp4" &&
        test -f "$wrist_dir/episode_$(printf '%06d' "$expected_last").mp4"
}

if [ ! -d "$DATASET_DIR" ] || ! dataset_complete; then
    if [ "$DOWNLOAD_IF_MISSING" != "1" ]; then
        echo "[ERROR] Dataset missing: $DATASET_DIR" >&2
        echo "[HINT] Set DOWNLOAD_IF_MISSING=1 or download it manually:" >&2
        echo "  huggingface-cli download --repo-type dataset $DATASET_REPO --local-dir $DATASET_DIR" >&2
        exit 1
    fi
    echo "[INFO] Dataset missing or incomplete; downloading $DATASET_REPO -> $DATASET_DIR"
    "$HF_CLI" download \
        --repo-type dataset "$DATASET_REPO" \
        --local-dir "$DATASET_DIR"
fi

if [ ! -f "$MODALITY_JSON" ]; then
    echo "[ERROR] Missing modality config: $MODALITY_JSON" >&2
    exit 1
fi

mkdir -p "$DATASET_DIR/meta"
cp -f "$MODALITY_JSON" "$DATASET_DIR/meta/"
mkdir -p "$OUTPUT_DIR"

export USE_WANDB
export WANDB_MODE
export NUM_GPUS
export MAX_STEPS
export GLOBAL_BATCH_SIZE
export SAVE_STEPS
export DATALOADER_NUM_WORKERS

echo "[INFO] Starting LIBERO-10 finetune on Thor"

exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_DIR/" \
    --embodiment-tag LIBERO_PANDA \
    --output-dir "$OUTPUT_DIR" \
    --state-dropout-prob "$STATE_DROPOUT_PROB"
