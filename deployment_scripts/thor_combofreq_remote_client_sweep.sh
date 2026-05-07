#!/usr/bin/env bash
set -euo pipefail

# Run on Thor, inside the GR00T repo/container.
#
# Required:
#   PC_CLIENT_SSH
#   THOR_IP
#
# Optional:
#   LATEST_CKPT
#   PC_REPO_DIR
#   OUT_DIR
#   CPU_FREQS
#   GPU_FREQS
#   EMC_FREQS
#   INCLUDE_DEFAULT   Defaults to 1 for one all-default baseline first
#   INCLUDE_AXIS_DEFAULTS Defaults to 1 to include default as a selectable level
#   MAX_COMBOS        Optional smoke-test limiter
#   DRY_RUN           Set to 1 to print the planned combinations only

if [[ -z "${PC_CLIENT_SSH:-}" ]]; then
  echo "ERROR: set PC_CLIENT_SSH, e.g. export PC_CLIENT_SSH=user@192.168.0.10" >&2
  exit 2
fi

if [[ -z "${THOR_IP:-}" ]]; then
  echo "ERROR: set THOR_IP to the address reachable from the PC client" >&2
  exit 2
fi

if [[ -z "${LATEST_CKPT:-}" ]]; then
  LATEST_CKPT="$(ls -d outputs/libero_spatial_thor/checkpoint-* | sort -V | tail -n 1)"
  export LATEST_CKPT
fi

if [[ ! -f "${LATEST_CKPT}/config.json" ]]; then
  echo "ERROR: checkpoint config not found: ${LATEST_CKPT}/config.json" >&2
  exit 2
fi

export PYTHONPATH="/workspace/repo:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-thor_measurements/combo_sweep_vla_auto}"
INTERVAL_MS="${INTERVAL_MS:-2}"
SETTLE_S="${SETTLE_S:-1}"
LOCK_RETRIES="${LOCK_RETRIES:-3}"
LOCK_VERIFY_S="${LOCK_VERIFY_S:-0.5}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-180}"
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
N_ENVS="${N_ENVS:-1}"
PC_REPO_DIR="${PC_REPO_DIR:-~/Workspace/jyjeong/Isaac-GR00T}"
PC_PYTHON="${PC_PYTHON:-gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
ENV_NAME="${ENV_NAME:-libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-outputs/libero_spatial_thor_onnx/dit_model_bf16.trt}"
INCLUDE_DEFAULT="${INCLUDE_DEFAULT:-1}"
INCLUDE_AXIS_DEFAULTS="${INCLUDE_AXIS_DEFAULTS:-1}"
MAX_COMBOS="${MAX_COMBOS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ${CPU_FREQS+x} ]]; then
  CPU_FREQS="${CPU_FREQS}"
else
  CPU_FREQS="648MHz 756MHz 864MHz 972MHz 1.08GHz 1.188GHz 1.296GHz 1.404GHz 1.512GHz 1.620GHz 1.728GHz 1.836GHz 1.944GHz 2.052GHz 2.160GHz 2.268GHz 2.376GHz 2.484GHz 2.601GHz"
fi
if [[ ${GPU_FREQS+x} ]]; then
  GPU_FREQS="${GPU_FREQS}"
else
  GPU_FREQS="504MHz 603MHz 702MHz 801MHz 900MHz 999MHz 1.107GHz 1.206GHz 1.305GHz 1.404GHz 1.503GHz 1.575GHz"
fi
if [[ ${EMC_FREQS+x} ]]; then
  EMC_FREQS="${EMC_FREQS}"
else
  EMC_FREQS="665.6MHz 1.2GHz 1.6GHz 2.133GHz 2.75GHz 3.2GHz 3.75GHz 4.266GHz"
fi

cpu_args=()
for freq in ${CPU_FREQS}; do
  cpu_args+=(--cpu-freq "${freq}")
done
gpu_args=()
for freq in ${GPU_FREQS}; do
  gpu_args+=(--gpu-freq "${freq}")
done
emc_args=()
for freq in ${EMC_FREQS}; do
  emc_args+=(--emc-freq "${freq}")
done

extra_args=()
if [[ "${INCLUDE_DEFAULT}" == "1" ]]; then
  extra_args+=(--include-default)
fi
if [[ "${INCLUDE_AXIS_DEFAULTS}" == "1" ]]; then
  extra_args+=(--include-axis-defaults)
fi
if [[ -n "${MAX_COMBOS}" ]]; then
  extra_args+=(--max-combos "${MAX_COMBOS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

cpu_count=0
for _ in ${CPU_FREQS}; do
  ((cpu_count+=1))
done
gpu_count=0
for _ in ${GPU_FREQS}; do
  ((gpu_count+=1))
done
emc_count=0
for _ in ${EMC_FREQS}; do
  ((emc_count+=1))
done
cpu_levels=${cpu_count}
gpu_levels=${gpu_count}
emc_levels=${emc_count}
baseline_count=0
if [[ "${INCLUDE_AXIS_DEFAULTS}" == "1" ]]; then
  ((cpu_levels+=1))
  ((gpu_levels+=1))
  ((emc_levels+=1))
else
  if [[ "${INCLUDE_DEFAULT}" == "1" ]]; then
    baseline_count=1
  fi
fi
total_runs=$((cpu_levels * gpu_levels * emc_levels + baseline_count))

echo "[INFO] PC_CLIENT_SSH=${PC_CLIENT_SSH}"
echo "[INFO] THOR_IP=${THOR_IP}"
echo "[INFO] LATEST_CKPT=${LATEST_CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] INCLUDE_DEFAULT=${INCLUDE_DEFAULT}"
echo "[INFO] INCLUDE_AXIS_DEFAULTS=${INCLUDE_AXIS_DEFAULTS}"
echo "[INFO] CPU_FREQS=${CPU_FREQS}"
echo "[INFO] GPU_FREQS=${GPU_FREQS}"
echo "[INFO] EMC_FREQS=${EMC_FREQS}"
echo "[INFO] Planned runs=${total_runs}"

python deployment_scripts/thor_combofreq_power_sweep.py \
  --out-dir "${OUT_DIR}" \
  --interval-ms "${INTERVAL_MS}" \
  --settle-s "${SETTLE_S}" \
  --lock-retries "${LOCK_RETRIES}" \
  --lock-verify-s "${LOCK_VERIFY_S}" \
  --skip-failed-locks \
  "${cpu_args[@]}" \
  "${gpu_args[@]}" \
  "${emc_args[@]}" \
  "${extra_args[@]}" \
  --unlock-at-end \
  -- \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/repo:${PYTHONPATH:-}

    python -u gr00t/eval/run_gr00t_server.py \
      --model-path "${LATEST_CKPT}" \
      --embodiment-tag LIBERO_PANDA \
      --use-sim-policy-wrapper \
      --inference-mode tensorrt \
      --trt-engine-path "'"${TRT_ENGINE_PATH}"'" &
    server_pid=$!

    cleanup() {
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    echo "[INFO] Waiting for server port 5555..."
    python - <<'"'"'PY'"'"'
import os
import socket
import sys
import time

deadline = time.time() + float(os.environ.get("WAIT_TIMEOUT_S", "180"))
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", 5555)) == 0:
            print("[INFO] Server is reachable on 127.0.0.1:5555")
            sys.exit(0)
    time.sleep(1.0)
print("[ERROR] Timed out waiting for server port 5555", file=sys.stderr)
sys.exit(1)
PY

    echo "[INFO] Running remote PC client..."
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "'"${PC_CLIENT_SSH}"'" \
      "cd '"${PC_REPO_DIR}"' && '"${PC_PYTHON}"' gr00t/eval/rollout_policy.py \
        --n_episodes '"${N_EPISODES}"' \
        --policy_client_host '"${THOR_IP}"' \
        --policy_client_port 5555 \
        --max_episode_steps='"${MAX_EPISODE_STEPS}"' \
        --env_name '"${ENV_NAME}"' \
        --n_action_steps '"${N_ACTION_STEPS}"' \
        --n_envs '"${N_ENVS}"'"

    echo "[INFO] Remote client finished."
  '
