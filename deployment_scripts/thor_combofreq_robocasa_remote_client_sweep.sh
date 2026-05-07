#!/usr/bin/env bash
set -euo pipefail

# Remote RoboCasa combo sweep:
#   - Thor runs the GR00T server and all CPU/GPU/EMC frequency control
#   - PC runs the RoboCasa rollout client over SSH
#
# Required:
#   PC_CLIENT_SSH
#   THOR_IP
#
# Optional:
#   MODEL_PATH
#   OUT_DIR
#   CPU_FREQS
#   GPU_FREQS
#   EMC_FREQS
#   PC_REPO_DIR
#   PC_PYTHON
#   ENV_NAME
#   N_EPISODES
#   MAX_COMBOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${PC_CLIENT_SSH:-}" ]]; then
  echo "ERROR: set PC_CLIENT_SSH, e.g. export PC_CLIENT_SSH=user@192.168.0.10" >&2
  exit 2
fi

if [[ -z "${THOR_IP:-}" ]]; then
  echo "ERROR: set THOR_IP to the address reachable from the PC client" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
INFERENCE_MODE="${INFERENCE_MODE:-pytorch}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-}"
OUT_DIR="${OUT_DIR:-thor_measurements/robocasa_combo_sweep_auto}"
INTERVAL_MS="${INTERVAL_MS:-2}"
SETTLE_S="${SETTLE_S:-1}"
LOCK_RETRIES="${LOCK_RETRIES:-3}"
LOCK_VERIFY_S="${LOCK_VERIFY_S:-0.5}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-180}"
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
N_ENVS="${N_ENVS:-1}"
POLICY_CLIENT_PORT="${POLICY_CLIENT_PORT:-5555}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GR1}"
ENV_NAME="${ENV_NAME:-gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env}"
PC_REPO_DIR="${PC_REPO_DIR:-~/Workspace/jyjeong/Isaac-GR00T}"
PC_PYTHON="${PC_PYTHON:-gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python}"
INCLUDE_DEFAULT="${INCLUDE_DEFAULT:-1}"
INCLUDE_AXIS_DEFAULTS="${INCLUDE_AXIS_DEFAULTS:-1}"
MAX_COMBOS="${MAX_COMBOS:-}"
DRY_RUN="${DRY_RUN:-0}"
CLIENT_EXTRA_ARGS="${CLIENT_EXTRA_ARGS:-}"

if [[ "${INFERENCE_MODE}" == "tensorrt" && -z "${TRT_ENGINE_PATH}" ]]; then
  echo "ERROR: set TRT_ENGINE_PATH when INFERENCE_MODE=tensorrt" >&2
  exit 2
fi

SERVER_TRT_PATH_ARG=""
if [[ "${INFERENCE_MODE}" == "tensorrt" ]]; then
  SERVER_TRT_PATH_ARG="--trt-engine-path ${TRT_ENGINE_PATH}"
fi

if [[ ${CPU_FREQS+x} ]]; then
  CPU_FREQS="${CPU_FREQS}"
else
  CPU_FREQS="648MHz 972MHz 1.242GHz 1.566GHz 1.836GHz 2.160GHz 2.430GHz 2.601GHz"
fi

if [[ ${GPU_FREQS+x} ]]; then
  GPU_FREQS="${GPU_FREQS}"
else
  GPU_FREQS="504MHz 702MHz 900MHz 1.107GHz 1.305GHz 1.503GHz"
fi

if [[ ${EMC_FREQS+x} ]]; then
  EMC_FREQS="${EMC_FREQS}"
else
  EMC_FREQS="665.6MHz 2.75GHz 3.2GHz 4.266GHz"
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

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] INFERENCE_MODE=${INFERENCE_MODE}"
if [[ "${INFERENCE_MODE}" == "tensorrt" ]]; then
  echo "[INFO] TRT_ENGINE_PATH=${TRT_ENGINE_PATH}"
fi
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] ENV_NAME=${ENV_NAME}"
echo "[INFO] PC_CLIENT_SSH=${PC_CLIENT_SSH}"
echo "[INFO] PC_REPO_DIR=${PC_REPO_DIR}"
echo "[INFO] PC_PYTHON=${PC_PYTHON}"
echo "[INFO] THOR_IP=${THOR_IP}"
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
    cd "'"${REPO_ROOT}"'"
    export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"

    python -u gr00t/eval/run_gr00t_server.py \
      --model-path "'"${MODEL_PATH}"'" \
      --embodiment-tag "'"${EMBODIMENT_TAG}"'" \
      --inference-mode "'"${INFERENCE_MODE}"'" \
      '"${SERVER_TRT_PATH_ARG}"' \
      --host 0.0.0.0 \
      --port "'"${POLICY_CLIENT_PORT}"'" \
      --use-sim-policy-wrapper &
    server_pid=$!

    cleanup() {
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    echo "[INFO] Waiting for server port '"${POLICY_CLIENT_PORT}"'..."
    python - <<'"'"'PY'"'"'
import os
import socket
import sys
import time

host = "127.0.0.1"
port = int(os.environ.get("POLICY_CLIENT_PORT", "5555"))
deadline = time.time() + float(os.environ.get("WAIT_TIMEOUT_S", "180"))
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, port)) == 0:
            print(f"[INFO] Server is reachable on {host}:{port}")
            sys.exit(0)
    time.sleep(1.0)
print(f"[ERROR] Timed out waiting for server port {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

    echo "[INFO] Running remote RoboCasa client..."
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "'"${PC_CLIENT_SSH}"'" \
      "cd '"${PC_REPO_DIR}"' && '"${PC_PYTHON}"' gr00t/eval/rollout_policy.py \
        --n_episodes '"${N_EPISODES}"' \
        --policy_client_host '"${THOR_IP}"' \
        --policy_client_port '"${POLICY_CLIENT_PORT}"' \
        --max_episode_steps '"${MAX_EPISODE_STEPS}"' \
        --env_name '"${ENV_NAME}"' \
        --n_action_steps '"${N_ACTION_STEPS}"' \
        --n_envs '"${N_ENVS}"' \
        ${CLIENT_EXTRA_ARGS:-}"

    echo "[INFO] Remote RoboCasa client finished."
  '
