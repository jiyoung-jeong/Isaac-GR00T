#!/usr/bin/env bash
set -euo pipefail

# Run this on Thor, inside the GR00T repo/container.
#
# Required environment:
#   PC_CLIENT_SSH  SSH target for the PC that runs rollout_policy.py, e.g. user@192.168.0.10
#   THOR_IP        IP address that the PC should use to reach this Thor server
#
# Common optional environment:
#   LATEST_CKPT    Defaults to newest outputs/libero_spatial_thor/checkpoint-*
#   PC_REPO_DIR    Defaults to ~/Workspace/jyjeong/Isaac-GR00T on the PC
#   OUT_DIR        Defaults to thor_measurements/gpu_sweep_vla_auto
#   FREQS          Space-separated list. Defaults to 504MHz..1.575GHz
#   INTERVAL_MS    Defaults to 2
#   N_EPISODES     Defaults to 10
#   WAIT_TIMEOUT_S Defaults to 180
#   LOCK_RETRIES   Defaults to 3
#   LOCK_VERIFY_S  Defaults to 0.5
#   INCLUDE_DEFAULT Set to 1 to run an unlocked/default governor baseline first

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

OUT_DIR="${OUT_DIR:-thor_measurements/gpu_sweep_vla_auto}"
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
FREQS="${FREQS:-504MHz 603MHz 702MHz 801MHz 900MHz 999MHz 1.107GHz 1.206GHz 1.305GHz 1.404GHz 1.503GHz 1.575GHz}"
INCLUDE_DEFAULT="${INCLUDE_DEFAULT:-0}"

freq_args=()
if [[ "${INCLUDE_DEFAULT}" == "1" ]]; then
  freq_args+=(--include-default)
fi
for freq in ${FREQS}; do
  freq_args+=(--freq "${freq}")
done

echo "[INFO] PC_CLIENT_SSH=${PC_CLIENT_SSH}"
echo "[INFO] THOR_IP=${THOR_IP}"
echo "[INFO] LATEST_CKPT=${LATEST_CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] INCLUDE_DEFAULT=${INCLUDE_DEFAULT}"
echo "[INFO] FREQS=${FREQS}"

python deployment_scripts/thor_gpufreq_power_sweep.py \
  --out-dir "${OUT_DIR}" \
  --interval-ms "${INTERVAL_MS}" \
  --settle-s "${SETTLE_S}" \
  --lock-retries "${LOCK_RETRIES}" \
  --lock-verify-s "${LOCK_VERIFY_S}" \
  --skip-failed-locks \
  "${freq_args[@]}" \
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
