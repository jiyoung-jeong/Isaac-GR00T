#!/usr/bin/env bash
# thor_server_measure_and_analyze.sh
# Thor 호스트 또는 Docker에서 실행.
# 1) idle 포함 측정 시작 -> 2) GR00T server 실행 -> 3) (사용자가 PC에서 client 실행) -> 4) 엔터로 BENCH_START/END 마커 ->
# 5) server/측정 종료 -> 6) 분석+그래프 생성 (NVTX 구간 있으면 inference별 에너지 적분)
# 결과는 기본 /tmp. 워크스페이스에 남기려면: THOR_OUTDIR_ROOT=/workspace/.../thor_measurements

set -euo pipefail

###############################################################################
# 사용자 설정 (필요하면 여기만 수정)
###############################################################################
IDLE_SEC=8               # benchmark 시작 전 idle 측정(초)
INTERVAL_MS=5            # 로깅 주기(ms). 실제 간격 = sysfs 읽기 + sleep (더 짧게 하려면 4 또는 2로 설정)
SERVER_WARMUP_SEC=2      # server 뜬 후 안정화(초)

# Thor 호스트: /home/Thor/... / Docker: old와 동일하게 /workspace/Workspace/jyjeong/Isaac-GR00T 먼저 시도, 없으면 /workspace
THOR_GR00T_DIR="${THOR_GR00T_DIR:-/workspace/Workspace/jyjeong/Isaac-GR00T}"
if [[ ! -d "${THOR_GR00T_DIR}" ]] && [[ -d /workspace ]]; then
  THOR_GR00T_DIR="/workspace"
fi
echo "[INFO] GR00T dir: ${THOR_GR00T_DIR}"

# denoising_steps: 1=빠름(~110ms/infer), 4=문서/nsys와 비슷(~160ms). nsys 결과와 비교하려면 4 권장.
DENOISING_STEPS="${DENOISING_STEPS:-4}"

###############################################################################
# CLI 인자 파싱 (--tensorRT 지원)
###############################################################################
USE_TENSORRT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensorRT|--tensorrt|--tensor-rt)
      USE_TENSORRT=1
      shift
      ;;
    *)
      echo "[WARN] Unknown argument: $1" >&2
      shift
      ;;
  esac
done

SERVER_CMD=(python3 scripts/inference_service.py --server \
  --host "*" --port 5555 \
  --denoising_steps "${DENOISING_STEPS}" \
  --model_path youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain \
  --data_config fourier_gr1_arms_waist)

if [[ "${USE_TENSORRT}" -eq 1 ]]; then
  SERVER_CMD+=(--use-tensorrt --trt-engine-path gr00t_engine)
  echo "[INFO] TensorRT mode: using --use-tensorrt --trt-engine-path gr00t_engine"
else
  echo "[INFO] PyTorch mode: TensorRT disabled (no --use-tensorrt)"
fi

###############################################################################
# sysfs 경로 (Thor)
###############################################################################
# INA3221 (3채널, 2-0040): hwmonX 번호는 부팅/드라이버 순서에 따라 바뀔 수 있으므로 자동 탐색
H3221_BASE="/sys/bus/i2c/devices/2-0040/hwmon"
H3221=""
if [[ -d "${H3221_BASE}" ]]; then
  for d in "${H3221_BASE}"/hwmon*; do
    if [[ -d "${d}" ]]; then
      H3221="${d}"
      break
    fi
  done
fi
if [[ -z "${H3221}" ]]; then
  echo "[WARN] INA3221 hwmon dir not found under ${H3221_BASE} (2-0040). GPU/CPU power will be 0." >&2
fi

# 채널 매핑: label 기준으로 보면 1=GPU, 2=CPU_SOC_MSS, 3=SYS_5V
GPU_CH=1
CPU_CH=2

if [[ -n "${H3221}" ]]; then
  GPU_V="${H3221}/in${GPU_CH}_input"      # mV
  GPU_I="${H3221}/curr${GPU_CH}_input"    # mA
  CPU_V="${H3221}/in${CPU_CH}_input"      # mV
  CPU_I="${H3221}/curr${CPU_CH}_input"    # mA
else
  GPU_V="/dev/null"
  GPU_I="/dev/null"
  CPU_V="/dev/null"
  CPU_I="/dev/null"
fi

# INA238 (VIN 전체 전력, 2-0044): hwmonX 자동 탐색
H238_BASE="/sys/bus/i2c/devices/2-0044/hwmon"
H238=""
if [[ -d "${H238_BASE}" ]]; then
  for d in "${H238_BASE}"/hwmon*; do
    if [[ -d "${d}" ]]; then
      H238="${d}"
      break
    fi
  done
fi
if [[ -z "${H238}" ]]; then
  echo "[WARN] INA238 hwmon dir not found under ${H238_BASE} (2-0044). VIN power will be 0." >&2
  VIN_PWR_UW="/dev/null"
else
  VIN_PWR_UW="${H238}/power1_input"       # 보통 uW
fi

# GPU freq (devfreq + bpmp debug clk)
GPUF="/sys/class/devfreq/gpu-gpc-0/cur_freq"   # Hz (Thor: gpu-gpc-0)
GPU_GPC0_RATE="/sys/kernel/debug/bpmp/debug/clk/gpu_gpc0/rate"
GPU_GPC1_RATE="/sys/kernel/debug/bpmp/debug/clk/gpu_gpc1/rate"
GPU_GPC2_RATE="/sys/kernel/debug/bpmp/debug/clk/gpu_gpc2/rate"
GPU_SYS_RATE="/sys/kernel/debug/bpmp/debug/clk/gpu_sys/rate"
GPU_NVD_RATE="/sys/kernel/debug/bpmp/debug/clk/gpu_nvd/rate"

# CPU freq (policy0는 거의 항상 존재)
CPU0="/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq"  # kHz
CPU4="/sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq"  # 있으면 같이

# EMC (메모리 컨트롤러) 클럭, Hz
EMC_RATE="/sys/kernel/debug/bpmp/debug/clk/emc/rate"

###############################################################################
# 출력 파일 (기본 /tmp. 워크스페이스에 남기려면: THOR_OUTDIR_ROOT=/workspace/.../thor_measurements)
###############################################################################
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR_ROOT="${THOR_OUTDIR_ROOT:-/tmp}"
OUTDIR="${OUTDIR_ROOT}/thor_gr00t_server_${TS}"
mkdir -p "${OUTDIR}"

RAW_CSV="${OUTDIR}/telemetry_raw.csv"
MARK_CSV="${OUTDIR}/markers.csv"
SUMMARY_TXT="${OUTDIR}/summary.txt"
PWR_PNG="${OUTDIR}/power_plot.png"
FREQ_PNG="${OUTDIR}/freq_plot.png"
SERVER_LOG="${OUTDIR}/server.log"
# NVTX 구간 로그 (서버가 /tmp/nvtx_ranges.csv에 monotonic ns로 기록)
NVTX_CSV="${OUTDIR}/nvtx_ranges.csv"
NVTX_SRC="/tmp/nvtx_ranges.csv"
INFERENCE_ENERGY_CSV="${OUTDIR}/inference_energy.csv"

echo "[INFO] Output dir: ${OUTDIR}"

###############################################################################
# 유틸 (monotonic ns 사용)
###############################################################################
base_ns() { awk '{printf "%.0f\n", $1*1000000000}' /proc/uptime; }

mark() {
  local name="$1"
  local ts_ns
  ts_ns="$(base_ns)"
  echo "${ts_ns},${name}" >> "${MARK_CSV}"
  echo "[MARK] ${name} @ ${ts_ns}"
}

sleep_ms() {
  local ms="$1"
  # usleep 대신 sleep 사용
  sleep "$(awk -v ms="$ms" 'BEGIN{printf "%.3f", ms/1000}')"
}

start_logger() {
  echo "ts_ns,vdd_gpu_W,vdd_cpu_soc_mss_W,vin_W,gpu_freq_hz,cpu0_khz,cpu4_khz,gpu_gpc0_hz,gpu_gpc1_hz,gpu_gpc2_hz,gpu_sys_hz,gpu_nvd_hz,emc_rate_hz" > "${RAW_CSV}"
  echo "ts_ns,marker" > "${MARK_CSV}"

  TELEM_PY="${THOR_GR00T_DIR}/deployment_scripts/thor_telemetry_logger.py"
  if command -v python3 >/dev/null 2>&1 && [[ -f "${TELEM_PY}" ]]; then
    export THOR_RAW_CSV="${RAW_CSV}"
    export THOR_INTERVAL_MS="${INTERVAL_MS}"
    export THOR_GPU_V="${GPU_V}" THOR_GPU_I="${GPU_I}"
    export THOR_CPU_V="${CPU_V}" THOR_CPU_I="${CPU_I}"
    export THOR_VIN_PWR_UW="${VIN_PWR_UW}"
    export THOR_GPUF="${GPUF}" THOR_CPU0="${CPU0}" THOR_CPU4="${CPU4}"
    export THOR_GPU_GPC0_RATE="${GPU_GPC0_RATE}" THOR_GPU_GPC1_RATE="${GPU_GPC1_RATE}" THOR_GPU_GPC2_RATE="${GPU_GPC2_RATE}"
    export THOR_GPU_SYS_RATE="${GPU_SYS_RATE}" THOR_GPU_NVD_RATE="${GPU_NVD_RATE}"
    export THOR_EMC_RATE="${EMC_RATE}"
    python3 "${TELEM_PY}" &
    LOGGER_PID=$!
    echo "[INFO] Logger (Python, ~${INTERVAL_MS}ms) PID=${LOGGER_PID}"
  else
    (
      while true; do
        ts_ns="$(base_ns)"
        vg="$([ -r "${GPU_V}" ] && cat "${GPU_V}" || echo 0)"
        ig="$([ -r "${GPU_I}" ] && cat "${GPU_I}" || echo 0)"
        vc="$([ -r "${CPU_V}" ] && cat "${CPU_V}" || echo 0)"
        ic="$([ -r "${CPU_I}" ] && cat "${CPU_I}" || echo 0)"
        vin_uw="$([ -r "${VIN_PWR_UW}" ] && cat "${VIN_PWR_UW}" || echo 0)"
        gf="$(cat "${GPUF}" 2>/dev/null || echo -1)"
        c0="$(cat "${CPU0}" 2>/dev/null || echo -1)"
        c4="$(cat "${CPU4}" 2>/dev/null || echo -1)"
        gpc0="$(cat "${GPU_GPC0_RATE}" 2>/dev/null || echo -1)"
        gpc1="$(cat "${GPU_GPC1_RATE}" 2>/dev/null || echo -1)"
        gpc2="$(cat "${GPU_GPC2_RATE}" 2>/dev/null || echo -1)"
        gsys="$(cat "${GPU_SYS_RATE}" 2>/dev/null || echo -1)"
        gnvd="$(cat "${GPU_NVD_RATE}" 2>/dev/null || echo -1)"
        emc="$(cat "${EMC_RATE}" 2>/dev/null || echo -1)"
        pg="$(awk -v v="$vg" -v i="$ig" 'BEGIN{printf "%.6f", (v*i)/1e6}')"
        pc="$(awk -v v="$vc" -v i="$ic" 'BEGIN{printf "%.6f", (v*i)/1e6}')"
        vin_w="$(awk -v p="$vin_uw" 'BEGIN{printf "%.6f", p/1e6}')"
        echo "${ts_ns},${pg},${pc},${vin_w},${gf},${c0},${c4},${gpc0},${gpc1},${gpc2},${gsys},${gnvd},${emc}" >> "${RAW_CSV}"
        sleep_ms "${INTERVAL_MS}"
      done
    ) &
    LOGGER_PID=$!
    echo "[INFO] Logger (bash, ~${INTERVAL_MS}ms) PID=${LOGGER_PID}"
  fi
}

stop_logger() {
  if [[ -n "${LOGGER_PID:-}" ]] && kill -0 "${LOGGER_PID}" 2>/dev/null; then
    kill "${LOGGER_PID}" || true
    wait "${LOGGER_PID}" 2>/dev/null || true
    echo "[INFO] Logger stopped."
  fi
}

start_server() {
  pushd "${THOR_GR00T_DIR}" >/dev/null
  echo "[INFO] Starting server: ${SERVER_CMD[*]}"
  echo "[INFO] Server log: ${SERVER_LOG} (client 무한대기 시 여기서 에러 확인)"
  "${SERVER_CMD[@]}" > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  popd >/dev/null
  echo "[INFO] Server PID=${SERVER_PID} (준비될 때까지 30초~수 분 걸릴 수 있음)"
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping server..."
    kill "${SERVER_PID}" || true
    sleep 1
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "[INFO] Server stopped."
  fi
}

analyze_and_plot() {
  echo "[INFO] Analyzing and plotting..."
  python3 - <<PY
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

outdir = Path("${OUTDIR}")
df = pd.read_csv(outdir/"telemetry_raw.csv")
mk = pd.read_csv(outdir/"markers.csv")

t0 = df["ts_ns"].iloc[0]
df["t_s"] = (df["ts_ns"] - t0) * 1e-9
mk["t_s"] = (mk["ts_ns"] - t0) * 1e-9

markers = {r["marker"]: r["t_s"] for _, r in mk.iterrows()}
bs = markers.get("BENCH_START")
be = markers.get("BENCH_END")

def stats(name, tmin=None, tmax=None):
    x = df
    if tmin is not None:
        x = x[x["t_s"] >= tmin]
    if tmax is not None:
        x = x[x["t_s"] <= tmax]
    if len(x) == 0:
        return f"{name}: no samples\n"
    return (
        f"{name} (N={len(x)}):\n"
        f"  VDD_GPU_W mean={x.vdd_gpu_W.mean():.3f}, p95={x.vdd_gpu_W.quantile(0.95):.3f}, max={x.vdd_gpu_W.max():.3f}\n"
        f"  VDD_CPU_SOC_MSS_W mean={x.vdd_cpu_soc_mss_W.mean():.3f}, p95={x.vdd_cpu_soc_mss_W.quantile(0.95):.3f}, max={x.vdd_cpu_soc_mss_W.max():.3f}\n"
        f"  VIN_W mean={x.vin_W.mean():.3f}, p95={x.vin_W.quantile(0.95):.3f}, max={x.vin_W.max():.3f}\n"
        f"  GPU_freq_GHz mean={(x.gpu_freq_hz/1e9).mean():.3f}, max={(x.gpu_freq_hz/1e9).max():.3f}\n"
        f"  CPU0_GHz mean={(x.cpu0_khz/1e6).mean():.3f}, max={(x.cpu0_khz/1e6).max():.3f}\n"
    )

txt = []
txt.append(stats("ALL"))
if bs is not None:
    txt.append(stats("IDLE (before BENCH_START)", tmax=bs))
else:
    txt.append(stats("IDLE (first 5s fallback)", tmax=5.0))
if bs is not None and be is not None:
    txt.append(stats("BENCH (BENCH_START~BENCH_END)", tmin=bs, tmax=be))

# NVTX 구간이 있으면 inference별 에너지 적분 (E = sum(P*dt))
nvtx_path = outdir / "nvtx_ranges.csv"
if nvtx_path.exists():
    nv = pd.read_csv(nvtx_path, header=None, names=["ts_ns", "event"])
    nv = nv[nv["event"].isin(["POLICY_INFER_START", "POLICY_INFER_END"])].sort_values("ts_ns")
    starts = nv[nv["event"] == "POLICY_INFER_START"]["ts_ns"].values
    ends = nv[nv["event"] == "POLICY_INFER_END"]["ts_ns"].values
    if len(starts) == len(ends) and len(starts) > 0:
        energies = []
        for i, (s_ns, e_ns) in enumerate(zip(starts, ends)):
            seg = df[(df["ts_ns"] >= s_ns) & (df["ts_ns"] <= e_ns)].sort_values("ts_ns")
            if len(seg) < 2:
                continue
            ts = seg["ts_ns"].values.astype(np.float64)
            dt = np.diff(ts) * 1e-9  # seconds
            E_gpu = np.sum(seg["vdd_gpu_W"].values[:-1] * dt)
            E_cpu = np.sum(seg["vdd_cpu_soc_mss_W"].values[:-1] * dt)
            E_vin = np.sum(seg["vin_W"].values[:-1] * dt)
            dur_ms = (e_ns - s_ns) * 1e-6
            energies.append({"inference_id": i + 1, "duration_ms": dur_ms, "E_gpu_J": E_gpu, "E_cpu_J": E_cpu, "E_vin_J": E_vin})
        if energies:
            en_df = pd.DataFrame(energies)
            en_df.to_csv(outdir / "inference_energy.csv", index=False)
            txt.append("INFERENCE (per-call energy from NVTX ranges):")
            txt.append(f"  N={len(energies)} calls, E_gpu_J sum={en_df['E_gpu_J'].sum():.3f}, mean={en_df['E_gpu_J'].mean():.3f}")
            txt.append(f"  E_vin_J sum={en_df['E_vin_J'].sum():.3f}, mean={en_df['E_vin_J'].mean():.3f}")
            txt.append(f"  Duration from CSV ts_ns (START~END): min={en_df['duration_ms'].min():.1f} ms, max={en_df['duration_ms'].max():.1f} ms.")
            import re
            nv_full = pd.read_csv(nvtx_path, header=None, names=["ts_ns", "event"])
            dur_events = nv_full[nv_full["event"].astype(str).str.match(r"^INFER_DURATION_MS_", na=False)]
            if len(dur_events) > 0:
                server_dur = [float(re.search(r"INFER_DURATION_MS_([\d.]+)", str(e)).group(1)) for e in dur_events["event"]]
                txt.append(f"  Server-reported duration (same process): min={min(server_dur):.1f} ms, max={max(server_dur):.1f} ms.")
            txt.append("  Note: 1st inference often 5-10x longer (CUDA/JIT warmup). If nsys shows ~169ms but above is ~70ms, see docs/nvtx_power_integration.md §5.")

(outdir/"summary.txt").write_text("\n".join(txt), encoding="utf-8")

# Power plot
plt.figure(figsize=(14,5))
plt.plot(df.t_s, df.vin_W, label="VIN_W")
plt.plot(df.t_s, df.vdd_gpu_W, label="VDD_GPU_W")
plt.plot(df.t_s, df.vdd_cpu_soc_mss_W, label="VDD_CPU_SOC_MSS_W")
if bs is not None: plt.axvline(bs, linestyle="--", label="BENCH_START")
if be is not None: plt.axvline(be, linestyle="--", label="BENCH_END")
plt.xlabel("Time (s, relative)")
plt.ylabel("Power (W)")
plt.title("Power vs Time (Thor)")
plt.legend()
plt.tight_layout()
plt.savefig(outdir/"power_plot.png", dpi=160)

# Freq plot (GPU -1은 NaN 제외; GPU를 나중에 그려서 주황/초록 위로 보이게)
plt.figure(figsize=(14,5))
plt.plot(df.t_s, df.cpu0_khz/1e6, label="CPU0 freq (GHz)", color="C1", linewidth=1, zorder=1)
if (df.cpu4_khz >= 0).any():
    plt.plot(df.t_s, df.cpu4_khz/1e6, label="CPU4 freq (GHz)", color="C2", linewidth=1, zorder=1)
gpu_ghz = np.where(df.gpu_freq_hz > 0, df.gpu_freq_hz / 1e9, np.nan)
if np.any(np.isfinite(gpu_ghz)):
    plt.plot(df.t_s, gpu_ghz, label="GPU freq (GHz)", color="C0", linewidth=2.5, zorder=2)
if "emc_rate_hz" in df.columns and (df.emc_rate_hz >= 0).any():
    plt.plot(df.t_s, df.emc_rate_hz / 1e9, label="EMC freq (GHz)", color="C3", linewidth=1, zorder=1)
if bs is not None: plt.axvline(bs, linestyle="--", label="BENCH_START")
if be is not None: plt.axvline(be, linestyle="--", label="BENCH_END")
plt.xlabel("Time (s, relative)")
plt.ylabel("Frequency (GHz)")
plt.title("Frequencies vs Time (Thor)")
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig(outdir/"freq_plot.png", dpi=160)

# inference_energy.csv 막대 그래프 (inference별 E_gpu_J, E_vin_J, duration_ms)
en_path = outdir / "inference_energy.csv"
if en_path.exists():
    en_df = pd.read_csv(en_path)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    x = en_df["inference_id"]
    axes[0].bar(x, en_df["E_gpu_J"], color="C1", label="E_gpu (J)")
    axes[0].set_ylabel("E_gpu (J)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(x, en_df["E_vin_J"], color="C0", label="E_vin (J)")
    axes[1].set_ylabel("E_vin (J)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[2].bar(x, en_df["duration_ms"], color="C2", label="duration (ms)")
    axes[2].set_ylabel("duration (ms)")
    axes[2].set_xlabel("inference_id")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    plt.suptitle("Inference energy (from NVTX ranges)")
    plt.tight_layout()
    plt.savefig(outdir / "inference_energy_bars.png", dpi=160)
    plt.close()

# nvtx_ranges.csv 타임라인 (시간 ms, 구간별 ViT/LLM/Action 등)
nv_path = outdir / "nvtx_ranges.csv"
if nv_path.exists():
    nv = pd.read_csv(nv_path, header=None, names=["ts_ns", "event"])
    nv = nv.sort_values("ts_ns")
    t0_ns = nv["ts_ns"].iloc[0]
    nv["t_ms"] = (nv["ts_ns"] - t0_ns) * 1e-6
    # START/END 페어로 구간 만들기 (이벤트명에서 _START/_END 제거해 라벨)
    starts = nv[nv["event"].str.endswith("_START")].copy()
    ends = nv[nv["event"].str.endswith("_END")].copy()
    starts["label"] = starts["event"].str.replace("_START", "")
    ends["label"] = ends["event"].str.replace("_END", "")
    # 라벨별로 매칭 (순서대로 같은 라벨이면 페어)
    segs = []
    for _, r in starts.iterrows():
        lab = r["label"]
        t_s = r["t_ms"]
        match = ends[(ends["label"] == lab) & (ends["t_ms"] > t_s)]
        if len(match) > 0:
            t_e = match["t_ms"].iloc[0]
            segs.append({"label": lab, "t_start": t_s, "t_end": t_e, "dur_ms": t_e - t_s})
    if segs:
        seg_df = pd.DataFrame(segs)
        order = ["POLICY_INFER", "BACKBONE", "Backbone_ViT", "Backbone_Projector", "Backbone_LLM", "ACTION_HEAD"]
        uniq = [l for l in order if l in seg_df["label"].values] + [l for l in seg_df["label"].unique() if l not in order]
        colors = {l: f"C{i % 10}" for i, l in enumerate(uniq)}
        fig, ax = plt.subplots(figsize=(14, max(5, len(uniq) * 0.8)))
        yh = 0.7
        for yi, lab in enumerate(uniq):
            s = seg_df[seg_df["label"] == lab]
            ranges = [(r["t_start"], r["dur_ms"]) for _, r in s.iterrows()]
            if ranges:
                ax.broken_barh(ranges, (yi - yh/2, yh), facecolors=colors.get(lab, "gray"), alpha=0.8, label=lab)
        ax.set_yticks(range(len(uniq)))
        ax.set_yticklabels(uniq)
        ax.set_xlabel("Time (ms from first NVTX event)")
        ax.set_ylabel("Phase")
        ax.set_title("NVTX ranges timeline (ms) — ViT / LLM / Action")
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(outdir / "nvtx_timeline_ms.png", dpi=160)
        plt.close()
    # inference별 그래프: GPU/CPU/VIN 전력 + ViT·LLM·Action 구간 (전력은 5ms 보간으로 부드럽게)
    pol_starts = nv[nv["event"] == "POLICY_INFER_START"].sort_values("ts_ns")["ts_ns"].values
    pol_ends = nv[nv["event"] == "POLICY_INFER_END"].sort_values("ts_ns")["ts_ns"].values
    phase_display = {"Backbone_ViT": "ViT", "Backbone_LLM": "LLM", "Backbone_Projector": "Projector", "ACTION_HEAD": "Action"}
    phase_colors = {"Backbone_ViT": "green", "Backbone_LLM": "blue", "Backbone_Projector": "gray", "ACTION_HEAD": "orange"}
    if len(pol_starts) == len(pol_ends) and len(pol_starts) > 0:
        inf_dir = outdir / "inference"
        inf_dir.mkdir(exist_ok=True)
        for idx in range(len(pol_starts)):
            s_ns, e_ns = pol_starts[idx], pol_ends[idx]
            dur_ms = (e_ns - s_ns) * 1e-6
            margin_ns = int(50e6)
            seg = df[(df["ts_ns"] >= s_ns - margin_ns) & (df["ts_ns"] <= e_ns + margin_ns)].sort_values("ts_ns")
            if len(seg) < 2:
                continue
            seg = seg.copy()
            seg["t_ms"] = (seg["ts_ns"] - s_ns) * 1e-6
            t_ms_grid = np.arange(0, dur_ms + 1e-6, 5.0)
            t_ms_grid = t_ms_grid[t_ms_grid <= dur_ms]
            if len(t_ms_grid) < 2:
                t_ms_grid = np.linspace(0, dur_ms, max(10, int(dur_ms)))
            p_gpu = np.interp(t_ms_grid, seg["t_ms"].values, seg["vdd_gpu_W"].values)
            p_cpu = np.interp(t_ms_grid, seg["t_ms"].values, seg["vdd_cpu_soc_mss_W"].values)
            p_vin = np.interp(t_ms_grid, seg["t_ms"].values, seg["vin_W"].values)
            nv_win = nv[(nv["ts_ns"] >= s_ns) & (nv["ts_ns"] <= e_ns)].copy()
            nv_win["t_ms"] = (nv_win["ts_ns"] - s_ns) * 1e-6
            phases = []
            for lab in ["Backbone_ViT", "Backbone_LLM", "Backbone_Projector", "ACTION_HEAD"]:
                st = nv_win[nv_win["event"] == f"{lab}_START"]
                en = nv_win[nv_win["event"] == f"{lab}_END"]
                for _, r in st.iterrows():
                    t_s = r["t_ms"]
                    match = en[en["t_ms"] > t_s]
                    if len(match) > 0:
                        t_e = match["t_ms"].iloc[0]
                        phases.append((lab, t_s, t_e))
            fig, ax = plt.subplots(figsize=(12, 5))
            for (lab, t_s, t_e) in phases:
                ax.axvspan(t_s, t_e, alpha=0.25, color=phase_colors.get(lab, "gray"))
            ax.plot(t_ms_grid, p_gpu, label="GPU (W)", color="C1", linewidth=1.2)
            ax.plot(t_ms_grid, p_cpu, label="CPU (W)", color="C2", linewidth=1.2)
            ax.plot(t_ms_grid, p_vin, label="VIN (W)", color="C0", linewidth=1.2)
            from matplotlib.patches import Patch
            phase_handles = [Patch(facecolor=phase_colors.get(lab, "gray"), alpha=0.25, label=phase_display.get(lab, lab)) for lab in ["Backbone_ViT", "Backbone_LLM", "Backbone_Projector", "ACTION_HEAD"] if any(p[0] == lab for p in phases)]
            leg_handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=phase_handles + leg_handles, loc="upper right")
            ax.set_xlabel("Time (ms from inference start)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"Inference {idx+1} — ViT / LLM / Action & GPU/CPU/VIN power")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(inf_dir / f"inference_{idx+1:02d}.png", dpi=160)
            plt.close()
PY
  echo "[INFO] Wrote: ${SUMMARY_TXT}"
  echo "[INFO] Wrote: ${PWR_PNG}"
  echo "[INFO] Wrote: ${FREQ_PNG}"
  [[ -f "${OUTDIR}/inference_energy_bars.png" ]] && echo "[INFO] Wrote: ${OUTDIR}/inference_energy_bars.png"
  [[ -f "${OUTDIR}/nvtx_timeline_ms.png" ]] && echo "[INFO] Wrote: ${OUTDIR}/nvtx_timeline_ms.png"
  [[ -d "${OUTDIR}/inference" ]] && echo "[INFO] Wrote: ${OUTDIR}/inference/ (per-inference power + ViT/LLM/Action plots)"
}

cleanup() {
  stop_logger
  stop_server
}
trap cleanup EXIT

###############################################################################
# 메인 실행
###############################################################################
echo "[INFO] INA3221 labels (for record):" | tee "${OUTDIR}/labels.txt"
{
  if [[ -n "${H3221}" ]]; then
    echo "H3221 dir: ${H3221}"
    echo "in1_label: $(cat ${H3221}/in1_label 2>/dev/null || echo)"
    echo "in2_label: $(cat ${H3221}/in2_label 2>/dev/null || echo)"
    echo "in3_label: $(cat ${H3221}/in3_label 2>/dev/null || echo)"
  else
    echo "H3221 dir: (not found)"
  fi
} | tee -a "${OUTDIR}/labels.txt"

start_logger
mark "MEASURE_START"

echo "[INFO] Idle logging for ${IDLE_SEC}s..."
sleep "${IDLE_SEC}"

# NVTX range 로그 초기화 (서버가 POLICY_INFER_START/END 기록)
rm -f "${NVTX_SRC}" 2>/dev/null || true

start_server
sleep "${SERVER_WARMUP_SEC}"

echo
echo "============================================================"
echo "이제 PC에서 client benchmark를 실행할 준비를 해."
echo "준비되면 아래 프롬프트에서 ENTER -> BENCH_START 마커가 찍힘."
echo "benchmark가 끝나면 다시 ENTER -> BENCH_END 마커가 찍힘."
echo "============================================================"
echo

read -r -p "[ACTION] Press ENTER to mark BENCH_START (you are about to run client on PC)..." _
mark "BENCH_START"

read -r -p "[ACTION] Press ENTER to mark BENCH_END (after client benchmark finishes)..." _
mark "BENCH_END"

# 종료
stop_server
stop_logger

# NVTX range 로그 수집 (inference별 에너지 적분용)
if [[ -f "${NVTX_SRC}" ]]; then
  cp "${NVTX_SRC}" "${NVTX_CSV}"
  echo "[INFO] NVTX range log copied: ${NVTX_CSV}"
else
  echo "[WARN] NVTX range log not found: ${NVTX_SRC}"
fi

analyze_and_plot

# nsys-rep가 있으면 NVTX 구간을 nsys에서 파싱해 telemetry와 매칭 → inference_energy_nsys.csv 생성
if [[ -n "${THOR_NSYS_REP:-}" ]] && [[ -f "${THOR_NSYS_REP}" ]]; then
  echo "[INFO] Merging nsys NVTX with telemetry: ${THOR_NSYS_REP}"
  if (cd "${THOR_GR00T_DIR}" && python3 deployment_scripts/nsys_telemetry_merge.py --run-dir "${OUTDIR}" --nsys-rep "${THOR_NSYS_REP}"); then
    echo "[INFO] inference_energy_nsys.csv written (infer_ms/energy from nsys-rep + telemetry)"
  else
    echo "[WARN] nsys_telemetry_merge.py failed (need nvtx_ranges.csv from same run for alignment)"
  fi
fi

echo
echo "[DONE] Output dir : ${OUTDIR}"
echo "[DONE] Raw CSV    : ${RAW_CSV}"
echo "[DONE] Markers    : ${MARK_CSV}"
echo "[DONE] Summary    : ${SUMMARY_TXT}"
echo "[DONE] Power plot : ${PWR_PNG}"
echo "[DONE] Freq plot  : ${FREQ_PNG}"
echo "[DONE] NVTX ranges: ${NVTX_CSV}"
echo "[DONE] Inference energy CSV : ${OUTDIR}/inference_energy.csv"
echo "[DONE] Inference energy bars: ${OUTDIR}/inference_energy_bars.png"
echo "[DONE] NVTX timeline (ms)   : ${OUTDIR}/nvtx_timeline_ms.png"
echo "[DONE] Inference folder      : ${OUTDIR}/inference/"
echo "[DONE] Inference energy (NVTX): ${INFERENCE_ENERGY_CSV}"
[[ -f "${OUTDIR}/inference_energy_nsys.csv" ]] && echo "[DONE] Inference energy (nsys NVTX): ${OUTDIR}/inference_energy_nsys.csv"
