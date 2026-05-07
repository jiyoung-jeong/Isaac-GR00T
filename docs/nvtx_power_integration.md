# A안: NVTX 구간 + monotonic ns 로그 → inference별 에너지 적분

NVTX는 프로파일러용이라 bash 로거가 직접 읽을 수 없으므로, **같은 구간의 monotonic ns 타임스탬프를 파일로 남기고** power 로그와 맞춰 에너지를 적분합니다.

---

## 1) inference_service.py — NVTX 구간에 monotonic ns 로그 추가

**패치 적용:**
```bash
cd /home/Thor/Workspace/jyjeong/Isaac-GR00T
patch -p1 < scripts/inference_service_nvtx_log.patch
```

또는 수동으로:

- **import 추가:** `import os`, `from pathlib import Path`
- **상수·함수 추가 (tyro import 아래):**
```python
# NVTX 구간을 power 로거와 맞추기 위한 monotonic ns 로그 (A안)
NVTX_LOG = Path(os.environ.get("NVTX_RANGES_CSV", "/tmp/nvtx_ranges.csv"))

def _log_nvtx_range(event: str) -> None:
    try:
        ts_ns = time.monotonic_ns()
        with open(NVTX_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts_ns},{event}\n")
    except OSError:
        pass
```

- **profiled_inference 래퍼:** `nvtx.range_push` 직후 `_log_nvtx_range("POLICY_INFER_START")`, `return original_method(...)` 직전은 그대로, `finally` 안에서 `_log_nvtx_range("POLICY_INFER_END")` 추가 후 `nvtx.range_pop()`.

`time.monotonic_ns()`와 Thor 측정 스크립트의 `base_ns()`(/proc/uptime)는 동일한 monotonic 클록이므로 power 로그와 정렬됩니다.

---

## 2) thor_server_measure_and_analyze.sh — NVTX 로그 수집

**출력 파일 섹션에 추가:**
```bash
NVTX_CSV="${OUTDIR}/nvtx_ranges.csv"
NVTX_SRC="/tmp/nvtx_ranges.csv"
```

**start_server 직전에 추가:**
```bash
# NVTX range 로그 초기화 (서버가 POLICY_INFER_START/END 기록)
rm -f "${NVTX_SRC}" 2>/dev/null || true
```

**stop_logger 직후, analyze_and_plot 직전에 추가:**
```bash
if [[ -f "${NVTX_SRC}" ]]; then
  cp "${NVTX_SRC}" "${NVTX_CSV}"
  echo "[INFO] NVTX range log copied: ${NVTX_CSV}"
else
  echo "[WARN] NVTX range log not found: ${NVTX_SRC}"
fi
```

---

## 3) analyze_and_plot() — inference별 에너지 적분 추가

기존 `txt.append(stats("BENCH (BENCH_START~BENCH_END)", tmin=bs, tmax=be))` **바로 다음**,  
`(outdir/"summary.txt").write_text(...)` **직전**에 아래 블록을 삽입합니다.

```python
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
```

- `nvtx_ranges.csv` 형식: 한 줄에 `ts_ns,event` (예: `1234567890000000,POLICY_INFER_START`).
- POLICY_INFER_START/END 페어만 사용해 구간을 만들고, 그 구간의 power 샘플로 `E = Σ P(t)·Δt` 적분.
- 결과는 `inference_energy.csv` (inference_id, duration_ms, E_gpu_J, E_cpu_J, E_vin_J)와 summary 텍스트에 추가됩니다.

위 1~3을 적용하면 NVTX로 표시된 inference 구간만 잘라서 에너지 적분할 수 있습니다.

---

## 4) 전력 그래프의 시간 보간 (interpolation)

inference별 그래프에서 **전력 곡선**은 telemetry 원본이 40ms 주기라서, 짧은 inference(예: 70ms)에서는 샘플이 2~3개뿐이라 끊겨 보입니다. 그래서:

- inference 구간 앞뒤로 **50ms 여유**를 두고 telemetry를 가져온 뒤
- **0 ~ duration_ms** 구간을 **5ms 간격**의 그리드(`t_ms_grid`)로 만들고
- `np.interp(t_ms_grid, seg["t_ms"], seg["vdd_gpu_W"])` 등으로 **선형 보간**해 곡선을 부드럽게 그립니다.

**duration 자체는 바꾸지 않습니다.** X축 길이는 여전히 `(POLICY_INFER_END_ts_ns - POLICY_INFER_START_ts_ns) * 1e-6` ms입니다. 보간은 “같은 구간 안에서 전력 값을 더 촘촘한 시간 격자에 맞춰 그리기”만 합니다.

---

## 5) nsys 169ms vs 우리 측정(한자리/두자리 ms) 차이

- **우리 측정**: 서버 프로세스에서 `time.monotonic_ns()`로 POLICY_INFER_START/END를 찍고, `duration_ms = (END_ns - START_ns) * 1e-6`으로 계산합니다. 즉 **Python 기준 wall time**(`model.get_action()` 진입 ~ 반환)입니다.
- **nsys/Nsight**: GPU 커널이 실제로 돌아간 시간을 측정합니다. 보통 PyTorch는 동기식이라 Python 반환 시점에 GPU 작업이 끝나 있어서, **Python 시간 ≥ GPU 시간**이어야 합니다.

그래서 우리가 70ms대인데 nsys가 169ms라면 가능한 원인은:

1. **서버와 telemetry의 시간 기준이 다름**  
   - telemetry는 셸 스크립트(호스트)의 `base_ns()`(/proc/uptime), NVTX CSV는 서버의 `time.monotonic_ns()`.  
   - 서버가 **Docker 안**에서 돌면, 컨테이너의 monotonic 클록이 호스트와 다르게 잡히는 경우가 있어, **같은 구간인데도** 우리가 쓰는 ts_ns 차이로 계산한 duration이 잘못 나올 수 있습니다.  
   - **검증**: 서버가 POLICY_INFER_END 찍을 때 **같은 프로세스 안에서** `(end_ns - start_ns) * 1e-6` ms를 한 번 더 로그해 두고(robot.py에 구현됨), 분석에서 “서버가 본 duration”과 “CSV ts_ns로 계산한 duration”이 맞는지 비교해 보세요.

2. **다른 런/설정**  
   - nsys 켠 런(169ms)과 전력 측정 런이 다르면(스텝 수, 배치, 모델 옵션 등) 당연히 다를 수 있습니다.

3. **비동기 실행**  
   - GPU 작업이 Python 반환 후에도 이어지면, nsys는 169ms를 보지만 Python 측 duration은 더 짧게 나올 수 있습니다. (일반적인 inference 코드에서는 드뭅니다.)

정리하면, **보간은 “그래프를 부드럽게 그리기 위한 표시용”**이고, **inference 길이는 NVTX의 START/END ts_ns 차이(그대로)**입니다. nsys와 다르게 나오면 위 1~3을 순서대로 점검하는 것이 좋습니다.
