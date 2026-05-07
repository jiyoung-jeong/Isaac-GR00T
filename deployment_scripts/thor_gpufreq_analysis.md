# Thor GPU 주파수별 측정 결과 해석

## 0. inference 시간: 우리 120ms vs nsys 160~170ms (denoising 4인데 차이가 나는 경우)

- **전제**: NVTX로 같은 구간(POLICY_INFER_START~END = Total_Policy_Inference)을 재므로, nsys와 우리 수치는 **같은 런이면 동일**해야 함. denoising 1단과 4단이 같은 시간으로 나오면 말이 안 되므로, **실제 런에서 쓰인 denoising_steps**를 확인해야 함.
- **실제 사용된 denoising_steps**: 서버가 첫 inference 직전에 NVTX CSV에 `CONFIG_DENOISING_STEPS_N`을 기록함. `compare_thor_runs.py` 출력의 **`de_steps`** 컬럼이 그 값 (`?`면 구버전 측정). 새로 측정한 런에서 `de_steps=4`인데도 120ms면 nsys에서 잡은 구간이 Total_Policy_Inference 한 개와 같은지 확인.
- **가능한 원인(차이 날 때)**
  1. **nsys에서 잡는 구간이 더 넓음**  
     우리는 `model.get_action(obs)` 호출 구간만 잡음 (robot.py에서 START 직전 ~ END 직후).  
     nsys에서 160~170ms로 본 구간이 **Total_Policy_Inference**만인지, **요청 수신~응답 전송** 전체인지 확인.  
     ZMQ 직렬화/전송 등이 포함되면 120ms(순수 추론) + 40~50ms(오버헤드) ≈ 160~170ms가 될 수 있음.
  2. **서로 다른 런 비교**  
     nsys 프로파일을 찍은 실행과 thor 측정(CSV) 런이 다르면 (TensorRT 유무, 전력/주파수, 부하 등) 구간 시간이 달라질 수 있음.
  3. **nsys가 보여주는 지표**  
     타임라인에서 “구간 길이”가 wall-clock인지, GPU 활동 구간만인지 등 확인.  
     우리 수치는 **같은 프로세스**의 `time.monotonic_ns()`로 잰 **get_action() wall-clock**임.
- **확인 권장**: nsys에서 160~170ms로 읽은 구간이 **정확히 NVTX "Total_Policy_Inference" 한 개**인지, 그 구간의 시작/끝이 Python의 POLICY_INFER_START/END와 같은지 보면 원인 좁히는 데 도움이 됨.

### nsys-rep + telemetry 파이프라인 (정확한 infer_ms / energy)

- **nsys_telemetry_merge.py**: nsys-rep에서 NVTX 구간(Total_Policy_Inference)을 파싱하고, **nvtx_ranges.csv**의 첫 inference START/END로 nsys 시간축을 telemetry(monotonic)에 정렬한 뒤, 구간별 **infer_ms**와 **E_gpu_J, E_vin_J** 등을 계산해 **inference_energy_nsys.csv**로 저장.
- **조건**: 같은 런에서 (1) telemetry 수집 (2) 서버가 nvtx_ranges.csv 기록 (3) nsys profile로 서버 실행해 .nsys-rep 확보.
- **사용**  
  - 측정 후 수동: `python3 deployment_scripts/nsys_telemetry_merge.py --run-dir <OUTDIR> --nsys-rep <path/to/report.nsys-rep>`  
  - 측정 스크립트와 연동: `THOR_NSYS_REP=/path/to/report.nsys-rep ./thor_server_measure_and_analyze.sh` 실행 시 분석 직후 자동으로 merge 실행 후 `inference_energy_nsys.csv` 생성.

---

## 1. 먼저 확인할 것: 실제 측정된 GPU 주파수

**설정한 cap과 실제 런타임 GPU 주파수가 다를 수 있습니다.**

- `telemetry_raw.csv`의 **BENCH 구간(BENCH_START~BENCH_END)** 안에서 `gpu_freq_hz` 평균/최대가 런마다 **실제로 다른지** 확인해 보세요.
- 비교 스크립트 사용:
  ```bash
  cd /workspace/Workspace/jyjeong/Isaac-GR00T
  python3 deployment_scripts/compare_thor_runs.py \
    thor_gr00t_server_20260304_065357_gpufreq_801MB \
    thor_gr00t_server_20260304_071027_gpufreq_1.305GHz \
    thor_gr00t_server_20260304_075559_gpufreq_1.575GHz \
    thor_gr00t_server_20260304_081336_gpufreq_free \
    thor_gr00t_server_20260304_105658_MAXN1
  ```
- **만약 BENCH 구간의 GPU freq가 모든 런에서 비슷하다면**  
  → 주파수 cap이 적용되지 않았거나, 부하 시 동일한 수준으로 부스트/쓰로틀링된 것입니다.  
  → 그 경우 power와 inference 시간이 비슷한 것이 자연스러운 결과입니다.

---

## 2. 주파수 cap이 제대로 적용됐는데도 비슷할 수 있는 이유

### 2.1 메모리 바운드 (Memory-bound)

- **GPU 연산량**보다 **메모리 대역폭**이 병목이면, 클럭을 올려도 연산은 빨리 끝나지만 메모리에서 데이터를 기다리는 시간이 그대로라 **전체 지연은 크게 줄지 않습니다.**
- GR00T 파이프라인(ViT → LLM → DiT) 중 상당 부분이 **attention 등 메모리 접근이 많은 연산**이라, Orin GPU에서 메모리 바운드에 가깝게 동작할 수 있습니다.
- 이 경우:
  - **주파수 ↑** → 소비 전력은 늘 수 있지만, **inference 시간은 거의 그대로**일 수 있음.
  - 또는 메모리/전력 제한으로 **실제 유효 주파수**가 비슷한 수준으로 수렴해, power도 비슷해 보일 수 있음.

### 2.2 CPU / 데이터 이동 병목

- 전처리, 후처리, Python ↔ CUDA 복사 등 **CPU 구간**이 길면, GPU만 빨라져도 **전체 end-to-end 시간**은 비슷하게 나올 수 있습니다.
- nsys에서 **GPU 커널 구간만** 보면 빨라졌는데, **전체 inference 시간**은 CPU·동기화 때문에 비슷할 수 있습니다.

### 2.3 Denoising step

- **thor_server_measure_and_analyze.sh**는 기본 **DENOISING_STEPS=4** 사용. inference_service를 단독 실행하면 기본값은 1이라 DiT 구간이 짧습니다.
- 대부분의 시간이 **ViT + LLM**에 쓰이므로, 이 부분이 메모리 바운드이면 GPU 클럭을 올려도 체감 지연·전력 차이가 작을 수 있습니다.

### 2.4 쓰로틀링 / 실제 유효 주파수

- **1.575GHz, free, MAXN1**처럼 높은 설정에서:
  - 짧은 구간만 높은 주파수로 동작하고, **열/전력 제한**으로 곧 낮은 주파수로 내려갈 수 있음.
  - 그러면 **평균 GPU freq**와 **평균 전력**이 중간 설정(1.305GHz 등)과 비슷해져, 런 간 차이가 작아 보일 수 있습니다.

### 2.5 전력 측정 구간과 샘플링

- 전력은 **5ms 간격**으로 샘플링됩니다. inference 한 번이 수십~백 ms 단위이므로, 구간 평균으로 보면 **짧은 주파수 변동**이 평균에 묻혀 런 간 차이가 작게 보일 수 있습니다.
- BENCH 구간이 짧거나 inference 횟수가 적으면, 평균 전력이 런마다 비슷해 보일 수 있습니다.

---

## 3. 정리 및 권장 확인 순서

| 단계 | 확인 내용 |
|------|-----------|
| 1 | `compare_thor_runs.py`로 **BENCH 구간 실제 gpu_freq_hz**가 런마다 다른지 확인 |
| 2 | 다르다면: 메모리 바운드/CPU 병목/쓰로틀링 등 위 요인 검토 |
| 3 | 같다면: 주파수 cap 적용 방법(nvpmodel, jetson_clocks, sysfs 등)과 측정 시점 재확인 |

- **주파수 cap 적용**: 측정 **직전**에 nvpmodel 또는 `echo ... > /sys/class/devfreq/gpu-gpc-0/max_freq` 등으로 설정했는지, 그리고 **측정 중** 다른 프로세스가 max_freq를 바꾸지 않았는지 확인하는 것이 좋습니다.
- nsys로 **GPU 커널 구간만** 구해 보면, 주파수에 따른 **GPU 시간** 차이가 있는지도 함께 보면 원인 파악에 도움이 됩니다.
