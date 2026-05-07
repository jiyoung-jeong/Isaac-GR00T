# Manual Frequency Runtime Switch Policy

`manual_freq_phase_metrics.csv` 기준으로 고른 backend별 프로파일:

- `tensorRT`
  - `busy`: `emcfreq=2.75GHz`, `gpufreq=1.575GHz`
  - `idle`: `emcfreq=3.2GHz`, `gpufreq=801MHz`
- `torchcompile`
  - `busy`: `emcfreq=4.26GHz`, `gpufreq=1.575GHz`
  - `idle`: `emcfreq=4.26GHz`, `gpufreq=801MHz`
- `pytorch`
  - `busy`: `emcfreq=4.26GHz`, `gpufreq=1.305GHz`
  - `idle`: `emcfreq=2.75GHz`, `gpufreq=801MHz`

## Runtime 제어 원칙

- **입력 신호**
  - 최근 `N`개 요청 기준 `p95 latency` (권장 N=30~100)
  - 현재 queue depth (`q`)
  - 선택: 최근 VIN power EMA (`vin_ema_w`)
- **히스테리시스**
  - `to_busy`: `p95 >= 1.03 * governor_latency_ms` 또는 `q >= Q_HIGH`
  - `to_idle`: `p95 <= 0.92 * governor_latency_ms` 그리고 `q <= Q_LOW`
  - 권장 `Q_HIGH=3`, `Q_LOW=1`
- **쿨다운**
  - 빈번한 토글 방지를 위해 마지막 전환 후 `COOLDOWN_SEC` 동안 재전환 금지 (권장 20~30초)

## Python 의사코드

```python
from dataclasses import dataclass
import time

@dataclass
class Profile:
    emcfreq: str
    gpufreq: str

PROFILES = {
    "tensorRT": {
        "busy": Profile("2.75GHz", "1.575GHz"),
        "idle": Profile("3.2GHz", "801MHz"),
        "gov_latency_ms": 73.92,
    },
    "torchcompile": {
        "busy": Profile("4.26GHz", "1.575GHz"),
        "idle": Profile("4.26GHz", "801MHz"),
        "gov_latency_ms": 118.92,
    },
    "pytorch": {
        "busy": Profile("4.26GHz", "1.305GHz"),
        "idle": Profile("2.75GHz", "801MHz"),
        "gov_latency_ms": 123.46,
    },
}

Q_HIGH = 3
Q_LOW = 1
COOLDOWN_SEC = 25

def apply_profile(profile: Profile):
    # TODO: 실제 주파수 적용 함수 연결
    # set_emcfreq(profile.emcfreq)
    # set_gpufreq(profile.gpufreq)
    pass

def choose_mode(cur_mode, p95_ms, queue_depth, gov_ms):
    to_busy = (p95_ms >= 1.03 * gov_ms) or (queue_depth >= Q_HIGH)
    to_idle = (p95_ms <= 0.92 * gov_ms) and (queue_depth <= Q_LOW)
    if cur_mode == "idle" and to_busy:
        return "busy"
    if cur_mode == "busy" and to_idle:
        return "idle"
    return cur_mode

def control_loop(backend, get_p95_ms, get_queue_depth):
    cfg = PROFILES[backend]
    mode = "idle"
    apply_profile(cfg[mode])
    last_switch_ts = time.time()

    while True:
        p95 = get_p95_ms()
        q = get_queue_depth()
        next_mode = choose_mode(mode, p95, q, cfg["gov_latency_ms"])

        now = time.time()
        if next_mode != mode and (now - last_switch_ts) >= COOLDOWN_SEC:
            apply_profile(cfg[next_mode])
            mode = next_mode
            last_switch_ts = now

        time.sleep(1.0)
```

## 운영 팁

- 시작은 `idle`에서 시작하고, 부하 상승 시 `busy`로 이동.
- `torchcompile`은 busy 프로파일의 latency 이득이 작아서, 현장에서는
  - `to_busy` 조건을 좀 더 보수적으로 잡거나,
  - governor 유지 fallback을 같이 두는 것이 안전.
- 주 1회 이상 같은 기준으로 재측정해서 프로파일 갱신 권장.

