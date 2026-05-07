# get_action 구간만 4ms 단위 전력 측정

get_action이 시작될 때부터 끝날 때까지만 짧은 주기(4ms)로 power를 측정하려면 아래 두 가지를 적용하면 됩니다.

## 1. inference_service.py (서버)

`get_action` 진입 시 트리거 파일을 만들고, 종료 시 제거합니다. 측정 스크립트는 이 파일 존재 여부로 4ms/일반 주기를 전환합니다.

**상단 import 추가:**
```python
import os
# ... 기존 ...
from pathlib import Path
```

**profiled_inference 래퍼 수정 (기존 블록 전체 교체):**
```python
        # Thor 전력 측정: get_action 구간만 4ms 샘플링 트리거
        _thor_trigger = Path(os.environ.get("THOR_INFERENCE_ACTIVE", "/tmp/thor_inference_active"))

        # 우리가 만든 '스톱워치 포함' 가짜 함수
        def profiled_inference(*args, **kwargs):
            nvtx.range_push("Total_Policy_Inference")
            try:
                _thor_trigger.touch()
            except OSError:
                pass
            try:
                return original_method(*args, **kwargs)
            finally:
                try:
                    _thor_trigger.unlink(missing_ok=True)
                except OSError:
                    pass
                nvtx.range_pop()
```

- Docker에서 서버를 돌릴 때 `-v /tmp:/tmp` 로 두면 호스트 Thor 측정 스크립트가 같은 파일을 볼 수 있습니다.

## 2. thor_server_measure_and_analyze.sh (측정)

**설정부에 추가 (INTERVAL_MS 아래):**
```bash
FAST_INTERVAL_MS=4
THOR_INFERENCE_ACTIVE="${THOR_INFERENCE_ACTIVE:-/tmp/thor_inference_active}"
```

**로거 루프에서 `sleep_ms` 부분만 다음으로 교체:**
```bash
      echo "${ts_ns},${pg},${pc},${vin_w},${gf},${c0},${c4}" >> "${RAW_CSV}"
      if [[ -f "${THOR_INFERENCE_ACTIVE}" ]]; then
        sleep_ms "${FAST_INTERVAL_MS}"
      else
        sleep_ms "${INTERVAL_MS}"
      fi
```

동작 요약:
- 평소: `INTERVAL_MS`(예: 10ms) 간격으로 계속 기록.
- 서버가 `get_action` 진입 시 `/tmp/thor_inference_active` 생성 → 로거가 파일을 감지하면 `FAST_INTERVAL_MS`(4ms)로 샘플링.
- `get_action` 종료 시 서버가 파일 삭제 → 로거는 다시 `INTERVAL_MS`로 복귀.

실제 주기는 sysfs 읽기 등 오버헤드로 4ms보다 다소 길어질 수 있습니다. 더 촘촘히 하려면 `FAST_INTERVAL_MS=2` 등으로 줄여 보면 됩니다.
