# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NVTX 구간을 monotonic ns로 파일에 기록. Power 로거와 같은 시간축으로 inference/ViT/LLM/Action 구간 분석용.
"""
import os
import time
from pathlib import Path

_NVTX_RANGES_CSV = Path(os.environ.get("NVTX_RANGES_CSV", "/tmp/nvtx_ranges.csv"))


def log_nvtx_range(event: str) -> None:
    """NVTX 구간 start/end를 monotonic ns로 파일에 기록."""
    try:
        ts_ns = time.monotonic_ns()
        with open(_NVTX_RANGES_CSV, "a", encoding="utf-8") as f:
            f.write(f"{ts_ns},{event}\n")
    except OSError:
        pass


def get_nvtx_log_path() -> Path:
    return _NVTX_RANGES_CSV
