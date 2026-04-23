from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import time

import torch


_NVTX_RANGES_CSV = os.environ.get("NVTX_RANGES_CSV")


def _log_nvtx_event(event: str) -> None:
    if not _NVTX_RANGES_CSV:
        return
    try:
        path = Path(_NVTX_RANGES_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{time.monotonic_ns()},{event}\n")
    except OSError:
        pass


@contextmanager
def nvtx_range(name: str):
    """Best-effort NVTX range that is a no-op when CUDA/NVTX is unavailable."""
    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        _log_nvtx_event(f"{name}_START")
        yield
    finally:
        _log_nvtx_event(f"{name}_END")
        if pushed:
            torch.cuda.nvtx.range_pop()


def wrap_nvtx_range(fn, name: str):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with nvtx_range(name):
            return fn(*args, **kwargs)

    return wrapped
