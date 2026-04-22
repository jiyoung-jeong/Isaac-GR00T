from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def nvtx_range(name: str):
    """Best-effort NVTX range that is a no-op when CUDA/NVTX is unavailable."""
    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def wrap_nvtx_range(fn, name: str):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with nvtx_range(name):
            return fn(*args, **kwargs)

    return wrapped
