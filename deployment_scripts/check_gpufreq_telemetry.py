#!/usr/bin/env python3
"""
(B) Check whether telemetry gpu_freq_hz matches expected gpufreq setting for gpufreq sweeps.

Reads result/thor_gr00t_server_*_{backend}_gpufreq_{freq}/telemetry_raw.csv
and prints a summary table of gpu_freq_hz statistics.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]
GPUFREQ_SUFFIXES = [("801MHz", 801_000_000), ("1.305GHz", 1_305_000_000), ("1.575GHz", 1_575_000_000)]


def find_telemetry_csv(result_dir: Path, backend: str, freq_token: str) -> Path:
    matches = sorted(result_dir.glob(f"thor_gr00t_server_*_{backend}_gpufreq_{freq_token}/telemetry_raw.csv"))
    if len(matches) != 1:
        raise SystemExit(f"Expected 1 telemetry match for {backend} gpufreq {freq_token}, got {len(matches)}: {matches}")
    return matches[0]


def summarize_freq(csv_path: Path, expected_hz: int):
    df = pd.read_csv(csv_path)
    if "gpu_freq_hz" not in df.columns:
        raise SystemExit(f"Missing gpu_freq_hz column in {csv_path}")
    x = df["gpu_freq_hz"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    x_pos = x[x > 0]
    if x_pos.size == 0:
        return {
            "n": int(x.size),
            "n_pos": 0,
            "mean_hz": float("nan"),
            "p50_hz": float("nan"),
            "p95_hz": float("nan"),
            "max_hz": float("nan"),
            "match_frac": float("nan"),
        }
    match_frac = float(np.mean(np.isclose(x_pos, expected_hz, rtol=0, atol=1)))
    return {
        "n": int(x.size),
        "n_pos": int(x_pos.size),
        "mean_hz": float(np.mean(x_pos)),
        "p50_hz": float(np.percentile(x_pos, 50)),
        "p95_hz": float(np.percentile(x_pos, 95)),
        "max_hz": float(np.max(x_pos)),
        "match_frac": match_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result")
    ap.add_argument("--csv-out", default="/home/Thor/compare_plots_means_sweeps/gpufreq_telemetry_check.csv")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)

    rows: List[dict] = []
    for backend in BACKENDS:
        for freq_token, expected_hz in GPUFREQ_SUFFIXES:
            telem = find_telemetry_csv(result_dir, backend, freq_token)
            s = summarize_freq(telem, expected_hz)
            rows.append(
                {
                    "backend": backend,
                    "gpufreq": freq_token,
                    "expected_hz": expected_hz,
                    "telemetry_csv": str(telem),
                    **s,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.csv_out, index=False)
    print(out.to_string(index=False, max_colwidth=80))
    print(f"\n[OK] wrote {args.csv_out}")


if __name__ == "__main__":
    main()

