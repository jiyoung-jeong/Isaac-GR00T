#!/usr/bin/env python3
"""
(A) Plot energy-per-inference sweep vs GPU freq for 3 backends, excluding 1st inference.

Uses result/thor_gr00t_server_*_{backend}_gpufreq_{freq}/inference_energy.csv
and plots mean E_gpu_J and mean E_vin_J across inferences 2..N.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]
GPUFREQ_SUFFIXES = [("801MHz", 0.801), ("1.305GHz", 1.305), ("1.575GHz", 1.575)]


def find_run_csv(result_dir: Path, backend: str, freq_token: str) -> Path:
    matches = sorted(result_dir.glob(f"thor_gr00t_server_*_{backend}_gpufreq_{freq_token}/inference_energy.csv"))
    if len(matches) != 1:
        raise SystemExit(f"Expected 1 match for {backend} gpufreq {freq_token}, got {len(matches)}: {matches}")
    return matches[0]


def find_no_suffix_csv(result_dir: Path, backend: str) -> Optional[Path]:
    """Run dir name ends with _backend (no _suffix after), e.g. ..._pytorch."""
    matches = [p for p in result_dir.glob(f"thor_gr00t_server_*_{backend}/inference_energy.csv") if p.parent.name.endswith("_" + backend)]
    return matches[-1] if matches else None


def load_energy_means(csv_path: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_path)
    df = df[df["inference_id"] >= 2].copy()
    if len(df) == 0:
        raise SystemExit(f"No inferences after excluding 1st in {csv_path}")
    return float(df["E_gpu_J"].mean()), float(df["E_vin_J"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result")
    ap.add_argument("--out", default="/home/Thor/compare_plots_means_sweeps/sweep_gpufreq_energy.png")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)

    x = np.array([v for _, v in GPUFREQ_SUFFIXES], dtype=float)
    xlabels = [t for t, _ in GPUFREQ_SUFFIXES]
    default_x = 0.72  # GPU 스윕만 0.801에 가깝게 (다른 건 0.38/0.52 유지)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for backend in BACKENDS:
        E_gpu = []
        E_vin = []
        for freq_token, _ in GPUFREQ_SUFFIXES:
            csv_path = find_run_csv(result_dir, backend, freq_token)
            eg, ev = load_energy_means(csv_path)
            E_gpu.append(eg)
            E_vin.append(ev)
        line0, = axes[0].plot(x, E_gpu, marker="o", linewidth=1.8, label=backend)
        line1, = axes[1].plot(x, E_vin, marker="o", linewidth=1.8, label=backend)
        p = find_no_suffix_csv(result_dir, backend)
        if p is not None:
            eg, ev = load_energy_means(p)
            axes[0].plot(default_x, eg, marker="^", markersize=10, linestyle="none", color=line0.get_color())
            axes[1].plot(default_x, ev, marker="^", markersize=10, linestyle="none", color=line1.get_color())

    axes[0].set_ylabel("Mean E_gpu per inference (J)\n(mean over i>=2)")
    axes[1].set_ylabel("Mean E_vin per inference (J)\n(mean over i>=2)")
    axes[1].set_xlabel("GPU frequency (GHz)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    x_ticks = np.append(x, default_x)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(xlabels + ["default"])
    fig.suptitle("Energy per inference vs GPU freq (exclude 1st inference)")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

