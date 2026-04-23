#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fmt_freq(freq_hz: float) -> str:
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.3f}GHz"
    return f"{freq_hz / 1e6:.0f}MHz"


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Thor GPU frequency sweep summary.")
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("thor_measurements/gpu_sweep_vla/summary.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit(f"Empty summary csv: {args.summary_csv}")

    out_dir = args.out_dir or args.summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("actual_hz").copy()
    x = df["actual_hz"] / 1e9
    xlabels = [fmt_freq(v) for v in df["actual_hz"]]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(x, df["gpu_power_mean_w"], marker="o", linewidth=2)
    axes[0].fill_between(x, df["gpu_power_mean_w"], alpha=0.15)
    axes[0].set_ylabel("GPU Mean Power (W)")
    axes[0].set_title("GPU Frequency Sweep")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df["gpu_energy_j"], marker="o", linewidth=2, color="C1")
    axes[1].fill_between(x, df["gpu_energy_j"], alpha=0.15, color="C1")
    axes[1].set_ylabel("GPU Energy (J)")
    axes[1].grid(True, alpha=0.3)

    # duration_s is the full benchmark duration here, not per-inference latency
    axes[2].plot(x, df["duration_s"], marker="o", linewidth=2, color="C2")
    axes[2].fill_between(x, df["duration_s"], alpha=0.15, color="C2")
    axes[2].set_ylabel("Run Duration (s)")
    axes[2].set_xlabel("GPU Frequency")
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xlabels, rotation=0)

    plt.tight_layout()
    out_path = out_dir / "gpu_sweep_summary.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    # Secondary figure with power mean / p95 / max
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df["gpu_power_mean_w"], marker="o", label="mean")
    ax.plot(x, df["gpu_power_p95_w"], marker="o", label="p95")
    ax.plot(x, df["gpu_power_max_w"], marker="o", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("GPU Power (W)")
    ax.set_xlabel("GPU Frequency")
    ax.set_title("GPU Power Statistics vs Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path2 = out_dir / "gpu_sweep_power_stats.png"
    plt.savefig(out_path2, dpi=180)
    plt.close(fig)

    print(f"Wrote {out_dir / 'gpu_sweep_summary.png'}")
    print(f"Wrote {out_dir / 'gpu_sweep_power_stats.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
