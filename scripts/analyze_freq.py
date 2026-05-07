#!/usr/bin/env python3
"""
GR00T inference frequency log 분석 스크립트

/tmp/gr00t_logs/n15_infer_baseline/freq.csv 를 로드하여
GPU/CPU 주파수 시계열, 통계, 분포를 분석하고 PNG로 저장합니다.

Usage:
    python scripts/analyze_freq.py
    python scripts/analyze_freq.py --input /path/to/freq.csv --out-dir ./freq_analysis
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_freq_csv(path: str) -> pd.DataFrame:
    """freq.csv 로드 후 ts_ns를 상대 시간(초)으로 변환."""
    df = pd.read_csv(path)
    if "ts_ns" not in df.columns:
        raise ValueError("freq.csv must have column 'ts_ns'")
    ts_ns = df["ts_ns"].values.astype(np.int64)
    df = df.copy()
    df["t_s"] = (ts_ns - ts_ns[0]) / 1e9
    return df


def freq_columns(df: pd.DataFrame):
    """GPU/CPU 주파수 컬럼 목록 (ts_ns, t_s 제외)."""
    skip = {"ts_ns", "t_s"}
    return [c for c in df.columns if c not in skip]


def _to_mhz(series: pd.Series, is_cpu_policy: bool) -> pd.Series:
    """Raw 값을 MHz로 변환. GPU는 Hz, CPU policy는 kHz로 가정."""
    if is_cpu_policy:
        return series / 1e3  # kHz -> MHz
    return series / 1e6  # Hz -> MHz


def analyze_summary(df: pd.DataFrame, freq_cols: list) -> pd.DataFrame:
    """주파수별 min/max/mean/std (Hz/kHz -> MHz)."""
    rows = []
    for col in freq_cols:
        raw = df[col].dropna()
        is_cpu = col.startswith("policy")
        mhz = _to_mhz(raw, is_cpu_policy=is_cpu)
        rows.append({
            "column": col,
            "min_MHz": mhz.min(),
            "max_MHz": mhz.max(),
            "mean_MHz": mhz.mean(),
            "std_MHz": mhz.std() if len(mhz) > 1 else 0,
            "count": len(mhz),
        })
    return pd.DataFrame(rows)


def plot_timeseries(df: pd.DataFrame, freq_cols: list, out_path: Path):
    """주파수 시계열 플롯 (GPU vs CPU 구분)."""
    t = df["t_s"].values

    gpu_cols = [c for c in freq_cols if c.startswith("gpu-")]
    cpu_cols = [c for c in freq_cols if c not in gpu_cols]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # GPU (Hz -> MHz)
    ax = axes[0]
    for col in gpu_cols:
        ax.plot(t, df[col].values / 1e6, label=col, alpha=0.9)
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("GPU frequency over time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    # CPU (kHz -> MHz)
    ax = axes[1]
    for col in sorted(cpu_cols, key=lambda x: int("".join(filter(str.isdigit, x)) or "0")):
        ax.plot(t, _to_mhz(df[col], is_cpu_policy=True).values, label=col, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("CPU cluster frequency over time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_distribution(df: pd.DataFrame, freq_cols: list, out_path: Path):
    """주파수 분포 히스토그램."""
    gpu_cols = [c for c in freq_cols if c.startswith("gpu-")]
    cpu_cols = [c for c in freq_cols if c not in gpu_cols]
    n_plots = len(gpu_cols) + len(cpu_cols)
    if n_plots == 0:
        return

    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx, col in enumerate(gpu_cols + cpu_cols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        is_cpu = col.startswith("policy")
        mhz = _to_mhz(df[col].dropna(), is_cpu_policy=is_cpu).values
        ax.hist(mhz, bins=min(50, max(10, len(np.unique(mhz)))), edgecolor="black", alpha=0.7)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("MHz")
        ax.set_ylabel("count")

    for idx in range(n_plots, axes.size):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary_bars(summary_df: pd.DataFrame, out_path: Path):
    """주파수별 min/mean/max 막대 그래프."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(summary_df))
    w = 0.25
    ax.bar(x - w, summary_df["min_MHz"], width=w, label="min", color="steelblue", alpha=0.8)
    ax.bar(x, summary_df["mean_MHz"], width=w, label="mean", color="green", alpha=0.8)
    ax.bar(x + w, summary_df["max_MHz"], width=w, label="max", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["column"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Frequency summary (min / mean / max)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GR00T inference freq.csv")
    parser.add_argument(
        "--input",
        type=str,
        default="/tmp/gr00t_logs/n15_infer_baseline/freq.csv",
        help="Path to freq.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: same dir as input)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print summary, do not save plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_freq_csv(str(input_path))
    freq_cols = freq_columns(df)

    if not freq_cols:
        print("No frequency columns found.")
        return

    # Summary table
    summary = analyze_summary(df, freq_cols)
    print("\n=== Frequency summary (MHz) ===")
    print(summary.to_string(index=False))

    # Optional: save summary CSV
    summary_path = out_dir / "freq_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    duration_s = df["t_s"].iloc[-1] - df["t_s"].iloc[0]
    n_samples = len(df)
    print(f"\nDuration: {duration_s:.3f} s, Samples: {n_samples}, Interval ~{duration_s / max(1, n_samples - 1) * 1000:.1f} ms")

    if not args.no_plot:
        plot_timeseries(df, freq_cols, out_dir / "freq_timeseries.png")
        plot_distribution(df, freq_cols, out_dir / "freq_distribution.png")
        plot_summary_bars(summary, out_dir / "freq_summary_bars.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
