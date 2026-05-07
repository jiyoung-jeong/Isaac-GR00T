#!/usr/bin/env python3
"""
Plot mean latency (duration_ms) sweeps, excluding 1st inference (use inferences 2..N).

Generates 3 plots in out-dir:
  - sweep_gpufreq_latency.png
  - sweep_emcfreq_latency.png
  - sweep_emcfreq_latency__gpufreq_{fixed}.png

Uses result/thor_gr00t_server_*/*/inference_energy.csv (per inference durations).
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]
GPUFREQ_TOKENS: List[Tuple[str, float]] = [("801MHz", 0.801), ("1.305GHz", 1.305), ("1.575GHz", 1.575)]
EMC_TOKENS: List[Tuple[str, float]] = [("665MHz", 0.665), ("2.75GHz", 2.75), ("3.2GHz", 3.2), ("4.26GHz", 4.26)]


def mean_latency_ms(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "inference_id" not in df.columns or "duration_ms" not in df.columns:
        raise ValueError(f"missing columns in {csv_path}")
    df = df[df["inference_id"] >= 2].copy()
    if len(df) == 0:
        raise ValueError(f"no inferences after excluding 1st in {csv_path}")
    return float(df["duration_ms"].astype(float).mean())


def find_one_or_none(result_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(result_dir.glob(pattern))
    if len(matches) == 0:
        return None
    return matches[-1]


def find_no_suffix_csv(result_dir: Path, backend: str) -> Optional[Path]:
    matches = [p for p in result_dir.glob(f"thor_gr00t_server_*_{backend}/inference_energy.csv") if p.parent.name.endswith("_" + backend)]
    return matches[-1] if matches else None


def plot(
    out: Path,
    title: str,
    xlabel: str,
    x: np.ndarray,
    xlabels: List[str],
    series: List[Tuple[str, List[float]]],
    default_vals: Optional[dict] = None,
    default_x: float = 0.5,
):
    plt.figure(figsize=(10, 5))
    for label, y in series:
        line, = plt.plot(x, y, marker="o", linewidth=1.9, label=label)
        if default_vals and label in default_vals:
            plt.plot(default_x, default_vals[label], marker="^", markersize=10, linestyle="none", color=line.get_color())
    plt.ylabel("Mean latency (ms)\n(mean over inferences i>=2)")
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    if default_vals:
        x_ticks = np.append(x, default_x)
        plt.xticks(x_ticks, xlabels + ["default"])
    else:
        plt.xticks(x, xlabels)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result")
    ap.add_argument("--out-dir", default="/home/Thor/compare_plots_means_sweeps")
    ap.add_argument("--fixed-gpufreq", default="1.305GHz")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)

    default_lat = {}
    for backend in BACKENDS:
        p = find_no_suffix_csv(result_dir, backend)
        if p is not None:
            default_lat[backend] = mean_latency_ms(p)

    # 1) GPUfreq sweep
    x = np.array([v for _, v in GPUFREQ_TOKENS], dtype=float)
    xlabels = [t for t, _ in GPUFREQ_TOKENS]
    series = []
    for backend in BACKENDS:
        y = []
        for token, _ in GPUFREQ_TOKENS:
            run_p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_gpufreq_{token}/inference_energy.csv")
            if run_p is None:
                y.append(float("nan"))
            else:
                y.append(mean_latency_ms(run_p))
        series.append((backend, y))
    plot(
        out_dir / "sweep_gpufreq_latency.png",
        "Mean latency vs GPU freq (exclude 1st inference)",
        "GPU frequency (GHz)",
        x,
        xlabels,
        series,
        default_vals=default_lat if default_lat else None,
        default_x=0.72,
    )

    # 2) EMCfreq sweep (no gpufreq constraint)
    x2 = np.array([v for _, v in EMC_TOKENS], dtype=float)
    x2labels = [t for t, _ in EMC_TOKENS]
    series2 = []
    for backend in BACKENDS:
        y = []
        for token, _ in EMC_TOKENS:
            run_p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}/inference_energy.csv")
            if run_p is None:
                y.append(float("nan"))
            else:
                y.append(mean_latency_ms(run_p))
        series2.append((backend, y))
    plot(
        out_dir / "sweep_emcfreq_latency.png",
        "Mean latency vs EMC freq (exclude 1st inference)",
        "EMC frequency (GHz)",
        x2,
        x2labels,
        series2,
        default_vals=default_lat if default_lat else None,
        default_x=0.38,
    )

    # 3) EMCfreq sweep with fixed GPU freq
    fixed = args.fixed_gpufreq
    series3 = []
    for backend in BACKENDS:
        y = []
        for token, _ in EMC_TOKENS:
            run_p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}_gpufreq_{fixed}/inference_energy.csv")
            if run_p is None:
                y.append(float("nan"))
            else:
                y.append(mean_latency_ms(run_p))
        series3.append((backend, y))
    plot(
        out_dir / f"sweep_emcfreq_latency__gpufreq_{fixed}.png",
        f"Mean latency vs EMC freq @ GPU freq {fixed} (exclude 1st inference)",
        "EMC frequency (GHz)",
        x2,
        x2labels,
        series3,
        default_vals=default_lat if default_lat else None,
        default_x=0.38,
    )


if __name__ == "__main__":
    main()

