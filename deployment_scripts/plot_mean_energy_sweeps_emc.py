#!/usr/bin/env python3
"""
Energy per inference vs EMC freq sweeps (exclude 1st inference).

Creates:
  - sweep_emcfreq_energy.png: emcfreq sweep where suffix is exactly emcfreq_{token}
  - sweep_emcfreq_energy__gpufreq_{fixed}.png: emcfreq sweep with fixed gpufreq

Energy is taken from inference_energy.csv columns:
  E_gpu_J, E_vin_J
and averaged across inferences 2..N.
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
EMC_TOKENS: List[Tuple[str, float]] = [("665MHz", 0.665), ("2.75GHz", 2.75), ("3.2GHz", 3.2), ("4.26GHz", 4.26)]


def mean_energy(csv_path: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_path)
    df = df[df["inference_id"] >= 2].copy()
    if len(df) == 0:
        raise ValueError(f"No inferences after excluding 1st in {csv_path}")
    return float(df["E_gpu_J"].mean()), float(df["E_vin_J"].mean())


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
    x: np.ndarray,
    xlabels: List[str],
    series: List[Tuple[str, List[float], List[float]]],
    default_vals: Optional[Dict[str, Tuple[float, float]]] = None,
    default_x: float = 0.4,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, eg, ev in series:
        line0, = axes[0].plot(x, eg, marker="o", linewidth=1.8, label=label)
        line1, = axes[1].plot(x, ev, marker="o", linewidth=1.8, label=label)
        if default_vals and label in default_vals:
            eg_d, ev_d = default_vals[label]
            axes[0].plot(default_x, eg_d, marker="^", markersize=10, linestyle="none", color=line0.get_color())
            axes[1].plot(default_x, ev_d, marker="^", markersize=10, linestyle="none", color=line1.get_color())

    axes[0].set_ylabel("Mean E_gpu per inference (J)\n(mean over i>=2)")
    axes[1].set_ylabel("Mean E_vin per inference (J)\n(mean over i>=2)")
    axes[1].set_xlabel("EMC frequency (GHz)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    if default_vals:
        x_ticks = np.append(x, default_x)
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels(xlabels + ["default"])
    else:
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(xlabels)
    fig.suptitle(title)
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

    x = np.array([v for _, v in EMC_TOKENS], dtype=float)
    xlabels = [t for t, _ in EMC_TOKENS]

    default_vals = {}
    for backend in BACKENDS:
        p = find_no_suffix_csv(result_dir, backend)
        if p is not None:
            default_vals[backend] = mean_energy(p)

    # 1) emcfreq only
    series = []
    for backend in BACKENDS:
        eg = []
        ev = []
        for token, _ in EMC_TOKENS:
            run_p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}/inference_energy.csv")
            if run_p is None:
                eg.append(float("nan"))
                ev.append(float("nan"))
            else:
                a, b = mean_energy(run_p)
                eg.append(a)
                ev.append(b)
        series.append((backend, eg, ev))

    plot(
        out_dir / "sweep_emcfreq_energy.png",
        "Energy per inference vs EMC freq (exclude 1st inference)",
        x,
        xlabels,
        series,
        default_vals=default_vals if default_vals else None,
        default_x=0.38,
    )

    # 2) emcfreq with fixed gpufreq (same default_vals)
    fixed = args.fixed_gpufreq
    series2 = []
    for backend in BACKENDS:
        eg = []
        ev = []
        for token, _ in EMC_TOKENS:
            run_p = find_one_or_none(
                result_dir,
                f"thor_gr00t_server_*_{backend}_emcfreq_{token}_gpufreq_{fixed}/inference_energy.csv",
            )
            if run_p is None:
                eg.append(float("nan"))
                ev.append(float("nan"))
            else:
                a, b = mean_energy(run_p)
                eg.append(a)
                ev.append(b)
        series2.append((backend, eg, ev))

    plot(
        out_dir / f"sweep_emcfreq_energy__gpufreq_{fixed}.png",
        f"Energy per inference vs EMC freq @ GPU freq {fixed} (exclude 1st inference)",
        x,
        xlabels,
        series2,
        default_vals=default_vals if default_vals else None,
        default_x=0.38,
    )


if __name__ == "__main__":
    main()

