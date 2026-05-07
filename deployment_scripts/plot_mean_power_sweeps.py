#!/usr/bin/env python3
"""
Plot mean/median power summary across frequency sweeps from mean_power_summary.csv.

Generates:
  1) GPUfreq sweep: gpufreq_{801MHz,1.305GHz,1.575GHz}
  2) EMCfreq sweep (no gpufreq in suffix): emcfreq_{665MHz,2.75GHz,3.2GHz,4.26GHz}
  3) EMCfreq sweep with gpufreq fixed (default 1.305GHz): emcfreq_*_gpufreq_1.305GHz

Each plot has 2 panels: GPU power and VIN power, with 3 backend lines.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]


def parse_freq_token(token: str) -> Optional[float]:
    """
    Convert e.g. '801MHz' -> 0.801, '1.305GHz' -> 1.305 (GHz units).
    """
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(MHz|GHz)", token)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "MHz":
        return val / 1000.0
    return val


def get_backend_cols(kind: str) -> List[str]:
    # kind: "P_gpu_W" or "P_vin_W"
    return [f"{b}_{kind}" for b in BACKENDS]


def plot_sweep(
    df: pd.DataFrame,
    xvals: List[float],
    xlabels: List[str],
    rows: List[pd.Series],
    title: str,
    out_png: Path,
    default_row: Optional[pd.Series] = None,
    default_x: Optional[float] = None,
    default_label: str = "default",
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    x = np.array(xvals, dtype=float)
    has_default = default_row is not None and default_x is not None and not default_row.isna().all()

    for b in BACKENDS:
        y_gpu = [float(r.get(f"{b}_P_gpu_W", np.nan)) for r in rows]
        y_vin = [float(r.get(f"{b}_P_vin_W", np.nan)) for r in rows]
        line0, = axes[0].plot(x, y_gpu, marker="o", linewidth=1.8, label=b)
        line1, = axes[1].plot(x, y_vin, marker="o", linewidth=1.8, label=b)
        if has_default:
            axes[0].plot(default_x, float(default_row.get(f"{b}_P_gpu_W", np.nan)), marker="^", markersize=10, linestyle="none", color=line0.get_color())
            axes[1].plot(default_x, float(default_row.get(f"{b}_P_vin_W", np.nan)), marker="^", markersize=10, linestyle="none", color=line1.get_color())

    axes[0].set_ylabel("GPU power (W)\n(summary from i>=2)")
    axes[1].set_ylabel("VIN power (W)\n(summary from i>=2)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    if has_default:
        x_ticks = np.append(x, default_x)
        x_tick_labels = xlabels + [default_label]
    else:
        x_ticks = x
        x_tick_labels = xlabels
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_tick_labels)

    fig.suptitle(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[OK] wrote {out_png}")


def pick_rows_by_suffix(df: pd.DataFrame, suffixes: List[str]) -> List[pd.Series]:
    idx = {s: i for i, s in enumerate(df["suffix"].astype(str).tolist())}
    out = []
    for s in suffixes:
        if s not in idx:
            raise SystemExit(f"Missing suffix in summary CSV: {s}")
        out.append(df.iloc[idx[s]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv",
        default="/home/Thor/compare_plots_means/mean_power_summary.csv",
        help="Path to mean_power_summary.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/Thor/compare_plots_means_sweeps",
        help="Output directory for sweep plots (must be writable)",
    )
    ap.add_argument(
        "--fixed-gpufreq",
        default="1.305GHz",
        help="For EMC sweep with fixed GPU freq: gpufreq token (default 1.305GHz)",
    )
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if "suffix" not in df.columns:
        raise SystemExit(f"Missing 'suffix' column in {summary_csv}")

    no_suffix_idx = df["suffix"].astype(str).eq("(no_suffix)")
    default_row = df.loc[no_suffix_idx].iloc[0] if no_suffix_idx.any() else None

    # 1) GPUfreq sweep (default at x=0.5)
    gpufreq_suffixes = ["gpufreq_801MHz", "gpufreq_1.305GHz", "gpufreq_1.575GHz"]
    gpufreq_rows = pick_rows_by_suffix(df, gpufreq_suffixes)
    gpux = [parse_freq_token(s.split("_", 1)[1]) for s in gpufreq_suffixes]
    if any(v is None for v in gpux):
        raise SystemExit("Failed to parse gpufreq tokens")
    plot_sweep(
        df,
        xvals=[float(v) for v in gpux],
        xlabels=[s.split("_", 1)[1] for s in gpufreq_suffixes],
        rows=gpufreq_rows,
        title="Mean per-inference avg power vs GPU freq (exclude 1st inference)",
        out_png=out_dir / "sweep_gpufreq.png",
        default_row=default_row,
        default_x=0.72,
        default_label="default",
    )

    # 2) EMCfreq sweep (no gpufreq constraint)
    emcfreq_suffixes = ["emcfreq_665MHz", "emcfreq_2.75GHz", "emcfreq_3.2GHz", "emcfreq_4.26GHz"]
    emc_rows = pick_rows_by_suffix(df, emcfreq_suffixes)
    emcx = [parse_freq_token(s.split("_", 1)[1]) for s in emcfreq_suffixes]
    if any(v is None for v in emcx):
        raise SystemExit("Failed to parse emcfreq tokens")
    plot_sweep(
        df,
        xvals=[float(v) for v in emcx],
        xlabels=[s.split("_", 1)[1] for s in emcfreq_suffixes],
        rows=emc_rows,
        title="Mean per-inference avg power vs EMC freq (exclude 1st inference)",
        out_png=out_dir / "sweep_emcfreq.png",
        default_row=default_row,
        default_x=0.38,
        default_label="default",
    )

    # 3) EMCfreq sweep with fixed GPU freq
    fixed = args.fixed_gpufreq
    emcfreq_fixed_suffixes = [
        f"emcfreq_665MHz_gpufreq_{fixed}",
        f"emcfreq_2.75GHz_gpufreq_{fixed}",
        f"emcfreq_3.2GHz_gpufreq_{fixed}",
        f"emcfreq_4.26GHz_gpufreq_{fixed}",
    ]
    emc_fixed_rows = pick_rows_by_suffix(df, emcfreq_fixed_suffixes)
    emc_fixed_x = [parse_freq_token(s.split("_", 2)[1]) for s in emcfreq_fixed_suffixes]  # token after 'emcfreq_'
    if any(v is None for v in emc_fixed_x):
        raise SystemExit("Failed to parse emcfreq tokens (fixed gpufreq sweep)")
    plot_sweep(
        df,
        xvals=[float(v) for v in emc_fixed_x],
        xlabels=[s.split("_", 2)[1] for s in emcfreq_fixed_suffixes],
        rows=emc_fixed_rows,
        title=f"Mean per-inference avg power vs EMC freq @ GPU freq {fixed} (exclude 1st inference)",
        out_png=out_dir / f"sweep_emcfreq__gpufreq_{fixed}.png",
        default_row=default_row,
        default_x=0.38,
        default_label="default",
    )

    print(f"[OK] out_dir={out_dir}")


if __name__ == "__main__":
    main()

