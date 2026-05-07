#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_power_series(csv_path: Path, label: str):
    df = pd.read_csv(csv_path)
    required = {"inference_id", "duration_ms", "E_gpu_J", "E_vin_J"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["inference_id"] >= 2].copy()  # exclude 1st inference
    if len(df) == 0:
        raise SystemExit(f"No inferences after excluding 1st in {csv_path}")

    t_s = df["duration_ms"].astype(float) / 1e3
    df["P_gpu_W"] = df["E_gpu_J"].astype(float) / t_s
    df["P_vin_W"] = df["E_vin_J"].astype(float) / t_s
    df["label"] = label
    return df


def main():
    p = argparse.ArgumentParser(description="Plot per-inference average power from inference_energy.csv (exclude 1st inference).")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="Per-inference average power (exclude 1st)", help="Plot title")
    p.add_argument("--csv", action="append", nargs=2, metavar=("LABEL", "CSV_PATH"), required=True,
                   help="Add a series: LABEL CSV_PATH (repeatable)")
    args = p.parse_args()

    series = [load_power_series(Path(csv_path), label) for label, csv_path in args.csv]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for df in series:
        axes[0].plot(df["inference_id"], df["P_gpu_W"], marker="o", linewidth=1.6, label=df["label"].iloc[0])
        axes[1].plot(df["inference_id"], df["P_vin_W"], marker="o", linewidth=1.6, label=df["label"].iloc[0])

    axes[0].set_ylabel("GPU avg power (W) = E_gpu_J / duration_s")
    axes[1].set_ylabel("VIN avg power (W) = E_vin_J / duration_s")
    axes[1].set_xlabel("inference_id")

    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.suptitle(args.title)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()

