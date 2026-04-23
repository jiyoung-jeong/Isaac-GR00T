#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TOP_LEVEL_PHASES = [
    "VLA/ViT",
    "VLA/LLM",
    "VLA/action_head",
    "VLA/backbone",
    "VLA/get_action",
]


def fmt_freq(freq_hz: float) -> str:
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.3f}GHz"
    return f"{freq_hz / 1e6:.0f}MHz"


def load_phase_rows(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for phase_csv in sorted(root.glob("gpufreq_*/phase_summary.csv")):
        df = pd.read_csv(phase_csv)
        if df.empty:
            continue
        df["run_dir"] = phase_csv.parent.name
        rows.append(df)

    if not rows:
        raise SystemExit(f"No phase_summary.csv files found under {root}")

    out = pd.concat(rows, ignore_index=True)
    out = out[out["phase"].isin(TOP_LEVEL_PHASES)].copy()
    out["requested_ok"] = out["requested_hz"] == out["actual_hz"]
    return out


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    phases = [p for p in TOP_LEVEL_PHASES if p in set(df["phase"])]
    valid = df[df["requested_ok"]].sort_values(["actual_hz", "phase"])
    suspect = df[~df["requested_ok"]].sort_values(["requested_hz", "phase"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for phase in phases:
        phase_df = valid[valid["phase"] == phase].sort_values("actual_hz")
        if phase_df.empty:
            continue
        x = phase_df["actual_hz"] / 1e9
        y = phase_df[metric]
        ax.plot(x, y, marker="o", linewidth=2, label=phase)

        suspect_df = suspect[suspect["phase"] == phase]
        if not suspect_df.empty:
            sx = suspect_df["requested_hz"] / 1e9
            sy = suspect_df[metric]
            ax.scatter(
                sx,
                sy,
                marker="x",
                s=80,
                linewidths=2,
                color=ax.lines[-1].get_color(),
                alpha=0.9,
            )

    tick_vals = sorted(valid["actual_hz"].unique())
    tick_labels = [fmt_freq(v) for v in tick_vals]
    if tick_vals:
        ax.set_xticks([v / 1e9 for v in tick_vals])
        ax.set_xticklabels(tick_labels)

    ax.set_xlabel("GPU Frequency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stacked_latency(df: pd.DataFrame, out_path: Path) -> None:
    valid = df[df["requested_ok"]].copy()
    valid = valid[valid["phase"].isin(["VLA/ViT", "VLA/LLM", "VLA/action_head"])].copy()
    if valid.empty:
        return

    pivot = (
        valid.pivot_table(
            index="actual_hz",
            columns="phase",
            values="latency_ms_mean",
            aggfunc="first",
        )
        .reindex(columns=["VLA/ViT", "VLA/LLM", "VLA/action_head"])
        .sort_index()
    )

    x = range(len(pivot.index))
    xlabels = [fmt_freq(v) for v in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = None
    for phase in pivot.columns:
        vals = pivot[phase].fillna(0.0)
        ax.bar(x, vals, bottom=bottom, label=phase)
        bottom = vals if bottom is None else bottom + vals

    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("GPU Frequency")
    ax.set_title("Top-Level Phase Latency Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Thor phase metrics from phase_summary.csv files.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("thor_measurements/gpu_sweep_vla"),
        help="Root directory that contains gpufreq_* subdirectories.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write plots. Defaults to <root>/plots_phase.",
    )
    args = ap.parse_args()

    df = load_phase_rows(args.root)
    out_dir = args.out_dir or (args.root / "plots_phase")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        df,
        "latency_ms_mean",
        "Latency (ms)",
        "Phase Latency vs GPU Frequency",
        out_dir / "phase_latency_vs_gpufreq.png",
    )
    plot_metric(
        df,
        "gpu_avg_power_w_mean",
        "GPU Avg Power (W)",
        "Phase Average GPU Power vs GPU Frequency",
        out_dir / "phase_gpu_power_vs_gpufreq.png",
    )
    plot_metric(
        df,
        "gpu_energy_j_mean",
        "GPU Energy (J)",
        "Phase GPU Energy vs GPU Frequency",
        out_dir / "phase_gpu_energy_vs_gpufreq.png",
    )
    plot_stacked_latency(df, out_dir / "phase_latency_stacked.png")

    suspect = df[~df["requested_ok"]][["run_dir", "requested_hz", "actual_hz"]].drop_duplicates()
    if not suspect.empty:
        suspect.to_csv(out_dir / "suspect_runs.csv", index=False)

    print(f"Wrote plots to {out_dir}")
    if not suspect.empty:
        print(f"Wrote {out_dir / 'suspect_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
