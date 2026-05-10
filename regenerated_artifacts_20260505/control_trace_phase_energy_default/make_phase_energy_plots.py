#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "source"
DEFAULT_OUT_DIR = SCRIPT_DIR

RAILS = [
    ("gpu_power_w", "GPU", "#f97316"),
    ("cpu_power_w", "CPU", "#16a34a"),
    ("vin_power_w", "VIN", "#2563eb"),
]
PHASE_ORDER = ["vit", "llm", "action_head"]
PHASE_LABELS = {"vit": "ViT", "llm": "LLM", "action_head": "Action"}


def integrate_segment(samples_df: pd.DataFrame, start_ns: int, end_ns: int, power_col: str) -> float:
    segment = samples_df[(samples_df["ts_ns"] >= start_ns) & (samples_df["ts_ns"] <= end_ns)].copy()
    if len(segment) < 2:
        return float("nan")

    ts = segment["ts_ns"].to_numpy(dtype=np.int64)
    power = segment[power_col].to_numpy(dtype=float)
    valid = np.isfinite(power)
    if valid.sum() < 2:
        return float("nan")

    ts = ts[valid]
    power = power[valid]
    dt = np.diff(ts) / 1e9
    return float(np.sum(0.5 * (power[:-1] + power[1:]) * dt))


def build_phase_energy(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    telemetry = pd.read_csv(source_dir / "telemetry_raw.csv").sort_values("ts_ns")
    phases = pd.read_csv(source_dir / "phase_ranges.csv")

    rows = []
    for row in phases.itertuples(index=False):
        for power_col, rail_name, _color in RAILS:
            rows.append(
                {
                    "inference_id": int(row.inference_id),
                    "phase": row.phase,
                    "rail": rail_name,
                    "energy_j": integrate_segment(telemetry, int(row.start_ns), int(row.end_ns), power_col),
                    "duration_ms": float(row.duration_ms),
                }
            )

    energy_df = pd.DataFrame(rows)
    summary = (
        energy_df.groupby(["phase", "rail"], as_index=False)
        .agg(
            energy_mean_j=("energy_j", "mean"),
            energy_std_j=("energy_j", "std"),
            duration_mean_ms=("duration_ms", "mean"),
        )
    )
    summary["phase_label"] = summary["phase"].map(PHASE_LABELS)
    return energy_df, summary


def plot_grouped_energy(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    x = np.arange(len(PHASE_ORDER))
    width = 0.24

    for i, (_power_col, rail_name, color) in enumerate(RAILS):
        sub = summary[summary["rail"] == rail_name].set_index("phase").reindex(PHASE_ORDER)
        means = sub["energy_mean_j"].to_numpy(dtype=float)
        stds = sub["energy_std_j"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            color=color,
            label=rail_name,
            yerr=stds,
            capsize=4,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS[p] for p in PHASE_ORDER])
    ax.set_ylabel("Phase energy (J)")
    ax.set_title("Control trace default: per-phase energy (5 measured inferences)", pad=14)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_energy_bar.png", bbox_inches="tight")
    fig.savefig(out_dir / "phase_energy_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy_composition(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    x = np.arange(len(PHASE_ORDER))
    gpu_means = (
        summary[summary["rail"] == "GPU"].set_index("phase").reindex(PHASE_ORDER)["energy_mean_j"].to_numpy(dtype=float)
    )
    vin_means = (
        summary[summary["rail"] == "VIN"].set_index("phase").reindex(PHASE_ORDER)["energy_mean_j"].to_numpy(dtype=float)
    )

    ax.bar(x, gpu_means, color="#f97316", alpha=0.85, label="GPU")
    cpu_means = (
        summary[summary["rail"] == "CPU"].set_index("phase").reindex(PHASE_ORDER)["energy_mean_j"].to_numpy(dtype=float)
    )
    ax.bar(x, cpu_means, bottom=gpu_means, color="#16a34a", alpha=0.85, label="CPU")
    ax.plot(x, vin_means, color="#2563eb", marker="o", linewidth=2.0, label="VIN total")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS[p] for p in PHASE_ORDER])
    ax.set_ylabel("Energy (J)")
    ax.set_title("Control trace default: phase energy vs VIN total", pad=14)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_energy_composition.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, source_dir: Path, out_dir: Path) -> None:
    try:
        source_display = source_dir.resolve().relative_to(out_dir.resolve())
    except ValueError:
        source_display = source_dir

    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Phase Energy Summary\n\n")
        f.write(f"- source run: `{source_display}`\n")
        f.write("- warmup excluded; 5 measured inferences used\n")
        f.write("- phase energies are integrated from raw telemetry over each phase window\n\n")
        f.write("| phase | rail | mean energy (J) | std (J) | mean duration (ms) |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in summary.itertuples(index=False):
            std_val = 0.0 if isinstance(row.energy_std_j, float) and math.isnan(row.energy_std_j) else row.energy_std_j
            f.write(
                f"| {row.phase_label} | {row.rail} | {row.energy_mean_j:.3f} | "
                f"{std_val:.3f} | {row.duration_mean_ms:.3f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-phase energy plots for the default control trace.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    energy_df, summary = build_phase_energy(args.source_dir)
    energy_df.to_csv(args.out_dir / "phase_energy_per_inference.csv", index=False)
    summary.to_csv(args.out_dir / "phase_energy_summary.csv", index=False)
    plot_grouped_energy(summary, args.out_dir)
    plot_energy_composition(summary, args.out_dir)
    write_summary(summary, args.source_dir, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
