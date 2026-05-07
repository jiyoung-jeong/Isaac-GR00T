#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE_ORDER = [
    ("VLA/ViT", "ViT", "#2ca02c"),
    ("VLA/LLM", "LLM", "#1f77b4"),
    ("VLA/action_head", "Action", "#ff7f0e"),
]

METRIC_CONFIG = {
    "latency": ("latency_ms", "Latency (ms)"),
    "cpu_power": ("cpu_soc_mss_avg_power_w", "Mean CPU_SOC_MSS Power (W)"),
    "cpu_energy": ("cpu_soc_mss_energy_j", "Mean CPU_SOC_MSS Energy (J)"),
    "gpu_power": ("gpu_avg_power_w", "Mean GPU Power (W)"),
    "gpu_energy": ("gpu_energy_j", "Mean GPU Energy (J)"),
    "emc_power": ("vin_avg_power_w", "Mean VIN Power (W)"),
    "emc_energy": ("vin_energy_j", "Mean VIN Energy (J)"),
}


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def fmt_freq(freq_hz: float) -> str:
    if freq_hz < 0:
        return "default"
    if freq_hz >= 1e9:
        s = f"{freq_hz / 1e9:.3f}".rstrip("0").rstrip(".")
        return f"{s}GHz"
    s = f"{freq_hz / 1e6:.1f}".rstrip("0").rstrip(".")
    return f"{s}MHz"


def read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_data_path(summary_csv: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        summary_csv.parent / path,
        summary_csv.parents[1] / path if len(summary_csv.parents) > 1 else Path(),
        summary_csv.parents[2] / path if len(summary_csv.parents) > 2 else Path(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return summary_csv.parent / path


def read_phase_metric_means(phase_metrics_csv: Path, metric_col: str) -> dict[str, float]:
    with phase_metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[float]] = {phase: [] for phase, *_ in PHASE_ORDER}
    for row in rows:
        try:
            inference_id = int(row["inference_id"])
        except (KeyError, ValueError):
            continue
        if inference_id <= 1:
            continue
        phase = row.get("phase", "")
        if phase not in grouped:
            continue
        try:
            value = float(row[metric_col])
        except (KeyError, ValueError):
            value = float("nan")
        grouped[phase].append(value)

    return {phase: finite_mean(vals) for phase, vals in grouped.items()}


def phase_metrics_path_from_row(summary_csv: Path, row: dict[str, str]) -> Path | None:
    phase_csv = row.get("phase_metrics_csv", "")
    if phase_csv:
        phase_path = resolve_data_path(summary_csv, phase_csv)
        if phase_path.exists():
            return phase_path

    raw_csv = row.get("raw_csv", "")
    if raw_csv:
        raw_path = resolve_data_path(summary_csv, raw_csv)
        fallback = raw_path.parent / "phase_metrics.csv"
        if fallback.exists():
            return fallback
    return None


def build_points(
    summary_csv: Path,
    mode: str,
    metric_key: str,
    default_fallback_phase_metrics: Path | None = None,
) -> list[tuple[str, dict[str, float]]]:
    metric_col, _ = METRIC_CONFIG[metric_key]
    rows = read_summary_rows(summary_csv)
    points: list[tuple[str, float, dict[str, float]]] = []

    default_added = False
    for row in rows:
        if row.get("mode") == "default":
            phase_path = phase_metrics_path_from_row(summary_csv, row)
            if phase_path is None or not phase_path.exists():
                phase_path = default_fallback_phase_metrics
            if phase_path is not None and phase_path.exists():
                points.append(("default", -1.0, read_phase_metric_means(phase_path, metric_col)))
                default_added = True
            break

    for row in rows:
        if row.get("mode") == "default":
            continue
        phase_path = phase_metrics_path_from_row(summary_csv, row)
        if phase_path is None or not phase_path.exists():
            continue

        if mode == "cpu":
            freq_hz = float(row.get("requested_hz") or row.get("requested_cpu_hz") or -1)
        elif mode == "gpu":
            freq_hz = float(row.get("requested_hz") or row.get("requested_gpu_hz") or -1)
        else:
            freq_hz = float(row.get("requested_hz") or row.get("requested_emc_hz") or -1)

        if freq_hz < 0:
            continue
        label = fmt_freq(freq_hz)
        points.append((label, freq_hz, read_phase_metric_means(phase_path, metric_col)))

    dedup: dict[str, tuple[float, dict[str, float]]] = {}
    for label, freq_hz, values in points:
        dedup[label] = (freq_hz, values)

    out: list[tuple[str, dict[str, float]]] = []
    if default_added and "default" in dedup:
        out.append(("default", dedup["default"][1]))
    for label, _freq, values in sorted(
        ((label, freq, values) for label, (freq, values) in dedup.items() if label != "default"),
        key=lambda item: item[1],
    ):
        out.append((label, values))
    return out


def plot_stacked(points: list[tuple[str, dict[str, float]]], title: str, ylabel: str, out_path: Path) -> None:
    xlabels = [label for label, _ in points]
    x = np.arange(len(points))
    width = 0.88

    fig, ax = plt.subplots(figsize=(max(10, len(points) * 1.2), 7))
    bottom = np.zeros(len(points), dtype=float)

    for phase, short, color in PHASE_ORDER:
        vals = np.array([values.get(phase, float("nan")) for _, values in points], dtype=float)
        vals0 = np.nan_to_num(vals, nan=0.0)
        seg_bottom = bottom.copy()
        ax.bar(x, vals0, width=width, bottom=bottom, color=color, label=short)
        for i, val in enumerate(vals0):
            if val <= 0:
                continue
            ax.text(
                x[i],
                seg_bottom[i] + val / 2.0,
                f"{val:.2f}" if val < 10 else f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.28, edgecolor="none"),
            )
        bottom += vals0

    for i, total in enumerate(bottom):
        if total <= 0:
            continue
        ax.text(
            x[i],
            total + max(0.03 * float(np.nanmax(bottom)), 0.05),
            f"{total:.2f}" if total < 10 else f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    ax.margins(x=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=32, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot stacked single-axis Thor sweep phase metrics from phase_metrics.csv.")
    ap.add_argument("--cpu-root", type=Path, default=Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/cpu_sweep_vla_auto"))
    ap.add_argument("--gpu-root", type=Path, default=Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/gpu_sweep_vla_auto"))
    ap.add_argument("--emc-root", type=Path, default=Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/emc_sweep_vla_auto"))
    ap.add_argument("--combo-root", type=Path, default=Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/combo_sweep_vla_auto_20260427_091937"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/sweep_phase_stacked_biglabels"))
    args = ap.parse_args()

    combo_default_phase_metrics = args.combo_root / "default" / "phase_metrics.csv"

    specs = [
        ("cpu", args.cpu_root / "summary.csv", None, [
            ("latency", "CPU Sweep: Stacked Phase Latency", args.out_dir / "cpu_latency_stacked.png"),
            ("cpu_power", "CPU Sweep: Stacked Phase CPU Power", args.out_dir / "cpu_power_stacked.png"),
            ("cpu_energy", "CPU Sweep: Stacked Phase CPU Energy", args.out_dir / "cpu_energy_stacked.png"),
        ]),
        ("gpu", args.gpu_root / "summary.csv", combo_default_phase_metrics, [
            ("latency", "GPU Sweep: Stacked Phase Latency", args.out_dir / "gpu_latency_stacked.png"),
            ("gpu_power", "GPU Sweep: Stacked Phase GPU Power", args.out_dir / "gpu_power_stacked.png"),
            ("gpu_energy", "GPU Sweep: Stacked Phase GPU Energy", args.out_dir / "gpu_energy_stacked.png"),
        ]),
        ("emc", args.emc_root / "summary.csv", None, [
            ("latency", "EMC Sweep: Stacked Phase Latency", args.out_dir / "emc_latency_stacked.png"),
            ("emc_power", "EMC Sweep: Stacked Phase VIN Power", args.out_dir / "emc_power_stacked.png"),
            ("emc_energy", "EMC Sweep: Stacked Phase VIN Energy", args.out_dir / "emc_energy_stacked.png"),
        ]),
    ]

    for mode, summary_csv, default_fallback, plots in specs:
        for metric_key, title, out_path in plots:
            points = build_points(summary_csv, mode, metric_key, default_fallback)
            _metric_col, ylabel = METRIC_CONFIG[metric_key]
            plot_stacked(points, title, ylabel, out_path)
            print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
