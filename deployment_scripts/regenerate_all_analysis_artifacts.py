#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T")
CPU_ROOT = ROOT / "thor_measurements/cpu_sweep_vla_auto"
GPU_ROOT = ROOT / "thor_measurements/gpu_sweep_vla_auto"
EMC_ROOT = ROOT / "thor_measurements/emc_sweep_vla_auto"
COMBO_ROOT = ROOT / "thor_measurements/combo_sweep_vla_auto_20260427_091937"

TOP_PHASES = ["VLA/ViT", "VLA/LLM", "VLA/action_head"]
PHASE_LABELS = {
    "VLA/ViT": "ViT",
    "VLA/LLM": "LLM",
    "VLA/action_head": "Action",
}
PHASE_COLORS = {
    "VLA/ViT": "#2ca02c",
    "VLA/LLM": "#1f77b4",
    "VLA/action_head": "#ff7f0e",
}
_OBSERVED_FREQ_CACHE: dict[Path, dict[str, float]] = {}


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def fmt_freq(freq_hz: float | int) -> str:
    freq_hz = float(freq_hz)
    if freq_hz < 0:
        return "default"
    if freq_hz >= 1e9:
        s = f"{freq_hz / 1e9:.3f}".rstrip("0").rstrip(".")
        return f"{s}GHz"
    s = f"{freq_hz / 1e6:.1f}".rstrip("0").rstrip(".")
    return f"{s}MHz"


def read_summary(summary_csv: Path) -> pd.DataFrame:
    return pd.read_csv(summary_csv)


def resolve_data_path(summary_csv: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        ROOT / path,
        summary_csv.parent / path,
        summary_csv.parents[1] / path if len(summary_csv.parents) > 1 else Path(),
        summary_csv.parents[2] / path if len(summary_csv.parents) > 2 else Path(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return summary_csv.parent / path


def phase_metrics_path_from_row(summary_csv: Path, row: pd.Series) -> Path | None:
    phase_csv = str(row.get("phase_metrics_csv", "") or "")
    if phase_csv:
        phase_path = resolve_data_path(summary_csv, phase_csv)
        if phase_path.exists():
            return phase_path
    raw_csv = str(row.get("raw_csv", "") or "")
    if raw_csv:
        raw_path = resolve_data_path(summary_csv, raw_csv)
        fallback = raw_path.parent / "phase_metrics.csv"
        if fallback.exists():
            return fallback
    return None


def read_phase_metric_means(phase_metrics_csv: Path, metric_col: str) -> dict[str, float]:
    with phase_metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[float]] = {}
    for row in rows:
        try:
            inference_id = int(row["inference_id"])
        except (KeyError, ValueError):
            continue
        if inference_id <= 1:
            continue
        phase = row.get("phase", "")
        if not phase:
            continue
        grouped.setdefault(phase, [])
        try:
            value = float(row[metric_col])
        except (KeyError, ValueError):
            value = float("nan")
        grouped[phase].append(value)
    return {phase: finite_mean(vals) for phase, vals in grouped.items()}


def build_phase_points(
    root: Path,
    mode: str,
    metric_col: str,
    default_fallback_phase_metrics: Path | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    summary_csv = root / "summary.csv"
    df = read_summary(summary_csv)
    points: list[tuple[str, float, dict[str, float]]] = []

    default_rows = df[df["mode"] == "default"]
    if not default_rows.empty:
        row = default_rows.iloc[0]
        phase_path = phase_metrics_path_from_row(summary_csv, row)
        if (phase_path is None or not phase_path.exists()) and default_fallback_phase_metrics is not None:
            phase_path = default_fallback_phase_metrics
        if phase_path is not None and phase_path.exists():
            points.append(("default", -1.0, read_phase_metric_means(phase_path, metric_col)))

    for _, row in df[df["mode"] != "default"].iterrows():
        phase_path = phase_metrics_path_from_row(summary_csv, row)
        if phase_path is None or not phase_path.exists():
            continue
        if mode == "cpu":
            freq_hz = float(row.get("requested_hz", row.get("requested_cpu_hz", -1)))
        elif mode == "gpu":
            freq_hz = float(row.get("requested_hz", row.get("requested_gpu_hz", -1)))
        else:
            freq_hz = float(row.get("requested_hz", row.get("requested_emc_hz", -1)))
        label = fmt_freq(freq_hz)
        points.append((label, freq_hz, read_phase_metric_means(phase_path, metric_col)))

    dedup: dict[str, tuple[float, dict[str, float]]] = {}
    for label, freq_hz, metrics in points:
        dedup[label] = (freq_hz, metrics)

    out: list[tuple[str, float, dict[str, float]]] = []
    if "default" in dedup:
        out.append(("default", -1.0, dedup["default"][1]))
    for label, (freq_hz, metrics) in sorted(
        ((k, v) for k, v in dedup.items() if k != "default"),
        key=lambda item: item[1][0],
    ):
        out.append((label, freq_hz, metrics))
    return out


def plot_phase_lines(points: list[tuple[str, float, dict[str, float]]], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(points) * 0.8), 6))
    x = np.arange(len(points))
    xlabels = [label for label, _, _ in points]

    for phase in TOP_PHASES:
        vals = [metrics.get(phase, float("nan")) for _, _, metrics in points]
        ax.plot(x, vals, marker="o", linewidth=2, label=PHASE_LABELS[phase], color=PHASE_COLORS[phase])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=32, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single_axis_summary(
    df: pd.DataFrame,
    freq_col: str,
    out_path: Path,
    kind: str,
) -> None:
    default_row = df[df["mode"] == "default"].iloc[0] if not df[df["mode"] == "default"].empty else None
    locked = df[df["mode"] != "default"].copy()
    locked = locked.sort_values(freq_col)
    labels = [fmt_freq(v) for v in locked[freq_col]]
    x = np.arange(len(locked))

    if kind == "cpu":
        metrics = [
            ("vin_power_mean_w", "VIN Mean Power (W)", "CPU Sweep: Mean VIN Power"),
            ("cpu_soc_mss_power_mean_w", "CPU_SOC_MSS Mean Power (W)", "CPU Sweep: Mean CPU_SOC_MSS Power"),
            ("gpu_power_mean_w", "GPU Mean Power (W)", "CPU Sweep: Mean GPU Power"),
            ("vin_energy_j", "VIN Energy (J)", "CPU Sweep: Total VIN Energy"),
        ]
        default_cols = ["vin_power_mean_w", "cpu_soc_mss_power_mean_w", "gpu_power_mean_w", "vin_energy_j"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    else:
        metrics = [
            ("vin_power_mean_w", "VIN Mean Power (W)", "EMC Sweep: Mean VIN Power"),
            ("cpu_soc_mss_power_mean_w", "CPU_SOC_MSS Mean Power (W)", "EMC Sweep: Mean CPU_SOC_MSS Power"),
            ("gpu_power_mean_w", "GPU Mean Power (W)", "EMC Sweep: Mean GPU Power"),
            ("vin_energy_j", "VIN Energy (J)", "EMC Sweep: Total VIN Energy"),
        ]
        default_cols = ["vin_power_mean_w", "cpu_soc_mss_power_mean_w", "gpu_power_mean_w", "vin_energy_j"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(12, len(labels) * 0.8), 14), sharex=True)
    for ax, (metric, ylabel, title), color, dcol in zip(axes, metrics, colors, default_cols):
        y = locked[metric]
        ax.plot(x, y, marker="o", linewidth=2, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)
        if default_row is not None and pd.notna(default_row[dcol]):
            ax.axhline(float(default_row[dcol]), linestyle="--", alpha=0.5, label="default", color=color)
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=32, ha="right")
    axes[-1].set_xlabel(f"{kind.upper()} Frequency")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gpu_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    default_row = df[df["mode"] == "default"].iloc[0] if not df[df["mode"] == "default"].empty else None
    locked = df[df["mode"] != "default"].copy().sort_values("actual_hz")
    labels = [fmt_freq(v) for v in locked["actual_hz"]]
    x = np.arange(len(locked))

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(labels) * 0.8), 12), sharex=True)
    specs = [
        ("gpu_power_mean_w", "GPU Mean Power (W)", "GPU Sweep: Mean GPU Power"),
        ("gpu_energy_j", "GPU Energy (J)", "GPU Sweep: Total GPU Energy"),
        ("duration_s", "Run Duration (s)", "GPU Sweep: Run Duration"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, specs):
        y = locked[metric]
        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, y, alpha=0.15)
        if default_row is not None and metric in default_row and pd.notna(default_row[metric]):
            ax.axhline(float(default_row[metric]), linestyle="--", alpha=0.5, label="default")
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=32, ha="right")
    axes[-1].set_xlabel("GPU Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "gpu_sweep_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.plot(x, locked["gpu_power_mean_w"], marker="o", label="mean")
    ax.plot(x, locked["gpu_power_p95_w"], marker="o", label="p95")
    ax.plot(x, locked["gpu_power_max_w"], marker="o", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("GPU Power (W)")
    ax.set_xlabel("GPU Frequency")
    ax.set_title("GPU Power Statistics vs Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gpu_sweep_power_stats.png", dpi=180)
    plt.close(fig)


def percent_improve(new: float, baseline: float, lower_is_better: bool = True) -> float:
    if baseline == 0 or not math.isfinite(new) or not math.isfinite(baseline):
        return float("nan")
    if lower_is_better:
        return (baseline - new) / baseline * 100.0
    return (new - baseline) / baseline * 100.0


def compute_pareto(latencies: np.ndarray, energies: np.ndarray) -> np.ndarray:
    keep = np.ones(len(latencies), dtype=bool)
    for i in range(len(latencies)):
        for j in range(len(latencies)):
            if i == j:
                continue
            if latencies[j] <= latencies[i] and energies[j] <= energies[i] and (
                latencies[j] < latencies[i] or energies[j] < energies[i]
            ):
                keep[i] = False
                break
    return keep


def load_combo_phase_metric(combo_root: Path, phase: str, metric_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for run_dir in sorted(p for p in combo_root.iterdir() if p.is_dir()):
        phase_metrics = run_dir / "phase_metrics.csv"
        if not phase_metrics.exists():
            continue
        metrics = read_phase_metric_means(phase_metrics, metric_col)
        if phase in metrics:
            out[run_dir.name] = metrics[phase]
    return out


def label_box(ax, x, y, text, dx=8, dy=8, ha="left"):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="0.5"),
        arrowprops=dict(arrowstyle="-", color="0.35", lw=1.0),
    )


def load_observed_freqs(raw_csv: Path) -> dict[str, float]:
    cached = _OBSERVED_FREQ_CACHE.get(raw_csv)
    if cached is not None:
        return cached
    df = pd.read_csv(raw_csv)
    out = {
        "cpu_hz": float("nan"),
        "gpu_hz": float("nan"),
        "emc_hz": float("nan"),
    }
    if "cpu_cur_freq_mean_khz" in df.columns:
        vals = pd.to_numeric(df["cpu_cur_freq_mean_khz"], errors="coerce").dropna()
        if not vals.empty:
            out["cpu_hz"] = float(vals.median()) * 1000.0
    if "gpu_cur_freq_hz" in df.columns:
        vals = pd.to_numeric(df["gpu_cur_freq_hz"], errors="coerce").dropna()
        if not vals.empty:
            out["gpu_hz"] = float(vals.median())
    if "emc_rate_hz" in df.columns:
        vals = pd.to_numeric(df["emc_rate_hz"], errors="coerce").dropna()
        if not vals.empty:
            out["emc_hz"] = float(vals.median())
    _OBSERVED_FREQ_CACHE[raw_csv] = out
    return out


def display_freq_hz(summary_csv: Path, row: pd.Series, axis: str) -> float:
    requested_col = f"requested_{axis}_hz"
    actual_col = f"actual_{axis}_hz"
    fixed_col = f"fixed_{axis}_hz"
    observed_key = f"{axis}_hz"

    for col in (requested_col, actual_col, fixed_col):
        if col in row:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value) and value > 0:
                return value

    if axis == "cpu" and "cpu_freq_mean_hz" in row:
        try:
            value = float(row["cpu_freq_mean_hz"])
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value) and value > 0:
            return value

    raw_csv = str(row.get("raw_csv", "") or "")
    if raw_csv:
        raw_path = resolve_data_path(summary_csv, raw_csv)
        if raw_path.exists():
            observed = load_observed_freqs(raw_path)
            value = observed.get(observed_key, float("nan"))
            if math.isfinite(value) and value > 0:
                return value

    return -1.0


def combo_freq_label(row: pd.Series, title: str) -> str:
    summary_csv = Path(str(row.get("_summary_csv", ""))) if row.get("_summary_csv", "") else ROOT / "summary.csv"
    cpu = fmt_freq(display_freq_hz(summary_csv, row, "cpu"))
    gpu = fmt_freq(display_freq_hz(summary_csv, row, "gpu"))
    emc = fmt_freq(display_freq_hz(summary_csv, row, "emc"))
    return f"{title}\nCPU {cpu}\nGPU {gpu}\nEMC {emc}"


def generate_combo_analysis(combo_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_all = pd.read_csv(combo_root / "summary.csv")
    summary_all["_summary_csv"] = str(combo_root / "summary.csv")
    summary_all["run_dir"] = summary_all["combo_name"].fillna(summary_all["mode"])
    get_action_latency = load_combo_phase_metric(combo_root, "VLA/get_action", "latency_ms")
    summary_all["get_action_latency_ms"] = summary_all["run_dir"].map(get_action_latency)
    default_all = summary_all[summary_all["mode"] == "default"].iloc[0]
    if pd.isna(default_all["get_action_latency_ms"]):
        default_all = default_all.copy()
        default_all["get_action_latency_ms"] = read_phase_metric_means(
            combo_root / "default" / "phase_metrics.csv",
            "latency_ms",
        ).get("VLA/get_action", float("nan"))
    summary = summary_all[pd.notna(summary_all["get_action_latency_ms"])].copy()

    default = default_all
    summary["edp"] = summary["get_action_latency_ms"] * summary["vin_energy_j"]
    summary["latency_improve_pct"] = summary["get_action_latency_ms"].apply(
        lambda v: percent_improve(v, float(default["get_action_latency_ms"]), True)
    )
    summary["energy_improve_pct"] = summary["vin_energy_j"].apply(
        lambda v: percent_improve(v, float(default["vin_energy_j"]), True)
    )
    ranked = summary.sort_values(["edp", "get_action_latency_ms", "vin_energy_j"]).copy()
    ranked.to_csv(out_dir / "overall_ranked.csv", index=False)

    best_latency = summary.loc[summary["get_action_latency_ms"].idxmin()]
    best_energy = summary.loc[summary["vin_energy_j"].idxmin()]
    best_tradeoff = summary.loc[summary["edp"].idxmin()]

    lat = summary["get_action_latency_ms"].to_numpy()
    ene = summary["vin_energy_j"].to_numpy()
    keep = compute_pareto(lat, ene)
    pareto = summary.loc[keep].sort_values("get_action_latency_ms").copy()
    pareto.to_csv(out_dir / "pareto_front.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(summary["get_action_latency_ms"], summary["vin_energy_j"], s=25, c="0.75", alpha=0.8)
    ax.plot(pareto["get_action_latency_ms"], pareto["vin_energy_j"], "-o", color="crimson", lw=2, ms=5)
    ax.scatter([default["get_action_latency_ms"]], [default["vin_energy_j"]], c="royalblue", s=45, zorder=5)
    ax.scatter([best_latency["get_action_latency_ms"]], [best_latency["vin_energy_j"]], c="darkgreen", s=45, zorder=5)
    ax.scatter([best_energy["get_action_latency_ms"]], [best_energy["vin_energy_j"]], c="purple", s=45, zorder=5)
    ax.scatter([best_tradeoff["get_action_latency_ms"]], [best_tradeoff["vin_energy_j"]], c="darkorange", s=45, zorder=5)
    label_box(
        ax,
        default["get_action_latency_ms"],
        default["vin_energy_j"],
        "Default",
        dx=-8,
        dy=18,
        ha="right",
    )
    label_box(
        ax,
        best_latency["get_action_latency_ms"],
        best_latency["vin_energy_j"],
        combo_freq_label(best_latency, "Best latency"),
        dx=20,
        dy=104,
    )
    label_box(
        ax,
        best_tradeoff["get_action_latency_ms"],
        best_tradeoff["vin_energy_j"],
        combo_freq_label(best_tradeoff, "Best trade-off"),
        dx=96,
        dy=28,
    )
    label_box(
        ax,
        best_energy["get_action_latency_ms"],
        best_energy["vin_energy_j"],
        combo_freq_label(best_energy, "Best energy"),
        dx=138,
        dy=6,
    )
    ax.set_xlabel("VLA/get_action Latency (ms)")
    ax.set_ylabel("Total VIN Energy (J)")
    ax.set_title("Pareto Front: Latency vs Total VIN Energy")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_latency_energy.png", dpi=180)
    plt.close(fig)

    phase_best_rows = []
    phase_pareto_panels = []
    for phase in TOP_PHASES:
        lat_map = load_combo_phase_metric(combo_root, phase, "latency_ms")
        gpu_power_map = load_combo_phase_metric(combo_root, phase, "gpu_avg_power_w")
        gpu_energy_map = load_combo_phase_metric(combo_root, phase, "gpu_energy_j")
        vin_power_map = load_combo_phase_metric(combo_root, phase, "vin_avg_power_w")
        vin_energy_map = load_combo_phase_metric(combo_root, phase, "vin_energy_j")

        phase_df = summary[["run_dir", "combo_name", "mode", "requested_cpu_hz", "requested_gpu_hz", "requested_emc_hz"]].copy()
        phase_df["latency_ms"] = phase_df["run_dir"].map(lat_map)
        phase_df["vin_energy_j"] = phase_df["run_dir"].map(vin_energy_map)
        phase_df["vin_power_w"] = phase_df["run_dir"].map(vin_power_map)
        phase_df["gpu_energy_j"] = phase_df["run_dir"].map(gpu_energy_map)
        phase_df["gpu_power_w"] = phase_df["run_dir"].map(gpu_power_map)
        phase_df["edp"] = phase_df["latency_ms"] * phase_df["vin_energy_j"]
        phase_df = phase_df[pd.notna(phase_df["latency_ms"])].copy()

        finite = phase_df[pd.notna(phase_df["vin_energy_j"])].copy()
        if not finite.empty:
            best_lat = finite.loc[finite["latency_ms"].idxmin()]
            best_en = finite.loc[finite["vin_energy_j"].idxmin()]
            best_edp = finite.loc[finite["edp"].idxmin()]
            phase_best_rows.extend([
                {"phase": phase, "criterion": "latency", **best_lat.to_dict()},
                {"phase": phase, "criterion": "energy", **best_en.to_dict()},
                {"phase": phase, "criterion": "edp", **best_edp.to_dict()},
            ])

            keep_phase = compute_pareto(finite["latency_ms"].to_numpy(), finite["vin_energy_j"].to_numpy())
            pf = finite.loc[keep_phase].sort_values("latency_ms")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(finite["latency_ms"], finite["vin_energy_j"], s=20, c="0.75", alpha=0.8)
            ax.plot(pf["latency_ms"], pf["vin_energy_j"], "-o", color="crimson", lw=2, ms=4)
            default_phase = finite[finite["mode"] == "default"].iloc[0]
            ax.scatter([default_phase["latency_ms"]], [default_phase["vin_energy_j"]], c="royalblue", s=40, zorder=5)
            ax.scatter([best_lat["latency_ms"]], [best_lat["vin_energy_j"]], c="darkgreen", s=40, zorder=5)
            ax.scatter([best_en["latency_ms"]], [best_en["vin_energy_j"]], c="purple", s=40, zorder=5)
            ax.scatter([best_edp["latency_ms"]], [best_edp["vin_energy_j"]], c="darkorange", s=40, zorder=5)
            label_box(
                ax,
                default_phase["latency_ms"],
                default_phase["vin_energy_j"],
                "Default",
                dx=8,
                dy=8,
            )
            label_box(
                ax,
                best_lat["latency_ms"],
                best_lat["vin_energy_j"],
                combo_freq_label(best_lat, "Best latency"),
                dx=10,
                dy=10,
            )
            label_box(
                ax,
                best_en["latency_ms"],
                best_en["vin_energy_j"],
                combo_freq_label(best_en, "Best energy"),
                dx=12,
                dy=22,
            )
            label_box(
                ax,
                best_edp["latency_ms"],
                best_edp["vin_energy_j"],
                combo_freq_label(best_edp, "Best trade-off"),
                dx=18,
                dy=34,
            )
            ax.set_xlabel(f"{PHASE_LABELS[phase]} Latency (ms)")
            ax.set_ylabel("Total VIN Energy (J)")
            ax.set_title(f"Pareto Front: {PHASE_LABELS[phase]} Latency vs Energy")
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            fname = {
                "VLA/ViT": "pareto_vit_latency_energy.png",
                "VLA/LLM": "pareto_llm_latency_energy.png",
                "VLA/action_head": "pareto_action_head_latency_energy.png",
            }[phase]
            plt.savefig(out_dir / fname, dpi=180)
            plt.close(fig)
            phase_pareto_panels.append((phase, finite, pf, default_phase, best_lat, best_en, best_edp))

    if phase_best_rows:
        phase_best_df = pd.DataFrame(phase_best_rows)
        phase_best_df.to_csv(out_dir / "phase_best_configs_finite.csv", index=False)

        order = ["VLA/ViT", "VLA/LLM", "VLA/action_head"]
        best_latency_rows = phase_best_df[phase_best_df["criterion"] == "latency"].set_index("phase").reindex(order)
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = best_latency_rows["latency_ms"].to_list()
        labels = [PHASE_LABELS[p] for p in order]
        colors = [PHASE_COLORS[p] for p in order]
        bars = ax.bar(labels, vals, color=colors)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Best Latency by Phase")
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "phase_best_latency.png", dpi=180)
        plt.close(fig)

    if phase_pareto_panels:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (phase, finite, pf, default_phase, best_lat, best_en, best_edp) in zip(axes, phase_pareto_panels):
            ax.scatter(finite["latency_ms"], finite["vin_energy_j"], s=16, c="0.75", alpha=0.8)
            ax.plot(pf["latency_ms"], pf["vin_energy_j"], "-o", color="crimson", lw=2, ms=3)
            ax.scatter([default_phase["latency_ms"]], [default_phase["vin_energy_j"]], c="royalblue", s=30, zorder=5)
            ax.scatter([best_lat["latency_ms"]], [best_lat["vin_energy_j"]], c="darkgreen", s=30, zorder=5)
            ax.scatter([best_en["latency_ms"]], [best_en["vin_energy_j"]], c="purple", s=30, zorder=5)
            ax.scatter([best_edp["latency_ms"]], [best_edp["vin_energy_j"]], c="darkorange", s=30, zorder=5)
            ax.set_title(PHASE_LABELS[phase])
            ax.set_xlabel("Latency (ms)")
            ax.grid(True, alpha=0.2)
        axes[0].set_ylabel("Total VIN Energy (J)")
        plt.tight_layout()
        plt.savefig(out_dir / "pareto_phase_latency_energy_3panel.png", dpi=180)
        plt.close(fig)

    summary_md = out_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Combo Analysis\n\n")
        f.write(f"- Best latency: `{best_latency['run_dir']}` ({best_latency['get_action_latency_ms']:.2f} ms)\n")
        f.write(f"- Best energy: `{best_energy['run_dir']}` ({best_energy['vin_energy_j']:.2f} J)\n")
        f.write(f"- Best trade-off: `{best_tradeoff['run_dir']}` (EDP {best_tradeoff['edp']:.2f})\n")


def generate_improvement_tables(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "cpu": CPU_ROOT / "summary.csv",
        "emc": EMC_ROOT / "summary.csv",
        "gpu": GPU_ROOT / "summary.csv",
    }
    lines = ["# Improvement vs Default", ""]
    for name, summary_csv in summaries.items():
        df = pd.read_csv(summary_csv)
        default = df[df["mode"] == "default"].iloc[0]
        rows = []
        if name == "cpu":
            freq_col = "requested_hz"
            latency_map = load_combo_phase_metric(CPU_ROOT, "VLA/get_action", "latency_ms")
            row_name = "cpufreq"
            energy_col = "vin_energy_j"
            power_col = "vin_power_mean_w"
        elif name == "emc":
            freq_col = "requested_hz"
            latency_map = load_combo_phase_metric(EMC_ROOT, "VLA/get_action", "latency_ms")
            row_name = "emcfreq"
            energy_col = "vin_energy_j"
            power_col = "vin_power_mean_w"
        else:
            freq_col = "requested_hz"
            latency_map = {}
            combo_default_phase = COMBO_ROOT / "default" / "phase_metrics.csv"
            for _, row in df.iterrows():
                mode = row["mode"]
                if mode == "default":
                    if combo_default_phase.exists():
                        latency_map["default"] = read_phase_metric_means(combo_default_phase, "latency_ms").get("VLA/get_action", float("nan"))
                else:
                    phase_path = phase_metrics_path_from_row(summary_csv, row)
                    if phase_path and phase_path.exists():
                        latency_map[fmt_freq(float(row[freq_col]))] = read_phase_metric_means(phase_path, "latency_ms").get("VLA/get_action", float("nan"))
            row_name = "gpufreq"
            energy_col = "gpu_energy_j"
            power_col = "gpu_power_mean_w"

        default_latency = latency_map["default"] if "default" in latency_map else float("nan")
        default_energy = float(default[energy_col])
        default_power = float(default[power_col])

        for _, row in df.iterrows():
            label = "default" if row["mode"] == "default" else fmt_freq(float(row[freq_col]))
            latency = latency_map.get(label, float("nan")) if name != "gpu" else latency_map.get(label, float("nan"))
            rows.append(
                {
                    "label": label,
                    "latency_ms": latency,
                    "latency_improve_pct": percent_improve(latency, default_latency, True),
                    "energy": float(row[energy_col]),
                    "energy_improve_pct": percent_improve(float(row[energy_col]), default_energy, True),
                    "power": float(row[power_col]),
                    "power_improve_pct": percent_improve(float(row[power_col]), default_power, True),
                }
            )
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_dir / f"{name}_improvement_vs_default.csv", index=False)
        lines.append(f"## {name.upper()}")
        lines.append(f"- rows: {len(out_df)}")
        best_latency = out_df[out_df["label"] != "default"].sort_values("latency_ms").iloc[0]
        best_energy = out_df[out_df["label"] != "default"].sort_values("energy").iloc[0]
        lines.append(f"- best latency: `{best_latency['label']}`")
        lines.append(f"- best energy: `{best_energy['label']}`")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate all analysis artifacts from saved Thor measurement CSVs.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "thor_measurements/regenerated_artifacts_20260429",
    )
    args = ap.parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    cpu_dir = out_root / "cpu_phase_plots_auto"
    gpu_dir = out_root / "gpu_phase_plots_auto"
    emc_dir = out_root / "emc_phase_plots_auto"
    stacked_dir = out_root / "sweep_phase_stacked_biglabels"
    combo_dir = out_root / "combo_analysis"
    tables_dir = out_root / "improvement_tables"

    cpu_df = pd.read_csv(CPU_ROOT / "summary.csv")
    plot_single_axis_summary(cpu_df, "requested_hz", cpu_dir / "cpu_sweep_summary.png", "cpu")
    emc_df = pd.read_csv(EMC_ROOT / "summary.csv")
    plot_single_axis_summary(emc_df, "requested_hz", emc_dir / "emc_sweep_summary.png", "emc")
    gpu_df = pd.read_csv(GPU_ROOT / "summary.csv")
    plot_gpu_summary(gpu_df, gpu_dir)

    combo_default_phase_metrics = COMBO_ROOT / "default" / "phase_metrics.csv"
    # CPU phase lines
    plot_phase_lines(
        build_phase_points(CPU_ROOT, "cpu", "latency_ms"),
        "CPU Sweep: Phase Latency",
        "Latency (ms)",
        cpu_dir / "cpu_phase_latency.png",
    )
    plot_phase_lines(
        build_phase_points(CPU_ROOT, "cpu", "cpu_soc_mss_avg_power_w"),
        "CPU Sweep: Phase CPU_SOC_MSS Power",
        "CPU_SOC_MSS Power (W)",
        cpu_dir / "cpu_phase_cpu_power.png",
    )
    plot_phase_lines(
        build_phase_points(CPU_ROOT, "cpu", "vin_avg_power_w"),
        "CPU Sweep: Phase VIN Power",
        "VIN Power (W)",
        cpu_dir / "cpu_phase_vin_power.png",
    )
    plot_phase_lines(
        build_phase_points(CPU_ROOT, "cpu", "vin_energy_j"),
        "CPU Sweep: Phase VIN Energy",
        "VIN Energy (J)",
        cpu_dir / "cpu_phase_vin_energy.png",
    )
    # EMC phase lines
    plot_phase_lines(
        build_phase_points(EMC_ROOT, "emc", "latency_ms"),
        "EMC Sweep: Phase Latency",
        "Latency (ms)",
        emc_dir / "emc_phase_latency.png",
    )
    plot_phase_lines(
        build_phase_points(EMC_ROOT, "emc", "vin_avg_power_w"),
        "EMC Sweep: Phase VIN Power",
        "VIN Power (W)",
        emc_dir / "emc_phase_vin_power.png",
    )
    plot_phase_lines(
        build_phase_points(EMC_ROOT, "emc", "vin_energy_j"),
        "EMC Sweep: Phase VIN Energy",
        "VIN Energy (J)",
        emc_dir / "emc_phase_vin_energy.png",
    )
    # GPU phase lines with combo fallback for default
    plot_phase_lines(
        build_phase_points(GPU_ROOT, "gpu", "latency_ms", combo_default_phase_metrics),
        "GPU Sweep: Phase Latency vs GPU Frequency",
        "Latency (ms)",
        gpu_dir / "phase_latency_vs_gpufreq.png",
    )
    plot_phase_lines(
        build_phase_points(GPU_ROOT, "gpu", "gpu_avg_power_w", combo_default_phase_metrics),
        "GPU Sweep: Phase Average GPU Power vs GPU Frequency",
        "GPU Avg Power (W)",
        gpu_dir / "phase_gpu_power_vs_gpufreq.png",
    )
    plot_phase_lines(
        build_phase_points(GPU_ROOT, "gpu", "gpu_energy_j", combo_default_phase_metrics),
        "GPU Sweep: Phase GPU Energy vs GPU Frequency",
        "GPU Energy (J)",
        gpu_dir / "phase_gpu_energy_vs_gpufreq.png",
    )

    # stacked single-axis phase plots
    subprocess.run(
        [
            "python",
            str(ROOT / "deployment_scripts/plot_single_axis_phase_stacked.py"),
            "--cpu-root",
            str(CPU_ROOT),
            "--gpu-root",
            str(GPU_ROOT),
            "--emc-root",
            str(EMC_ROOT),
            "--combo-root",
            str(COMBO_ROOT),
            "--out-dir",
            str(stacked_dir),
        ],
        check=True,
    )

    generate_combo_analysis(COMBO_ROOT, combo_dir)
    generate_improvement_tables(tables_dir)

    manifest = out_root / "README.md"
    manifest.write_text(
        "\n".join(
            [
                "# Regenerated Analysis Artifacts",
                "",
                f"- CPU plots: `{cpu_dir}`",
                f"- GPU plots: `{gpu_dir}`",
                f"- EMC plots: `{emc_dir}`",
                f"- Stacked single-axis plots: `{stacked_dir}`",
                f"- Combo analysis: `{combo_dir}`",
                f"- Improvement tables: `{tables_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote regenerated artifacts under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
