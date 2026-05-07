#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PHASES = ["VLA/ViT", "VLA/LLM", "VLA/action_head"]
PHASE_COLORS = {
    "VLA/ViT": "#88c999",
    "VLA/LLM": "#8f93f5",
    "VLA/action_head": "#f7c66d",
}
PHASE_LABELS = {
    "VLA/ViT": "ViT",
    "VLA/LLM": "LLM",
    "VLA/action_head": "Action",
}


def add_relative_time(df: pd.DataFrame, origin_ns: int) -> pd.DataFrame:
    out = df.copy()
    out["t_ms"] = (out["ts_ns"] - origin_ns) / 1e6
    return out


def cumulative_energy(samples_df: pd.DataFrame, power_col: str, origin_ns: int) -> pd.DataFrame:
    df = samples_df[["ts_ns", power_col]].copy().sort_values("ts_ns")
    power = df[power_col].to_numpy(dtype=float)
    ts = df["ts_ns"].to_numpy(dtype=np.int64)
    energy = np.zeros(len(df), dtype=float)
    if len(df) >= 2:
        for i in range(1, len(df)):
            dt = (ts[i] - ts[i - 1]) / 1e9
            pa = power[i - 1]
            pb = power[i]
            if np.isfinite(pa) and np.isfinite(pb):
                energy[i] = energy[i - 1] + 0.5 * (pa + pb) * dt
            else:
                energy[i] = energy[i - 1]
    return pd.DataFrame({"t_ms": (ts - origin_ns) / 1e6, "energy_j": energy})


def valid_or_nan(series: pd.Series, scale: float = 1.0) -> np.ndarray:
    arr = series.to_numpy(dtype=float) / scale
    arr[arr < 0] = np.nan
    return arr


def shade_phases(ax, phase_rows: pd.DataFrame, origin_ns: int) -> None:
    for _, row in phase_rows.iterrows():
        s_ms = (row["start_ns"] - origin_ns) / 1e6
        e_ms = (row["end_ns"] - origin_ns) / 1e6
        ax.axvspan(s_ms, e_ms, alpha=0.28, color=PHASE_COLORS[row["phase"]], zorder=0)


def annotate_inference_windows(ax, infer_df: pd.DataFrame, origin_ns: int) -> None:
    y_top = ax.get_ylim()[1]
    for _, row in infer_df.iterrows():
        s_ms = (row["start_ns"] - origin_ns) / 1e6
        e_ms = (row["end_ns"] - origin_ns) / 1e6
        mid = 0.5 * (s_ms + e_ms)
        ax.axvline(s_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(e_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(mid, y_top, f"Inference {int(row['inference_id'])}", ha="center", va="bottom", fontsize=9)


def subset_middle_inferences(infer_df: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    if len(infer_df) <= count:
        return infer_df.copy()
    start = max(0, (len(infer_df) - count) // 2)
    return infer_df.iloc[start : start + count].copy()


def _plot_valid_series(ax, x, y, *, label: str, color: str, linewidth: float = 1.4, step: bool = False):
    arr = np.asarray(y, dtype=float)
    if not np.isfinite(arr).any():
        return
    if step:
        ax.step(x, arr, where="post", label=label, color=color, linewidth=linewidth)
    else:
        ax.plot(x, arr, label=label, color=color, linewidth=linewidth)


def make_trace_plots(out_dir: Path, samples_df: pd.DataFrame, phase_df: pd.DataFrame, infer_df: pd.DataFrame, title: str, *, stem: str = "") -> None:
    origin_ns = int(infer_df["start_ns"].min())
    telem_df = add_relative_time(samples_df, origin_ns)

    cpu_ghz = valid_or_nan(telem_df["policy4_cur_khz"], scale=1e6)
    gpu_ghz = valid_or_nan(telem_df["gpu_gpc0_hz"], scale=1e9)
    emc_ghz = valid_or_nan(telem_df["emc_rate_hz"], scale=1e9)

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.28, label=PHASE_LABELS[p]) for p in PHASES
    ]

    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], cpu_ghz, label="CPU policy4 (GHz)", color="C2", linewidth=1.5, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], gpu_ghz, label="GPU GPC0 (GHz)", color="C1", linewidth=1.5, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], emc_ghz, label="EMC (GHz)", color="C3", linewidth=1.5, step=True)
    ax.set_xlabel("Time (ms relative to first selected inference start)")
    ax.set_ylabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")
    fig.suptitle(f"{title} — CPU/GPU/EMC frequency", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}freq_trace.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vdd_gpu_w"], label="VDD_GPU (W)", color="C1", linewidth=1.4)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vdd_cpu_soc_mss_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.4)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_w"], label="VIN (W)", color="C0", linewidth=1.4)
    ax.set_xlabel("Time (ms relative to first selected inference start)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")
    fig.suptitle(f"{title} — VDD_GPU / VDD_CPU_SOC_MSS / VIN power", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}power_trace.png", bbox_inches="tight")
    plt.close(fig)

    gpu_e = cumulative_energy(samples_df, "vdd_gpu_w", origin_ns)
    cpu_e = cumulative_energy(samples_df, "vdd_cpu_soc_mss_w", origin_ns)
    vin_e = cumulative_energy(samples_df, "vin_w", origin_ns)
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, gpu_e["t_ms"], gpu_e["energy_j"], label="VDD_GPU (J)", color="C1", linewidth=1.6)
    _plot_valid_series(ax, cpu_e["t_ms"], cpu_e["energy_j"], label="VDD_CPU_SOC_MSS (J)", color="C2", linewidth=1.6)
    _plot_valid_series(ax, vin_e["t_ms"], vin_e["energy_j"], label="VIN (J)", color="C0", linewidth=1.6)
    ax.set_xlabel("Time (ms relative to first selected inference start)")
    ax.set_ylabel("Cumulative energy (J)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper left")
    fig.suptitle(f"{title} — cumulative VDD_GPU / VDD_CPU_SOC_MSS / VIN energy", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}energy_trace.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), dpi=160, sharex=True)
    ax = axes[0]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], cpu_ghz, label="CPU policy4 (GHz)", color="C2", linewidth=1.4, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], gpu_ghz, label="GPU GPC0 (GHz)", color="C1", linewidth=1.4, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], emc_ghz, label="EMC (GHz)", color="C3", linewidth=1.4, step=True)
    ax.set_ylabel("Freq (GHz)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")

    ax = axes[1]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vdd_gpu_w"], label="VDD_GPU (W)", color="C1", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vdd_cpu_soc_mss_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_w"], label="VIN (W)", color="C0", linewidth=1.3)
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    ax.legend(loc="upper right")

    ax = axes[2]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, gpu_e["t_ms"], gpu_e["energy_j"], label="VDD_GPU (J)", color="C1", linewidth=1.4)
    _plot_valid_series(ax, cpu_e["t_ms"], cpu_e["energy_j"], label="VDD_CPU_SOC_MSS (J)", color="C2", linewidth=1.4)
    _plot_valid_series(ax, vin_e["t_ms"], vin_e["energy_j"], label="VIN (J)", color="C0", linewidth=1.4)
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (ms relative to first selected inference start)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, infer_df, origin_ns)
    ax.legend(loc="upper left")
    fig.suptitle(f"{title} — consecutive measured inferences", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}trace_overview.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, infer_df: pd.DataFrame, phase_df: pd.DataFrame, selection_desc: str) -> None:
    phase_wide = (
        phase_df.pivot(index="inference_id", columns="phase", values="latency_ms")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged = infer_df.merge(phase_wide, on="inference_id", how="left")
    merged = merged.rename(
        columns={
            "latency_ms": "duration_ms",
            "gpu_energy_j": "gpu_energy_j",
            "cpu_soc_mss_energy_j": "cpu_energy_j",
            "vin_energy_j": "vin_energy_j",
            "VLA/ViT": "vit_ms",
            "VLA/LLM": "llm_ms",
            "VLA/action_head": "action_head_ms",
        }
    )
    merged["backbone_ms"] = merged[["vit_ms", "llm_ms"]].sum(axis=1)
    merged.to_csv(out_dir / "per_inference_summary.csv", index=False)

    metric_cols = ["duration_ms", "vit_ms", "llm_ms", "backbone_ms", "action_head_ms", "gpu_energy_j", "cpu_energy_j", "vin_energy_j"]
    with (out_dir / "trace_stats.md").open("w", encoding="utf-8") as f:
        f.write("# Benchmark Trace Stats\n\n")
        f.write(f"- selection: `{selection_desc}`\n")
        f.write("- note: inference 1 is excluded to avoid startup/cold-start skew.\n\n")
        f.write("## Per-inference\n\n")
        cols = list(merged.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in merged.itertuples(index=False):
            values = []
            for v in row:
                if isinstance(v, (float, np.floating)) and math.isfinite(v):
                    values.append(f"{v:.3f}")
                else:
                    values.append(str(v))
            f.write("| " + " | ".join(values) + " |\n")
        f.write("\n\n## Aggregate\n\n")
        f.write("| metric | mean | std | median |\n")
        f.write("|---|---:|---:|---:|\n")
        for col in metric_cols:
            vals = pd.to_numeric(merged[col], errors="coerce").dropna()
            if vals.empty:
                continue
            f.write(f"| {col} | {vals.mean():.3f} | {vals.std(ddof=0):.3f} | {vals.median():.3f} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot detailed benchmark traces from an existing default run.")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--title", required=True, type=str)
    parser.add_argument("--start_inference_id", type=int, default=2)
    parser.add_argument("--num_inferences", type=int, default=5)
    parser.add_argument("--padding_ms", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry = pd.read_csv(run_dir / "telemetry_raw.csv")
    phase_metrics = pd.read_csv(run_dir / "phase_metrics.csv")

    infer_ids = list(range(args.start_inference_id, args.start_inference_id + args.num_inferences))
    infer_df = phase_metrics[(phase_metrics["phase"] == "VLA/get_action") & (phase_metrics["inference_id"].isin(infer_ids))].copy()
    infer_df = infer_df.sort_values("inference_id")
    phase_df = phase_metrics[(phase_metrics["phase"].isin(PHASES)) & (phase_metrics["inference_id"].isin(infer_ids))].copy()
    phase_df = phase_df.rename(columns={"phase": "phase", "latency_ms": "latency_ms"})

    start_ns = int(infer_df["start_ns"].min() - args.padding_ms * 1e6)
    end_ns = int(infer_df["end_ns"].max() + args.padding_ms * 1e6)
    samples_df = telemetry[(telemetry["ts_ns"] >= start_ns) & (telemetry["ts_ns"] <= end_ns)].copy()

    phase_df[["inference_id", "phase", "start_ns", "end_ns", "latency_ms"]].to_csv(out_dir / "phase_ranges.csv", index=False)
    samples_df.to_csv(out_dir / "telemetry_raw.csv", index=False)

    selection_desc = f"inference_ids={infer_ids[0]}..{infer_ids[-1]}"
    make_trace_plots(out_dir, samples_df, phase_df[["inference_id", "phase", "start_ns", "end_ns", "latency_ms"]], infer_df, args.title)
    write_summary(out_dir, infer_df, phase_df[["inference_id", "phase", "start_ns", "end_ns", "latency_ms", "gpu_energy_j", "cpu_soc_mss_energy_j", "vin_energy_j"]], selection_desc)

    middle_infer = subset_middle_inferences(infer_df, count=3)
    if len(middle_infer) >= 2:
        middle_ids = set(middle_infer["inference_id"].tolist())
        middle_start_ns = int(middle_infer["start_ns"].min() - args.padding_ms * 1e6)
        middle_end_ns = int(middle_infer["end_ns"].max() + args.padding_ms * 1e6)
        middle_samples = telemetry[(telemetry["ts_ns"] >= middle_start_ns) & (telemetry["ts_ns"] <= middle_end_ns)].copy()
        middle_phases = phase_df[phase_df["inference_id"].isin(middle_ids)].copy()
        make_trace_plots(
            out_dir,
            middle_samples,
            middle_phases[["inference_id", "phase", "start_ns", "end_ns", "latency_ms"]],
            middle_infer,
            args.title + " (middle inferences)",
            stem="middle3_",
        )


if __name__ == "__main__":
    main()
