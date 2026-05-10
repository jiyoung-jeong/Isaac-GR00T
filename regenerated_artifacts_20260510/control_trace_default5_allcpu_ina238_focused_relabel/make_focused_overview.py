#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path("/home/Thor/Workspace/jyjeong/Isaac-GR00T")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deployment_scripts import thor_telemetry_logger as telem

PHASE_COLORS = {
    "vit": "#88c999",
    "llm": "#8f93f5",
    "action_head": "#f7c66d",
}
PHASE_LABELS = {
    "vit": "ViT",
    "llm": "LLM",
    "action_head": "Action",
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


SRC = Path("/home/Thor/regenerated_artifacts_20260510/control_trace_default5_allcpu_ina238_source")
OUT = Path("/home/Thor/regenerated_artifacts_20260510/control_trace_default5_allcpu_ina238_focused_relabel")


def _plot_valid_series(ax, x, y, *, label: str, color: str, linewidth: float = 1.4, step: bool = False):
    arr = np.asarray(y, dtype=float)
    if not np.isfinite(arr).any():
        return None
    if step:
        (line,) = ax.step(x, arr, where="post", label=label, color=color, linewidth=linewidth)
    else:
        (line,) = ax.plot(x, arr, label=label, color=color, linewidth=linewidth)
    return line


def _annotate_inference_windows(ax, records: list[dict[str, int]], origin_ns: int) -> None:
    y_top = ax.get_ylim()[1]
    for rec in records:
        s_ms = (rec["infer_start_ns"] - origin_ns) / 1e6
        e_ms = (rec["infer_end_ns"] - origin_ns) / 1e6
        mid = 0.5 * (s_ms + e_ms)
        ax.axvline(s_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(e_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(mid, y_top, f"Inference {rec['inference_id']}", ha="center", va="bottom", fontsize=9)


def _annotate_phase_durations(ax, phase_df: pd.DataFrame, origin_ns: int) -> None:
    y0, y1 = ax.get_ylim()
    y = y0 + 0.91 * (y1 - y0)
    for _, row in phase_df.iterrows():
        s_ms = (row["start_ns"] - origin_ns) / 1e6
        e_ms = (row["end_ns"] - origin_ns) / 1e6
        mid = 0.5 * (s_ms + e_ms)
        phase = str(row["phase"])
        label = PHASE_LABELS.get(phase, phase)
        ax.text(
            mid,
            y,
            f"{label}\n{float(row['duration_ms']):.1f} ms",
            ha="center",
            va="top",
            fontsize=7,
            color="0.18",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            clip_on=True,
        )


def _make_focused_plot(out_dir: Path, samples_df: pd.DataFrame, phase_df: pd.DataFrame, records: list[dict[str, int]]) -> None:
    origin_ns = records[0]["infer_start_ns"]
    x_end_ms = (records[-1]["infer_end_ns"] - origin_ns) / 1e6
    telem_df = add_relative_time(samples_df, origin_ns)
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.28, label=PHASE_LABELS[p])
        for p in ["vit", "llm", "action_head"]
    ]

    cpu_policy_cols = [f"cpu{policy_id}_khz" for policy_id in telem.CPU_POLICY_IDS if f"cpu{policy_id}_khz" in telem_df.columns]
    cpu_policy_arrays = {col: valid_or_nan(telem_df[col], scale=1e6) for col in cpu_policy_cols}
    gpu_gpc_cols = [f"gpu_gpc{i}_hz" for i in range(3) if f"gpu_gpc{i}_hz" in telem_df.columns]
    gpu_gpc_arrays = {col: valid_or_nan(telem_df[col], scale=1e9) for col in gpu_gpc_cols}
    gpu_gpc_values = list(gpu_gpc_arrays.values())
    gpu_gpcs_equal = bool(gpu_gpc_values) and all(
        np.allclose(gpu_gpc_values[0], arr, equal_nan=True) for arr in gpu_gpc_values[1:]
    )
    emc_ghz = valid_or_nan(telem_df["emc_rate_hz"], scale=1e9)
    cpu_policy_colors = plt.cm.viridis(np.linspace(0.12, 0.9, max(len(cpu_policy_cols), 1)))

    gpu_e = cumulative_energy(samples_df, "gpu_power_w", origin_ns)
    cpu_e = cumulative_energy(samples_df, "cpu_power_w", origin_ns)
    vin_e = cumulative_energy(samples_df, "vin_power_w", origin_ns)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11.5), dpi=160, sharex=True)

    ax = axes[0]
    shade_phases(ax, phase_df, origin_ns)
    for color, col in zip(cpu_policy_colors, cpu_policy_cols):
        policy_id = col.replace("cpu", "").replace("_khz", "")
        label = f"CPU policy{policy_id}"
        _plot_valid_series(
            ax,
            telem_df["t_ms"],
            cpu_policy_arrays[col],
            label=f"{label} (GHz)",
            color=color,
            linewidth=1.2,
            step=True,
        )
    if gpu_gpcs_equal:
        gpu_label = "GPU " + "/".join(col.removeprefix("gpu_").removesuffix("_hz").upper() for col in gpu_gpc_cols)
        _plot_valid_series(ax, telem_df["t_ms"], gpu_gpc_values[0], label=f"{gpu_label} (GHz)", color="C1", linewidth=1.4, step=True)
    else:
        for idx, col in enumerate(gpu_gpc_cols):
            label = col.removeprefix("gpu_").removesuffix("_hz").upper()
            _plot_valid_series(ax, telem_df["t_ms"], gpu_gpc_arrays[col], label=f"GPU {label} (GHz)", color=f"C{idx + 1}", linewidth=1.4, step=True)
    for label, arr, color in [
        ("EMC", emc_ghz, "C3"),
    ]:
        _plot_valid_series(ax, telem_df["t_ms"], arr, label=f"{label} (GHz)", color=color, linewidth=1.4, step=True)
    ax.set_ylabel("Freq (GHz)")
    ax.grid(True, alpha=0.3)
    _annotate_inference_windows(ax, records, origin_ns)
    _annotate_phase_durations(ax, phase_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=phase_handles + line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        fontsize=8,
        ncol=6,
        framealpha=0.88,
    )

    ax = axes[1]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["gpu_power_w"], label="VDD_GPU (W)", color="C1", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["cpu_power_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_power_w"], label="VIN (W)", color="C0", linewidth=1.3)
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    _annotate_inference_windows(ax, records, origin_ns)
    _annotate_phase_durations(ax, phase_df, origin_ns)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.88)

    ax = axes[2]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, gpu_e["t_ms"], gpu_e["energy_j"], label="VDD_GPU (J)", color="C1", linewidth=1.4)
    _plot_valid_series(ax, cpu_e["t_ms"], cpu_e["energy_j"], label="VDD_CPU_SOC_MSS (J)", color="C2", linewidth=1.4)
    _plot_valid_series(ax, vin_e["t_ms"], vin_e["energy_j"], label="VIN (J)", color="C0", linewidth=1.4)
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (ms relative to inference 1 start)")
    ax.grid(True, alpha=0.3)
    _annotate_inference_windows(ax, records, origin_ns)
    _annotate_phase_durations(ax, phase_df, origin_ns)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.88)

    ticks = np.arange(0, int(np.floor(x_end_ms / 100.0) * 100) + 1, 100)
    for axis in axes:
        axis.set_xlim(0, x_end_ms)
        axis.set_xticks(ticks)
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.set_xlabel("Time (ms relative to inference 1 start)")
    axes[2].set_xlabel("Time (ms relative to inference 1 start)")

    fig.suptitle(
        "Default control trace — 5 measured inferences (all CPU policies, INA238 VIN)",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0)
    fig.savefig(out_dir / "trace_overview_focused.png", bbox_inches="tight")
    plt.close(fig)


def _make_frequency_power_pdf(out_dir: Path, samples_df: pd.DataFrame, phase_df: pd.DataFrame, records: list[dict[str, int]]) -> None:
    origin_ns = records[0]["infer_start_ns"]
    x_end_ms = (records[-1]["infer_end_ns"] - origin_ns) / 1e6
    telem_df = add_relative_time(samples_df, origin_ns)
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.28, label=PHASE_LABELS[p])
        for p in ["vit", "llm", "action_head"]
    ]

    cpu_policy_cols = [f"cpu{policy_id}_khz" for policy_id in telem.CPU_POLICY_IDS if f"cpu{policy_id}_khz" in telem_df.columns]
    cpu_policy_arrays = {col: valid_or_nan(telem_df[col], scale=1e6) for col in cpu_policy_cols}
    gpu_gpc_cols = [f"gpu_gpc{i}_hz" for i in range(3) if f"gpu_gpc{i}_hz" in telem_df.columns]
    gpu_gpc_arrays = {col: valid_or_nan(telem_df[col], scale=1e9) for col in gpu_gpc_cols}
    gpu_gpc_values = list(gpu_gpc_arrays.values())
    gpu_gpcs_equal = bool(gpu_gpc_values) and all(
        np.allclose(gpu_gpc_values[0], arr, equal_nan=True) for arr in gpu_gpc_values[1:]
    )
    emc_ghz = valid_or_nan(telem_df["emc_rate_hz"], scale=1e9)
    cpu_policy_colors = plt.cm.viridis(np.linspace(0.12, 0.9, max(len(cpu_policy_cols), 1)))

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.6), dpi=160, sharex=True)

    ax = axes[0]
    shade_phases(ax, phase_df, origin_ns)
    for color, col in zip(cpu_policy_colors, cpu_policy_cols):
        policy_id = col.replace("cpu", "").replace("_khz", "")
        _plot_valid_series(
            ax,
            telem_df["t_ms"],
            cpu_policy_arrays[col],
            label=f"CPU policy{policy_id} (GHz)",
            color=color,
            linewidth=1.2,
            step=True,
        )
    if gpu_gpcs_equal:
        gpu_label = "GPU " + "/".join(col.removeprefix("gpu_").removesuffix("_hz").upper() for col in gpu_gpc_cols)
        _plot_valid_series(ax, telem_df["t_ms"], gpu_gpc_values[0], label=f"{gpu_label} (GHz)", color="C1", linewidth=1.4, step=True)
    else:
        for idx, col in enumerate(gpu_gpc_cols):
            label = col.removeprefix("gpu_").removesuffix("_hz").upper()
            _plot_valid_series(ax, telem_df["t_ms"], gpu_gpc_arrays[col], label=f"GPU {label} (GHz)", color=f"C{idx + 1}", linewidth=1.4, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], emc_ghz, label="EMC (GHz)", color="C3", linewidth=1.4, step=True)
    ax.set_ylabel("Freq (GHz)")
    ax.grid(True, alpha=0.3)
    _annotate_inference_windows(ax, records, origin_ns)
    _annotate_phase_durations(ax, phase_df, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=phase_handles + line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        fontsize=8,
        ncol=6,
        framealpha=0.88,
    )

    ax = axes[1]
    shade_phases(ax, phase_df, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["gpu_power_w"], label="VDD_GPU (W)", color="C1", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["cpu_power_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_power_w"], label="VIN (W)", color="C0", linewidth=1.3)
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    _annotate_inference_windows(ax, records, origin_ns)
    _annotate_phase_durations(ax, phase_df, origin_ns)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.88)

    ticks = np.arange(0, int(np.floor(x_end_ms / 100.0) * 100) + 1, 100)
    for axis in axes:
        axis.set_xlim(0, x_end_ms)
        axis.set_xticks(ticks)
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.set_xlabel("Time (ms relative to inference 1 start)")

    fig.suptitle(
        "Default control trace - frequency and power (5 measured inferences)",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.0)
    fig.savefig(out_dir / "trace_freq_power_focused.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SRC / "telemetry_raw.csv")
    phases = pd.read_csv(SRC / "phase_ranges.csv")
    per_inf = pd.read_csv(SRC / "per_inference_summary.csv")

    records = []
    for row in per_inf.itertuples(index=False):
        inf_phases = phases[phases["inference_id"] == row.inference_id]
        infer_end_ns = int(inf_phases["end_ns"].max())
        infer_start_ns = infer_end_ns - int(float(row.duration_ms) * 1e6)
        records.append(
            {
                "inference_id": int(row.inference_id),
                "infer_start_ns": infer_start_ns,
                "infer_end_ns": infer_end_ns,
            }
        )

    start_ns = records[0]["infer_start_ns"] - int(12e6)
    end_ns = records[-1]["infer_end_ns"] + int(12e6)
    focused_samples = samples[(samples["ts_ns"] >= start_ns) & (samples["ts_ns"] <= end_ns)].copy()

    _make_focused_plot(OUT, focused_samples, phases, records)
    _make_frequency_power_pdf(OUT, focused_samples, phases, records)
    (OUT / "source_run.txt").write_text(str(SRC) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(
        "# Focused Trace Variant\n\n"
        f"- source: `{SRC}`\n"
        "- focused relabel: `trace_overview_focused.png`\n"
        "- focused frequency/power PDF: `trace_freq_power_focused.pdf`\n"
        "- VIN definition: `INA238 2-0044/hwmon*/power1_input`\n"
        "- frequency labels are shown in the legend above the frequency panel\n"
        "- ViT/LLM/Action spans include per-phase duration labels\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
