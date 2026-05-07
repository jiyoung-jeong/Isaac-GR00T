#!/usr/bin/env python3
"""
Trace two or more consecutive control-scenario inferences with raw Thor telemetry.

This script is for "control scenarios" where we want a detailed time-series view:
  - pre-inference idle window
  - consecutive inferences
  - post-inference idle window

It reuses the same input/data path as benchmark_input_sweep.py, but instead of
aggregating many iterations it records raw per-sample telemetry and per-phase
timestamps for a small number of consecutive inferences.

Outputs:
  - telemetry_raw.csv
  - phase_ranges.csv
  - per_inference_summary.csv
  - freq_trace.png
  - power_trace.png
  - energy_trace.png
  - trace_overview.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from deployment_scripts import thor_telemetry_logger as telem
import gr00t.utils.nvtx as nvtx_utils
from scripts.deployment.benchmark_input_sweep import (
    EmbodimentTag,
    Gr00tPolicy,
    LeRobotEpisodeLoader,
    PRESET_CONFIGS,
    build_power_sampler,
    build_observation,
    mutate_observation,
    parse_config_spec,
    parse_image_size,
    parse_view_config_spec,
    prepare_model_inputs,
    replace_dit_with_tensorrt,
    sample_power_row,
    set_policy_view_keys,
    set_seed,
    settle_freq_config,
    unlock_all_freqs,
)


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


def _telemetry_sample_dict(ina3221, gpu_ch, cpu_ch, vin_ch) -> dict[str, float]:
    row = telem._sample()
    power_row = sample_power_row(ina3221, gpu_ch, cpu_ch, vin_ch)
    return {
        "ts_ns": int(row[0]),
        "gpu_power_w": float(power_row["gpu_power_w"]),
        "cpu_power_w": float(power_row["cpu_soc_mss_power_w"]),
        "vin_power_w": float(power_row["vin_power_w"]),
        "gpu_cur_freq_hz": int(row[4]),
        "cpu0_khz": int(row[5]),
        "cpu4_khz": int(row[6]),
        "gpu_gpc0_hz": int(row[7]),
        "gpu_gpc1_hz": int(row[8]),
        "gpu_gpc2_hz": int(row[9]),
        "gpu_sys_hz": int(row[10]),
        "gpu_nvd_hz": int(row[11]),
        "emc_rate_hz": int(row[12]),
    }


def telemetry_loop(stop_flag: list[bool], samples: list[dict[str, float]], interval_s: float, ina3221, gpu_ch, cpu_ch, vin_ch) -> None:
    while not stop_flag[0]:
        samples.append(_telemetry_sample_dict(ina3221, gpu_ch, cpu_ch, vin_ch))
        time.sleep(interval_s)


def _choose_config(args: argparse.Namespace):
    if args.config:
        return parse_config_spec(args.config)
    preset = PRESET_CONFIGS[args.config_preset]
    for cfg in preset:
        if cfg.name == args.config_name:
            return cfg
    raise ValueError(f"Config name '{args.config_name}' not found in preset '{args.config_preset}'")


def _choose_policy(args: argparse.Namespace, embodiment: EmbodimentTag) -> Gr00tPolicy:
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=embodiment,
        device="cuda",
        strict=True,
    )
    if args.inference_mode == "tensorrt":
        if not args.trt_engine_path:
            raise ValueError("--trt_engine_path is required for tensorrt mode")
        replace_dit_with_tensorrt(policy, args.trt_engine_path)
    elif args.inference_mode == "compile":
        policy.model.action_head.model.forward = torch.compile(
            policy.model.action_head.model.forward,
            mode="max-autotune",
        )
    return policy


def trace_single_inference(policy: Gr00tPolicy, observation: dict[str, Any], inference_id: int) -> dict[str, int]:
    rec: dict[str, int] = {"inference_id": inference_id}

    rec["infer_start_ns"] = time.monotonic_ns()

    rec["data_processing_start_ns"] = rec["infer_start_ns"]
    collated_inputs = prepare_model_inputs(policy, observation)
    rec["data_processing_end_ns"] = time.monotonic_ns()

    torch.cuda.synchronize()
    rec["backbone_start_ns"] = time.monotonic_ns()
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()
    rec["backbone_end_ns"] = time.monotonic_ns()

    torch.cuda.synchronize()
    rec["action_head_start_ns"] = time.monotonic_ns()
    with torch.inference_mode():
        _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
    torch.cuda.synchronize()
    rec["action_head_end_ns"] = time.monotonic_ns()

    rec["infer_end_ns"] = rec["action_head_end_ns"]
    return rec


def warmup(policy: Gr00tPolicy, observation: dict[str, Any], warmup_runs: int) -> None:
    for _ in range(warmup_runs):
        collated_inputs = prepare_model_inputs(policy, observation)
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
            _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
    torch.cuda.synchronize()


def integrate_segment(samples_df: pd.DataFrame, start_ns: int, end_ns: int, power_col: str) -> float:
    seg = samples_df[(samples_df["ts_ns"] >= start_ns) & (samples_df["ts_ns"] <= end_ns)].copy()
    if len(seg) < 2:
        return float("nan")
    ts = seg["ts_ns"].to_numpy(dtype=np.int64)
    p = seg[power_col].to_numpy(dtype=float)
    valid = np.isfinite(p)
    if valid.sum() < 2:
        return float("nan")
    ts = ts[valid]
    p = p[valid]
    dt = np.diff(ts) / 1e9
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * dt))


def load_nvtx_ranges(nvtx_csv: Path) -> pd.DataFrame:
    if not nvtx_csv.exists():
        return pd.DataFrame(columns=["name", "start_ns", "end_ns", "duration_ms"])

    stacks: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []
    with nvtx_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw_ts, event in reader:
            try:
                ts_ns = int(raw_ts)
            except ValueError:
                continue
            if event.endswith("_START"):
                name = event[: -len("_START")]
                stacks.setdefault(name, []).append(ts_ns)
            elif event.endswith("_END"):
                name = event[: -len("_END")]
                stack = stacks.get(name)
                if not stack:
                    continue
                start_ns = stack.pop()
                rows.append(
                    {
                        "name": name,
                        "start_ns": start_ns,
                        "end_ns": ts_ns,
                        "duration_ms": (ts_ns - start_ns) / 1e6,
                    }
                )
    return pd.DataFrame(rows)


def _extract_phase_span(
    nvtx_df: pd.DataFrame,
    rec: dict[str, int],
    *,
    event_name: str,
    phase_name: str,
) -> dict[str, Any] | None:
    if nvtx_df.empty:
        return None
    seg = nvtx_df[
        (nvtx_df["name"] == event_name)
        & (nvtx_df["start_ns"] >= rec["infer_start_ns"])
        & (nvtx_df["end_ns"] <= rec["infer_end_ns"])
    ].copy()
    if seg.empty:
        return None
    start_ns = int(seg["start_ns"].min())
    end_ns = int(seg["end_ns"].max())
    return {
        "inference_id": rec["inference_id"],
        "phase": phase_name,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ms": (end_ns - start_ns) / 1e6,
    }


def build_phase_rows(records: list[dict[str, int]], nvtx_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        vit_row = _extract_phase_span(nvtx_df, rec, event_name="VLA/ViT", phase_name="vit")
        if vit_row is not None:
            rows.append(vit_row)
        llm_row = _extract_phase_span(nvtx_df, rec, event_name="VLA/LLM", phase_name="llm")
        if llm_row is not None:
            rows.append(llm_row)
        rows.append(
            {
                "inference_id": rec["inference_id"],
                "phase": "action_head",
                "start_ns": rec["action_head_start_ns"],
                "end_ns": rec["action_head_end_ns"],
                "duration_ms": (rec["action_head_end_ns"] - rec["action_head_start_ns"]) / 1e6,
            }
        )
    return rows


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


def annotate_inference_windows(ax, records: list[dict[str, int]], origin_ns: int) -> None:
    y_top = ax.get_ylim()[1]
    for rec in records:
        s_ms = (rec["infer_start_ns"] - origin_ns) / 1e6
        e_ms = (rec["infer_end_ns"] - origin_ns) / 1e6
        mid = 0.5 * (s_ms + e_ms)
        ax.axvline(s_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(e_ms, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(mid, y_top, f"Inference {rec['inference_id']}", ha="center", va="bottom", fontsize=9)


def select_middle_records(records: list[dict[str, int]], count: int = 3) -> list[dict[str, int]]:
    if len(records) <= count:
        return records
    start = max(0, (len(records) - count) // 2)
    return records[start : start + count]


def subset_phase_rows(phase_df: pd.DataFrame, inference_ids: set[int]) -> pd.DataFrame:
    return phase_df[phase_df["inference_id"].isin(inference_ids)].copy()


def make_trace_plots(
    out_dir: Path,
    samples_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    records: list[dict[str, int]],
    title: str,
    *,
    stem: str = "",
) -> None:
    origin_ns = records[0]["infer_start_ns"]
    telem_df = add_relative_time(samples_df, origin_ns)
    phase_local = phase_df.copy()

    cpu_ghz = valid_or_nan(telem_df["cpu4_khz"], scale=1e6)
    gpu_ghz = valid_or_nan(telem_df["gpu_gpc0_hz"], scale=1e9)
    emc_ghz = valid_or_nan(telem_df["emc_rate_hz"], scale=1e9)

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.28, label=PHASE_LABELS[p])
        for p in ["vit", "llm", "action_head"]
    ]

    def _plot_valid_series(ax, x, y, *, label: str, color: str, linewidth: float = 1.4, step: bool = False):
        arr = np.asarray(y, dtype=float)
        if not np.isfinite(arr).any():
            return
        if step:
            ax.step(x, arr, where="post", label=label, color=color, linewidth=linewidth)
        else:
            ax.plot(x, arr, label=label, color=color, linewidth=linewidth)

    # Frequency plot
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], cpu_ghz, label="CPU policy4 (GHz)", color="C2", linewidth=1.5, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], gpu_ghz, label="GPU GPC0 (GHz)", color="C1", linewidth=1.5, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], emc_ghz, label="EMC (GHz)", color="C3", linewidth=1.5, step=True)
    ax.set_xlabel("Time (ms relative to inference 1 start)")
    ax.set_ylabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")
    fig.suptitle(f"{title} — CPU/GPU/EMC frequency", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}freq_trace.png", bbox_inches="tight")
    plt.close(fig)

    # Power plot
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["gpu_power_w"], label="VDD_GPU (W)", color="C1", linewidth=1.4)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["cpu_power_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.4)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_power_w"], label="VIN (W)", color="C0", linewidth=1.4)
    ax.set_xlabel("Time (ms relative to inference 1 start)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")
    fig.suptitle(f"{title} — VDD_GPU / VDD_CPU_SOC_MSS / VIN power", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}power_trace.png", bbox_inches="tight")
    plt.close(fig)

    # Cumulative energy plot
    gpu_e = cumulative_energy(samples_df, "gpu_power_w", origin_ns)
    cpu_e = cumulative_energy(samples_df, "cpu_power_w", origin_ns)
    vin_e = cumulative_energy(samples_df, "vin_power_w", origin_ns)
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, gpu_e["t_ms"], gpu_e["energy_j"], label="VDD_GPU (J)", color="C1", linewidth=1.6)
    _plot_valid_series(ax, cpu_e["t_ms"], cpu_e["energy_j"], label="VDD_CPU_SOC_MSS (J)", color="C2", linewidth=1.6)
    _plot_valid_series(ax, vin_e["t_ms"], vin_e["energy_j"], label="VIN (J)", color="C0", linewidth=1.6)
    ax.set_xlabel("Time (ms relative to inference 1 start)")
    ax.set_ylabel("Cumulative energy (J)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper left")
    fig.suptitle(f"{title} — cumulative VDD_GPU / VDD_CPU_SOC_MSS / VIN energy", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}energy_trace.png", bbox_inches="tight")
    plt.close(fig)

    # Combined overview
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), dpi=160, sharex=True)
    # Freq
    ax = axes[0]
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], cpu_ghz, label="CPU policy4 (GHz)", color="C2", linewidth=1.4, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], gpu_ghz, label="GPU GPC0 (GHz)", color="C1", linewidth=1.4, step=True)
    _plot_valid_series(ax, telem_df["t_ms"], emc_ghz, label="EMC (GHz)", color="C3", linewidth=1.4, step=True)
    ax.set_ylabel("Freq (GHz)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_handles, loc="upper right")
    # Power
    ax = axes[1]
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["gpu_power_w"], label="VDD_GPU (W)", color="C1", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["cpu_power_w"], label="VDD_CPU_SOC_MSS (W)", color="C2", linewidth=1.3)
    _plot_valid_series(ax, telem_df["t_ms"], telem_df["vin_power_w"], label="VIN (W)", color="C0", linewidth=1.3)
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    ax.legend(loc="upper right")
    # Energy
    ax = axes[2]
    shade_phases(ax, phase_local, origin_ns)
    _plot_valid_series(ax, gpu_e["t_ms"], gpu_e["energy_j"], label="VDD_GPU (J)", color="C1", linewidth=1.4)
    _plot_valid_series(ax, cpu_e["t_ms"], cpu_e["energy_j"], label="VDD_CPU_SOC_MSS (J)", color="C2", linewidth=1.4)
    _plot_valid_series(ax, vin_e["t_ms"], vin_e["energy_j"], label="VIN (J)", color="C0", linewidth=1.4)
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (ms relative to inference 1 start)")
    ax.grid(True, alpha=0.3)
    annotate_inference_windows(ax, records, origin_ns)
    ax.legend(loc="upper left")
    fig.suptitle(f"{title} — consecutive measured inferences", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}trace_overview.png", bbox_inches="tight")
    plt.close(fig)


def write_stats_markdown(out_dir: Path, per_inf_df: pd.DataFrame) -> None:
    metric_cols = [
        "duration_ms",
        "data_processing_ms",
        "vit_ms",
        "llm_ms",
        "action_head_ms",
        "gpu_energy_j",
        "cpu_energy_j",
        "vin_energy_j",
    ]
    with (out_dir / "trace_stats.md").open("w", encoding="utf-8") as f:
        f.write("# Trace Stats\n\n")
        f.write("Warmup is excluded. The rows below summarize the measured consecutive inferences.\n\n")
        f.write("## Per-inference\n\n")
        cols = list(per_inf_df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in per_inf_df.itertuples(index=False):
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
            vals = pd.to_numeric(per_inf_df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            f.write(f"| {col} | {vals.mean():.3f} | {vals.std(ddof=0):.3f} | {vals.median():.3f} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace control-scenario inferences with raw frequency/power/energy telemetry.")
    parser.add_argument("--model_path", type=str, default="outputs/libero_long_hf_ckpt")
    parser.add_argument("--dataset_path", type=str, default="examples/LIBERO/libero_10_no_noops_1.0.0_lerobot")
    parser.add_argument("--embodiment_tag", type=str, default="libero_panda")
    parser.add_argument("--trt_engine_path", type=str, default="outputs/libero_10_thor_onnx/dit_model_bf16.trt")
    parser.add_argument("--inference_mode", choices=["pytorch", "compile", "tensorrt"], default="tensorrt")
    parser.add_argument("--config_preset", choices=sorted(PRESET_CONFIGS.keys()), default="control_reduced_combo_with_default")
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--config", type=str, default="", help="Explicit name|cpu|gpu|emc config. Overrides --config_preset/--config_name.")
    parser.add_argument("--text_length", type=int, default=64)
    parser.add_argument("--text_unit", choices=["words", "chars"], default="words")
    parser.add_argument("--image_size", type=str, default="orig")
    parser.add_argument("--view_config", type=str, default="both_views|image,wrist_image")
    parser.add_argument("--filler_token", type=str, default="token")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num_inferences", type=int, default=2)
    parser.add_argument("--power_interval_ms", type=float, default=2.0)
    parser.add_argument("--pre_idle_ms", type=float, default=50.0)
    parser.add_argument("--inter_idle_ms", type=float, default=10.0)
    parser.add_argument("--post_idle_ms", type=float, default=100.0)
    parser.add_argument("--freq_settle_s", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="thor_measurements/control_trace_default")
    args = parser.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvtx_csv = out_dir / "nvtx_ranges.csv"
    if nvtx_csv.exists():
        nvtx_csv.unlink()
    os.environ["NVTX_RANGES_CSV"] = str(nvtx_csv)
    nvtx_utils._NVTX_RANGES_CSV = str(nvtx_csv)

    embodiment = EmbodimentTag(args.embodiment_tag)
    policy = _choose_policy(args, embodiment)
    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    base_observation, _base_text = build_observation(dataset, modality_config, embodiment, step_index=args.step_index)
    view_cfg = parse_view_config_spec(args.view_config)
    set_policy_view_keys(policy, view_cfg.keys)
    obs = mutate_observation(
        base_observation,
        image_size=parse_image_size(args.image_size),
        view_keys=view_cfg.keys,
        text_length=args.text_length,
        text_unit=args.text_unit,
        filler_token=args.filler_token,
    )

    config = _choose_config(args)

    try:
        settle_freq_config(config, settle_s=args.freq_settle_s)
        warmup_runs = args.warmup + (5 if args.inference_mode == "tensorrt" else 0)
        warmup(policy, obs, warmup_runs=warmup_runs)

        samples: list[dict[str, float]] = []
        ina3221, gpu_ch, cpu_ch, vin_ch = build_power_sampler()
        stop_flag = [False]
        worker = threading.Thread(
            target=telemetry_loop,
            args=(stop_flag, samples, args.power_interval_ms / 1000.0, ina3221, gpu_ch, cpu_ch, vin_ch),
            daemon=True,
        )
        worker.start()

        time.sleep(args.pre_idle_ms / 1000.0)
        records: list[dict[str, int]] = []
        for idx in range(args.num_inferences):
            records.append(trace_single_inference(policy, obs, inference_id=idx + 1))
            if idx != args.num_inferences - 1 and args.inter_idle_ms > 0:
                time.sleep(args.inter_idle_ms / 1000.0)
        time.sleep(args.post_idle_ms / 1000.0)
    finally:
        stop_flag[0] = True
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
        unlock_all_freqs()

    if not samples:
        raise RuntimeError("No telemetry samples captured.")
    samples.append(_telemetry_sample_dict(ina3221, gpu_ch, cpu_ch, vin_ch))

    samples_df = pd.DataFrame(samples).sort_values("ts_ns")
    nvtx_df = load_nvtx_ranges(nvtx_csv)
    phase_rows = build_phase_rows(records, nvtx_df)
    phase_df = pd.DataFrame(phase_rows)

    samples_df.to_csv(out_dir / "telemetry_raw.csv", index=False)
    phase_df.to_csv(out_dir / "phase_ranges.csv", index=False)
    if not nvtx_df.empty:
        nvtx_df.to_csv(out_dir / "nvtx_ranges.csv", index=False)

    per_inf_rows: list[dict[str, Any]] = []
    for rec in records:
        inf_phase = phase_df[phase_df["inference_id"] == rec["inference_id"]]
        vit_ms = float(inf_phase.loc[inf_phase["phase"] == "vit", "duration_ms"].sum())
        llm_ms = float(inf_phase.loc[inf_phase["phase"] == "llm", "duration_ms"].sum())
        row = {
            "inference_id": rec["inference_id"],
            "duration_ms": (rec["infer_end_ns"] - rec["infer_start_ns"]) / 1e6,
            "data_processing_ms": (rec["data_processing_end_ns"] - rec["data_processing_start_ns"]) / 1e6,
            "vit_ms": vit_ms,
            "llm_ms": llm_ms,
            "backbone_ms": (rec["backbone_end_ns"] - rec["backbone_start_ns"]) / 1e6,
            "action_head_ms": (rec["action_head_end_ns"] - rec["action_head_start_ns"]) / 1e6,
            "gpu_energy_j": integrate_segment(samples_df, rec["infer_start_ns"], rec["infer_end_ns"], "gpu_power_w"),
            "cpu_energy_j": integrate_segment(samples_df, rec["infer_start_ns"], rec["infer_end_ns"], "cpu_power_w"),
            "vin_energy_j": integrate_segment(samples_df, rec["infer_start_ns"], rec["infer_end_ns"], "vin_power_w"),
        }
        per_inf_rows.append(row)
    pd.DataFrame(per_inf_rows).to_csv(out_dir / "per_inference_summary.csv", index=False)
    per_inf_df = pd.DataFrame(per_inf_rows)

    title = (
        f"default trace | {args.inference_mode} | text={args.text_length}_{args.text_unit} | "
        f"views={view_cfg.name} ({len(view_cfg.keys)})"
        if config.name == "default"
        else (
            f"{config.name} | {args.inference_mode} | text={args.text_length}_{args.text_unit} | "
            f"views={view_cfg.name} ({len(view_cfg.keys)})"
        )
    )
    make_trace_plots(out_dir, samples_df, phase_df, records, title)
    write_stats_markdown(out_dir, per_inf_df)

    middle_records = select_middle_records(records, count=3)
    if len(middle_records) >= 2:
        middle_ids = {r["inference_id"] for r in middle_records}
        middle_start_ns = middle_records[0]["infer_start_ns"] - int(20e6)
        middle_end_ns = middle_records[-1]["infer_end_ns"] + int(20e6)
        middle_samples = samples_df[(samples_df["ts_ns"] >= middle_start_ns) & (samples_df["ts_ns"] <= middle_end_ns)].copy()
        middle_phases = subset_phase_rows(phase_df, middle_ids)
        make_trace_plots(
            out_dir,
            middle_samples,
            middle_phases,
            middle_records,
            title + " (middle inferences)",
            stem="middle3_",
        )

    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Control Trace Summary\n\n")
        f.write(f"- config: `{config.name}`\n")
        f.write(f"- cpu: `{config.cpu_label}`\n")
        f.write(f"- gpu: `{config.gpu_label}`\n")
        f.write(f"- emc: `{config.emc_label}`\n")
        f.write(f"- mode: `{args.inference_mode}`\n")
        f.write(f"- text: `{args.text_length}_{args.text_unit}`\n")
        f.write(f"- views: `{view_cfg.name}` -> `{','.join(view_cfg.keys)}`\n")
        f.write(f"- num_inferences: `{args.num_inferences}`\n")
        f.write(f"- warmup_runs: `{warmup_runs}`\n")
        f.write(f"- power interval: `{args.power_interval_ms} ms`\n")
        f.write(f"- output: `{out_dir}`\n")

    print(f"[DONE] Wrote: {out_dir/'telemetry_raw.csv'}")
    print(f"[DONE] Wrote: {out_dir/'phase_ranges.csv'}")
    if not nvtx_df.empty:
        print(f"[DONE] Wrote: {out_dir/'nvtx_ranges.csv'}")
    print(f"[DONE] Wrote: {out_dir/'per_inference_summary.csv'}")
    print(f"[DONE] Wrote: {out_dir/'freq_trace.png'}")
    print(f"[DONE] Wrote: {out_dir/'power_trace.png'}")
    print(f"[DONE] Wrote: {out_dir/'energy_trace.png'}")
    print(f"[DONE] Wrote: {out_dir/'trace_overview.png'}")
    print(f"[DONE] Wrote: {out_dir/'trace_stats.md'}")
    if len(records) >= 3:
        print(f"[DONE] Wrote: {out_dir/'middle3_trace_overview.png'}")


if __name__ == "__main__":
    main()
