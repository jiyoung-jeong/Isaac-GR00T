#!/usr/bin/env python3
"""
Sweep input-side hyperparameters for GR00T inference timing.

This script focuses on input/control variations rather than environment
rollouts. It benchmarks model inference while varying:

  - language instruction length
  - image resolution
  - denoising steps

It reuses the same model/data pipeline as deployment benchmarking, and writes:

  - raw per-configuration summary CSV
  - optional heatmaps when both text lengths and image sizes have >1 choices

Examples:

python scripts/deployment/benchmark_input_sweep.py \
  --model_path outputs/libero_long_hf_ckpt \
  --dataset_path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot \
  --embodiment_tag libero_panda \
  --trt_engine_path outputs/libero_10_thor_onnx/dit_model_bf16.trt \
  --inference_modes pytorch tensorrt \
  --text_lengths 8 16 32 64 \
  --image_sizes 224x224 256x256 320x320 \
  --num_iterations 20 \
  --warmup 5 \
  --out_dir thor_measurements/input_sweep_libero10
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch

from deployment_scripts import thor_cpufreq_power_sweep as cpu_mod
from deployment_scripts import thor_emcfreq_power_sweep as emc_mod
from deployment_scripts import thor_gpufreq_power_sweep as gpu_mod


def log_phase_event(name: str, event: str, **fields: object) -> None:
    path = os.environ.get("THOR_PHASE_EVENTS_CSV")
    if not path:
        return
    line_fields = {
        "ts_ns": time.monotonic_ns(),
        "name": name,
        "event": event,
        **fields,
    }
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(line_fields.keys()) + "\n")
        f.write(",".join(str(v) for v in line_fields.values()) + "\n")


@dataclass(frozen=True)
class FreqConfig:
    name: str
    cpu_hz: int | None
    gpu_hz: int | None
    emc_hz: int | None

    @property
    def cpu_label(self) -> str:
        return fmt_freq(self.cpu_hz)

    @property
    def gpu_label(self) -> str:
        return fmt_freq(self.gpu_hz)

    @property
    def emc_label(self) -> str:
        return fmt_freq(self.emc_hz)


@dataclass(frozen=True)
class ViewConfig:
    name: str
    keys: tuple[str, ...]

    @property
    def label(self) -> str:
        return self.name


def fmt_freq(freq_hz: int | None) -> str:
    if freq_hz is None:
        return "default"
    if freq_hz >= 1_000_000_000:
        return f"{freq_hz / 1e9:.3f}GHz"
    return f"{int(round(freq_hz / 1e6))}MHz"


def _build_control_reduced_configs() -> list[FreqConfig]:
    configs: list[FreqConfig] = []
    cpu_vals = [2_430_000_000, 2_601_000_000]
    gpu_vals = [None, 1_107_000_000, 1_305_000_000, 1_503_000_000]
    emc_vals = [None, 2_750_000_000, 3_200_000_000, 4_266_000_000]
    for cpu_hz in cpu_vals:
        for gpu_hz in gpu_vals:
            for emc_hz in emc_vals:
                name = f"cpu_{fmt_freq(cpu_hz)}_gpu_{fmt_freq(gpu_hz)}_emc_{fmt_freq(emc_hz)}"
                configs.append(FreqConfig(name=name, cpu_hz=cpu_hz, gpu_hz=gpu_hz, emc_hz=emc_hz))
    return configs


PRESET_CONFIGS: dict[str, list[FreqConfig]] = {
    "libero_spatial_candidates": [
        FreqConfig("default", None, None, None),
        FreqConfig("best_latency", 2_601_000_000, None, None),
        FreqConfig("best_energy", 2_430_000_000, 1_107_000_000, 3_200_000_000),
        FreqConfig("best_tradeoff", 2_430_000_000, 1_503_000_000, None),
    ],
    "libero10_candidates": [
        FreqConfig("default", None, None, None),
        FreqConfig("best_latency", 2_601_000_000, None, 4_266_000_000),
        FreqConfig("best_energy", 2_430_000_000, 1_305_000_000, 2_750_000_000),
        FreqConfig("best_tradeoff", 2_430_000_000, None, 4_266_000_000),
    ],
    "control_reduced_combo": _build_control_reduced_configs(),
    "control_reduced_combo_with_default": [FreqConfig("default", None, None, None), *_build_control_reduced_configs()],
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def parse_freq_or_default(text: str) -> int | None:
    lowered = text.strip().lower()
    if lowered in {"default", "none", "-1"}:
        return None
    return cpu_mod.parse_freq_hz(text)


def parse_config_spec(spec: str) -> FreqConfig:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --config '{spec}'. Use name|cpu|gpu|emc, e.g. best|2.430GHz|default|4.266GHz"
        )
    name, cpu_s, gpu_s, emc_s = parts
    return FreqConfig(
        name=name,
        cpu_hz=parse_freq_or_default(cpu_s),
        gpu_hz=parse_freq_or_default(gpu_s),
        emc_hz=parse_freq_or_default(emc_s),
    )


def parse_view_config_spec(spec: str) -> ViewConfig:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --view_config '{spec}'. Use name|key1,key2, e.g. front_only|image"
        )
    name, keys_s = parts
    keys = tuple(k.strip() for k in keys_s.split(",") if k.strip())
    if not keys:
        raise ValueError(f"Invalid --view_config '{spec}': must contain at least one key")
    return ViewConfig(name=name, keys=keys)


def choose_configs(args: argparse.Namespace) -> list[FreqConfig]:
    configs: list[FreqConfig] = []
    for preset in args.config_presets:
        configs.extend(PRESET_CONFIGS[preset])
    for spec in args.config:
        configs.append(parse_config_spec(spec))
    if not configs:
        configs.append(FreqConfig("default", None, None, None))

    deduped: list[FreqConfig] = []
    seen = set()
    for cfg in configs:
        key = (cfg.name, cfg.cpu_hz, cfg.gpu_hz, cfg.emc_hz)
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    return deduped


def choose_view_configs(args: argparse.Namespace, available_keys: list[str]) -> list[ViewConfig]:
    preset_view_configs: dict[str, list[ViewConfig]] = {
        "all_views": [ViewConfig("all_views", tuple(available_keys))],
        "first_last_views": [
            ViewConfig("first_only", (available_keys[0],)),
            ViewConfig("last_only", (available_keys[-1],)),
            ViewConfig("all_views", tuple(available_keys)),
        ],
    }
    if set(["image", "wrist_image"]).issubset(set(available_keys)):
        preset_view_configs["libero_dual_view"] = [
            ViewConfig("image_only", ("image",)),
            ViewConfig("wrist_only", ("wrist_image",)),
            ViewConfig("both_views", ("image", "wrist_image")),
        ]

    configs: list[ViewConfig] = []
    for preset in args.view_config_presets:
        if preset not in preset_view_configs:
            raise ValueError(
                f"Unknown view-config preset '{preset}'. Available presets: {sorted(preset_view_configs.keys())}"
            )
        configs.extend(preset_view_configs[preset])
    for spec in args.view_config:
        configs.append(parse_view_config_spec(spec))
    if not configs:
        configs.append(ViewConfig("all_views", tuple(available_keys)))

    deduped: list[ViewConfig] = []
    seen = set()
    available_set = set(available_keys)
    for cfg in configs:
        missing = set(cfg.keys) - available_set
        if missing:
            raise ValueError(
                f"View config '{cfg.name}' requests keys {sorted(missing)} not present in available views {sorted(available_keys)}"
            )
        key = (cfg.name, cfg.keys)
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    return deduped


def set_policy_view_keys(policy: Gr00tPolicy, view_keys: tuple[str, ...]) -> None:
    policy.modality_configs["video"].modality_keys = list(view_keys)


def set_policy_denoising_steps(policy: Gr00tPolicy, denoising_steps: int) -> None:
    policy.model.action_head.num_inference_timesteps = denoising_steps


def apply_freq_config(config: FreqConfig) -> None:
    if config.cpu_hz is None:
        cpu_mod.unlock_cpu_freq_all()
    else:
        cpu_mod.set_cpu_freq_all(config.cpu_hz)

    if config.gpu_hz is None:
        gpu_mod.unlock_gpu_freq()
    else:
        gpu_mod.set_gpu_freq(config.gpu_hz)

    if config.emc_hz is None:
        emc_mod.unlock_emc_freq()
    else:
        emc_mod.set_emc_freq(config.emc_hz)


def unlock_all_freqs() -> None:
    cpu_mod.unlock_cpu_freq_all()
    gpu_mod.unlock_gpu_freq()
    emc_mod.unlock_emc_freq()


def config_matches(config: FreqConfig) -> bool:
    cpu_ok = True if config.cpu_hz is None else cpu_mod.cpu_lock_matches(config.cpu_hz)[0]
    gpu_ok = True if config.gpu_hz is None else gpu_mod.gpu_lock_matches(config.gpu_hz)[0]
    emc_ok = True if config.emc_hz is None else emc_mod.emc_lock_matches(config.emc_hz)[0]
    return cpu_ok and gpu_ok and emc_ok


def settle_freq_config(config: FreqConfig, settle_s: float, timeout_s: float = 5.0) -> None:
    apply_freq_config(config)
    time.sleep(settle_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if config_matches(config):
            return
        time.sleep(0.05)
    raise RuntimeError(
        "Failed to settle frequency config "
        f"{config.name}: CPU {config.cpu_label} | GPU {config.gpu_label} | EMC {config.emc_label}"
    )


def build_power_sampler():
    ina3221 = cpu_mod.find_hwmon(cpu_mod.INA3221_ROOT, required_label="GPU")
    gpu_ch = cpu_mod.find_ina3221_channel(ina3221, "GPU") if ina3221 else None
    cpu_ch = cpu_mod.find_ina3221_channel(ina3221, "CPU_SOC_MSS") if ina3221 else None
    vin_ch = None
    return ina3221, gpu_ch, cpu_ch, vin_ch


def sample_power_row(ina3221, gpu_ch, cpu_ch, vin_ch) -> dict[str, float]:
    return {
        "ts_ns": time.monotonic_ns(),
        "gpu_power_w": cpu_mod.read_power_from_channel(ina3221, gpu_ch),
        "cpu_soc_mss_power_w": cpu_mod.read_power_from_channel(ina3221, cpu_ch),
        "vin_power_w": cpu_mod.read_vin_power_w(),
    }


def integrate_energy(samples: list[dict[str, float]], key: str) -> float:
    if len(samples) < 2:
        return float("nan")
    energy_j = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = (b["ts_ns"] - a["ts_ns"]) / 1e9
        pa = a[key]
        pb = b[key]
        if not np.isfinite(pa) or not np.isfinite(pb):
            continue
        energy_j += 0.5 * (pa + pb) * dt
    return energy_j


def telemetry_loop(stop_flag, samples, interval_s: float, ina3221, gpu_ch, cpu_ch, vin_ch):
    while not stop_flag[0]:
        samples.append(sample_power_row(ina3221, gpu_ch, cpu_ch, vin_ch))
        time.sleep(interval_s)


def prepare_model_inputs(policy, observation):
    unbatched_obs = []
    batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
    for i in range(batch_size):
        unbatched_value = {
            "video": {k: v[i] for k, v in observation["video"].items()},
            "state": {k: v[i] for k, v in observation["state"].items()},
            "language": {k: v[i] for k, v in observation["language"].items()},
        }
        unbatched_obs.append(unbatched_value)

    processed_inputs = []
    for obs in unbatched_obs:
        vla_step_data = VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = collated_inputs["inputs"]
    return _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)


def benchmark_data_processing(policy, observation, num_iterations=20, warmup=10):
    log_phase_event("data_processing_warmup", "start", iterations=warmup)
    for _ in range(warmup):
        _ = prepare_model_inputs(policy, observation)
    log_phase_event("data_processing_warmup", "end", iterations=warmup)
    times = []
    for i in range(num_iterations):
        log_phase_event("data_processing_iter", "start", iteration=i)
        start = time.perf_counter()
        _ = prepare_model_inputs(policy, observation)
        end = time.perf_counter()
        log_phase_event("data_processing_iter", "end", iteration=i)
        times.append(end - start)
    return np.array(times) * 1000.0


def benchmark_components(policy, observation, num_iterations=20, warmup=3):
    log_phase_event("gpu_warmup", "start", iterations=warmup)
    for i in range(warmup):
        collated_inputs = prepare_model_inputs(policy, observation)
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
            _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
    torch.cuda.synchronize()
    log_phase_event("gpu_warmup", "end", iterations=warmup)

    backbone_times = []
    action_head_times = []
    for i in range(num_iterations):
        collated_inputs = prepare_model_inputs(policy, observation)

        torch.cuda.synchronize()
        log_phase_event("backbone_iter", "start", iteration=i)
        start = time.perf_counter()
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        log_phase_event("backbone_iter", "end", iteration=i)
        backbone_times.append(end - start)

        torch.cuda.synchronize()
        log_phase_event("action_head_iter", "start", iteration=i)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        log_phase_event("action_head_iter", "end", iteration=i)
        action_head_times.append(end - start)

    data_processing_times = benchmark_data_processing(policy, observation, num_iterations, warmup=10)
    return {
        "data_processing": data_processing_times,
        "backbone": np.array(backbone_times) * 1000.0,
        "action_head": np.array(action_head_times) * 1000.0,
    }


def compute_e2e_from_components(components):
    return components["data_processing"] + components["backbone"] + components["action_head"]


def replace_dit_with_tensorrt(policy: Gr00tPolicy | Any, trt_engine_path: str, device: int = 0):
    from scripts.deployment.standalone_inference_script import replace_dit_with_tensorrt as _replace

    _replace(policy, trt_engine_path, device=device)


def parse_image_size(text: str) -> tuple[int, int]:
    lowered = text.strip().lower()
    if lowered == "orig":
        return (-1, -1)
    if "x" not in lowered:
        raise ValueError(f"Invalid image size '{text}'. Use HxW, e.g. 224x224")
    h, w = lowered.split("x", 1)
    return int(h), int(w)


def format_image_size(size: tuple[int, int]) -> str:
    h, w = size
    return "orig" if h < 0 or w < 0 else f"{h}x{w}"


def resize_frame(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if target_h < 0 or target_w < 0:
        return frame
    img = Image.fromarray(frame)
    resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(resized)


def resize_video_array(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if target_hw[0] < 0 or target_hw[1] < 0:
        return arr
    out = np.empty(arr.shape[:-3] + (target_hw[0], target_hw[1], arr.shape[-1]), dtype=arr.dtype)
    for idx in np.ndindex(arr.shape[:-3]):
        out[idx] = resize_frame(arr[idx], target_hw)
    return out


def adjust_text_length(text: str, target_len: int, unit: str, filler_token: str) -> str:
    if target_len <= 0:
        return text
    if unit == "words":
        words = text.split()
        if len(words) >= target_len:
            return " ".join(words[:target_len])
        return " ".join(words + [filler_token] * (target_len - len(words)))
    if unit == "chars":
        if len(text) >= target_len:
            return text[:target_len]
        extra = (" " + filler_token) * max(1, target_len - len(text))
        return (text + extra)[:target_len]
    raise ValueError(f"Unsupported text unit: {unit}")


def build_observation(dataset, modality_config, embodiment_tag: EmbodimentTag, step_index: int = 0):
    episode_data = dataset[0]
    step_data = extract_step_data(
        episode_data,
        step_index=step_index,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        allow_padding=False,
    )
    obs = {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
        "state": {k: step_data.states[k][None] for k in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }
    return obs, step_data.text


def mutate_observation(
    base_observation: dict[str, Any],
    *,
    image_size: tuple[int, int],
    view_keys: tuple[str, ...],
    text_length: int | None,
    text_unit: str,
    filler_token: str,
) -> dict[str, Any]:
    obs = copy.deepcopy(base_observation)
    obs["video"] = {k: copy.deepcopy(v) for k, v in obs["video"].items() if k in set(view_keys)}
    for key, value in obs["video"].items():
        obs["video"][key] = resize_video_array(value, image_size)

    if text_length is not None:
        lang_key = list(obs["language"].keys())[0]
        original = obs["language"][lang_key][0][0]
        obs["language"][lang_key][0][0] = adjust_text_length(
            original,
            target_len=text_length,
            unit=text_unit,
            filler_token=filler_token,
        )
    return obs


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def save_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
    *,
    row_col: str = "text_label",
    col_col: str = "image_label",
    row_axis_label: str = "Text length",
    col_axis_label: str = "Image size",
) -> None:
    pivot = df.pivot_table(index=row_col, columns=col_col, values=metric_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.8 * max(3, len(pivot.columns)), 1.5 * max(3, len(pivot.index))), dpi=150)
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_col)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel(col_axis_label)
    ax.set_ylabel(row_axis_label)
    ax.set_title(title)
    threshold = float(np.nanmedian(data[np.isfinite(data)])) if np.isfinite(data).any() else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white" if val >= threshold else "black",
                fontsize=10,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def maybe_save_metric_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    group_cols: list[str],
    heatmap_col_specs: list[tuple[str, str, str, str]],
) -> None:
    metric_cols = [
        "data_processing_median_ms",
        "backbone_median_ms",
        "action_head_median_ms",
        "e2e_median_ms",
        "gpu_power_mean_w",
        "cpu_soc_mss_power_mean_w",
        "vin_power_mean_w",
        "gpu_energy_j",
        "cpu_soc_mss_energy_j",
        "vin_energy_j",
    ]
    for row_col, col_col, row_axis_label, col_axis_label in heatmap_col_specs:
        if row_col not in df.columns or col_col not in df.columns:
            continue
        if len(set(df[row_col])) <= 1 or len(set(df[col_col])) <= 1:
            continue
        for group_key, sub in df.groupby(group_cols):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_items = dict(zip(group_cols, group_key))
            mode_dir = out_dir / str(group_items["mode"]) / str(group_items["config_name"])
            mode_dir.mkdir(parents=True, exist_ok=True)
            denoise_label = f" | denoise={group_items['denoising_steps']}" if "denoising_steps" in group_items else ""
            suffix = f"{row_col}_vs_{col_col}".replace("_label", "")
            for metric in metric_cols:
                save_heatmap(
                    sub,
                    metric_col=metric,
                    title=(
                        f"{group_items['mode']} | {group_items['config_name']}{denoise_label}: "
                        f"{metric} ({row_axis_label.lower()} x {col_axis_label.lower()})"
                    ),
                    out_path=mode_dir / f"{metric}_{suffix}.png",
                    row_col=row_col,
                    col_col=col_col,
                    row_axis_label=row_axis_label,
                    col_axis_label=col_axis_label,
                )


def run_benchmark_with_telemetry(
    policy,
    observation,
    *,
    num_iterations: int,
    warmup: int,
    power_interval_ms: float,
):
    ina3221, gpu_ch, cpu_ch, vin_ch = build_power_sampler()
    samples = []
    stop_flag = [False]
    worker = threading.Thread(
        target=telemetry_loop,
        args=(stop_flag, samples, power_interval_ms / 1000.0, ina3221, gpu_ch, cpu_ch, vin_ch),
        daemon=True,
    )
    worker.start()
    start_ns = time.monotonic_ns()
    try:
        components = benchmark_components(
            policy,
            observation,
            num_iterations=num_iterations,
            warmup=warmup,
        )
    finally:
        stop_flag[0] = True
        worker.join(timeout=2.0)
    end_ns = time.monotonic_ns()
    if not samples or samples[-1]["ts_ns"] < end_ns:
        samples.append(sample_power_row(ina3221, gpu_ch, cpu_ch, vin_ch))
        samples[-1]["ts_ns"] = end_ns

    telemetry = {
        "duration_s": (end_ns - start_ns) / 1e9,
        "gpu_power_mean_w": float(np.nanmean([s["gpu_power_w"] for s in samples])),
        "cpu_soc_mss_power_mean_w": float(np.nanmean([s["cpu_soc_mss_power_w"] for s in samples])),
        "vin_power_mean_w": float(np.nanmean([s["vin_power_w"] for s in samples])),
        "gpu_energy_j": integrate_energy(samples, "gpu_power_w"),
        "cpu_soc_mss_energy_j": integrate_energy(samples, "cpu_soc_mss_power_w"),
        "vin_energy_j": integrate_energy(samples, "vin_power_w"),
        "telemetry_samples": len(samples),
    }
    return components, telemetry


def main():
    parser = argparse.ArgumentParser(description="Sweep text length and image size for GR00T inference.")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--embodiment_tag", type=str, default="gr1")
    parser.add_argument("--trt_engine_path", type=str, default=None)
    parser.add_argument("--inference_modes", nargs="+", default=["pytorch"], choices=["pytorch", "compile", "tensorrt"])
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat_runs", type=int, default=1, help="Repeat the same condition multiple times and save one row per repeat.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--text_lengths", nargs="*", type=int, default=None, help="Target text lengths. If omitted, only original text is used.")
    parser.add_argument("--text_unit", choices=["words", "chars"], default="words")
    parser.add_argument("--image_sizes", nargs="*", default=None, help="Image sizes like 224x224 256x256. Use 'orig' to keep source size.")
    parser.add_argument(
        "--denoising_steps",
        nargs="*",
        type=int,
        default=None,
        help="Numbers of denoising steps to test. If omitted, uses the policy default.",
    )
    parser.add_argument("--filler_token", type=str, default="token")
    parser.add_argument(
        "--view_config_presets",
        nargs="*",
        default=[],
        help="Named camera-view config sets, e.g. libero_dual_view or first_last_views.",
    )
    parser.add_argument(
        "--view_config",
        nargs="*",
        default=[],
        help="Explicit view config specs: name|key1,key2. Example: image_only|image",
    )
    parser.add_argument(
        "--config_presets",
        nargs="*",
        default=[],
        choices=sorted(PRESET_CONFIGS.keys()),
        help="Named frequency config sets to compare, e.g. libero_spatial_candidates",
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Explicit config specs: name|cpu|gpu|emc. Use 'default' for an axis.",
    )
    parser.add_argument("--power_interval_ms", type=float, default=2.0)
    parser.add_argument("--freq_settle_s", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="thor_measurements/input_sweep")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: CUDA GPU required.", file=sys.stderr)
        sys.exit(1)

    if args.dataset_path is None:
        repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
        args.dataset_path = os.path.join(repo_path, "demo_data/gr1.PickNPlace")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embodiment = EmbodimentTag(args.embodiment_tag)
    base_policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=embodiment,
        device=device,
        strict=True,
    )
    modality_config = base_policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    base_observation, base_text = build_observation(dataset, modality_config, embodiment, step_index=args.step_index)
    available_view_keys = list(base_observation["video"].keys())

    text_lengths = args.text_lengths or [None]
    text_labels = [
        ("orig" if t is None else f"{t}_{args.text_unit}")
        for t in text_lengths
    ]
    image_sizes = [parse_image_size(v) for v in (args.image_sizes or ["orig"])]
    denoising_steps = args.denoising_steps or [int(base_policy.model.action_head.num_inference_timesteps)]
    view_configs = choose_view_configs(args, available_view_keys)
    freq_configs = choose_configs(args)

    policies: dict[str, Gr00tPolicy] = {"pytorch": base_policy}
    if "compile" in args.inference_modes:
        policy_compiled = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=embodiment,
            device=device,
            strict=True,
        )
        policy_compiled.model.action_head.model.forward = torch.compile(
            policy_compiled.model.action_head.model.forward, mode="max-autotune"
        )
        policies["compile"] = policy_compiled
    if "tensorrt" in args.inference_modes:
        if not args.trt_engine_path or not os.path.exists(args.trt_engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {args.trt_engine_path}")
        policy_trt = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=embodiment,
            device=device,
            strict=True,
        )
        replace_dit_with_tensorrt(policy_trt, args.trt_engine_path)
        policies["tensorrt"] = policy_trt

    rows: list[dict[str, Any]] = []
    try:
        for cfg in freq_configs:
            settle_freq_config(cfg, settle_s=args.freq_settle_s)
            for view_cfg in view_configs:
                for mode in args.inference_modes:
                    set_policy_view_keys(policies[mode], view_cfg.keys)
                for denoise_steps in denoising_steps:
                    for mode in args.inference_modes:
                        set_policy_denoising_steps(policies[mode], denoise_steps)
                    for text_len, text_label in zip(text_lengths, text_labels):
                        for image_size in image_sizes:
                            image_label = format_image_size(image_size)
                            observation = mutate_observation(
                                base_observation,
                                image_size=image_size,
                                view_keys=view_cfg.keys,
                                text_length=text_len,
                                text_unit=args.text_unit,
                                filler_token=args.filler_token,
                            )
                            actual_text = observation["language"][list(observation["language"].keys())[0]][0][0]
                            for repeat_id in range(args.repeat_runs):
                                for mode in args.inference_modes:
                                    policy = policies[mode]
                                    warmup = args.warmup + (5 if mode == "tensorrt" else 0)
                                    components, telemetry = run_benchmark_with_telemetry(
                                        policy,
                                        observation,
                                        num_iterations=args.num_iterations,
                                        warmup=warmup,
                                        power_interval_ms=args.power_interval_ms,
                                    )
                                    e2e = compute_e2e_from_components(components)
                                    row = {
                                        "repeat_id": repeat_id,
                                        "config_name": cfg.name,
                                        "cpu_label": cfg.cpu_label,
                                        "gpu_label": cfg.gpu_label,
                                        "emc_label": cfg.emc_label,
                                        "mode": mode,
                                        "view_label": view_cfg.label,
                                        "view_keys": ",".join(view_cfg.keys),
                                        "num_views": len(view_cfg.keys),
                                        "denoising_steps": denoise_steps,
                                        "text_length_target": text_len if text_len is not None else -1,
                                        "text_label": text_label,
                                        "image_h": image_size[0],
                                        "image_w": image_size[1],
                                        "image_label": image_label,
                                        "actual_text_chars": len(actual_text),
                                        "actual_text_words": len(actual_text.split()),
                                        **telemetry,
                                    }
                                    for metric_name, values in [
                                        ("data_processing", components["data_processing"]),
                                        ("backbone", components["backbone"]),
                                        ("action_head", components["action_head"]),
                                        ("e2e", e2e),
                                    ]:
                                        stats = summarize_metric(values)
                                        for key, value in stats.items():
                                            row[f"{metric_name}_{key}"] = value
                                    rows.append(row)
                                    print(
                                        f"[DONE] rep={repeat_id} cfg={cfg.name} mode={mode} view={view_cfg.label} "
                                        f"denoise={denoise_steps} text={text_label} image={image_label} "
                                        f"e2e_median={row['e2e_median_ms']:.2f} ms vin_energy={row['vin_energy_j']:.2f} J"
                                    )
    finally:
        unlock_all_freqs()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    maybe_save_metric_heatmaps(
        df,
        out_dir,
        group_cols=["mode", "config_name", "denoising_steps"],
        heatmap_col_specs=[
            ("text_label", "image_label", "Text length", "Image size"),
            ("text_label", "view_label", "Text length", "View config"),
            ("text_label", "denoising_steps", "Text length", "Denoising steps"),
            ("view_label", "denoising_steps", "View config", "Denoising steps"),
        ],
    )

    # A small markdown summary is handy when we come back later.
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Input Sweep Summary\n\n")
        f.write(f"- model: `{args.model_path}`\n")
        f.write(f"- dataset: `{args.dataset_path}`\n")
        f.write(f"- embodiment: `{args.embodiment_tag}`\n")
        f.write(f"- base text chars: `{len(base_text)}`\n")
        f.write(f"- base text words: `{len(base_text.split())}`\n")
        f.write(f"- modes: `{', '.join(args.inference_modes)}`\n")
        f.write(f"- configs: `{', '.join(cfg.name for cfg in freq_configs)}`\n")
        f.write(f"- view configs: `{', '.join(cfg.label for cfg in view_configs)}`\n")
        f.write(f"- denoising steps: `{', '.join(str(v) for v in denoising_steps)}`\n")
        f.write(f"- output: `{out_dir}`\n")


if __name__ == "__main__":
    main()
