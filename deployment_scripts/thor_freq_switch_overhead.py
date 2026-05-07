#!/usr/bin/env python3
"""
Measure Jetson Thor frequency switching overhead for CPU, GPU, and EMC.

This script reuses the same sysfs/debugfs control paths as the existing sweep
helpers and records three timestamps per transition:

  - write_overhead_ms: time spent issuing the sysfs/debugfs writes
  - feedback_ms: time from the first write until the target state is first
    observed through the chosen sysfs/debugfs feedback path
  - stable_ms: time from the first write until the feedback path reports the
    target state for N consecutive samples

For CPU, the existing sweep logic treats scaling_min/max as the lock source of
truth, so CPU feedback uses those values. cpuinfo_cur_freq is recorded as an
observation but not used for convergence. For GPU and EMC, the feedback source
matches the existing sweep semantics.

By default this script measures pairwise transitions for the selected single
axis modes (CPU/GPU/EMC) and emits per-axis heatmaps. Optional combo
measurement is provided in a cheaper "from-default" style rather than full
pairwise state explosion.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deployment_scripts import thor_cpufreq_power_sweep as cpu_mod
from deployment_scripts import thor_emcfreq_power_sweep as emc_mod
from deployment_scripts import thor_gpufreq_power_sweep as gpu_mod


@dataclass(frozen=True)
class SingleAxisState:
    axis: str
    freq_hz: int | None

    @property
    def label(self) -> str:
        return format_state_freq(self.freq_hz)


@dataclass(frozen=True)
class ComboState:
    cpu_hz: int | None
    gpu_hz: int | None
    emc_hz: int | None

    @property
    def label(self) -> str:
        return (
            f"CPU {format_state_freq(self.cpu_hz)} | "
            f"GPU {format_state_freq(self.gpu_hz)} | "
            f"EMC {format_state_freq(self.emc_hz)}"
        )


def format_state_freq(freq_hz: int | None) -> str:
    if freq_hz is None:
        return "default"
    if freq_hz >= 1_000_000_000:
        return f"{freq_hz / 1e9:.3f}GHz"
    if freq_hz >= 1_000_000:
        return f"{freq_hz / 1e6:.0f}MHz"
    return str(freq_hz)


def parse_freq_hz(value: str) -> int:
    s = value.strip().lower()
    if s.endswith("ghz"):
        return int(float(s[:-3]) * 1e9)
    if s.endswith("mhz"):
        return int(float(s[:-3]) * 1e6)
    if s.endswith("khz"):
        return int(float(s[:-3]) * 1e3)
    return int(s)


def default_cpu_matches() -> tuple[bool, dict[str, dict[str, int]]]:
    status = cpu_mod.read_cpu_policy_status()
    matched = True
    for policy in cpu_mod.cpu_policies():
        expected_min = cpu_mod.read_int(policy / "cpuinfo_min_freq")
        expected_max = cpu_mod.read_int(policy / "cpuinfo_max_freq")
        row = status[policy.name]
        if row["min_khz"] != expected_min or row["max_khz"] != expected_max:
            matched = False
            break
    return matched, status


def default_gpu_matches() -> tuple[bool, dict[str, int]]:
    status: dict[str, int] = {}
    matched = True
    for clk in gpu_mod.GPU_DEBUG_CLKS:
        clk_dir = gpu_mod.BPMP_CLK_ROOT / clk
        lock = gpu_mod.read_int(clk_dir / "mrq_rate_locked")
        rate = gpu_mod.read_int(clk_dir / "rate")
        status[f"{clk}_locked"] = lock
        status[f"{clk}_rate"] = rate
        if lock != 0 or rate != 0:
            matched = False
    status["gpu_cur_freq_hz"] = gpu_mod.read_int(gpu_mod.GPU_CUR_FREQ)
    return matched, status


def default_emc_matches() -> tuple[bool, dict[str, int]]:
    status = emc_mod.read_emc_status()
    matched = (
        status["mrq_rate_locked"] == 0
        and status["state"] == 0
        and status["bwmgr_halt"] == 0
    )
    return matched, status


def apply_single_axis_state(state: SingleAxisState) -> None:
    if state.axis == "cpu":
        if state.freq_hz is None:
            cpu_mod.unlock_cpu_freq_all()
        else:
            cpu_mod.set_cpu_freq_all(state.freq_hz)
        return
    if state.axis == "gpu":
        if state.freq_hz is None:
            gpu_mod.unlock_gpu_freq()
        else:
            gpu_mod.set_gpu_freq(state.freq_hz)
        return
    if state.axis == "emc":
        if state.freq_hz is None:
            emc_mod.unlock_emc_freq()
        else:
            emc_mod.set_emc_freq(state.freq_hz)
        return
    raise ValueError(f"Unknown axis: {state.axis}")


def single_axis_matches(state: SingleAxisState) -> tuple[bool, dict[str, object]]:
    if state.axis == "cpu":
        if state.freq_hz is None:
            matched, status = default_cpu_matches()
            return matched, status
        matched, status = cpu_mod.cpu_lock_matches(state.freq_hz)
        return matched, status
    if state.axis == "gpu":
        if state.freq_hz is None:
            matched, status = default_gpu_matches()
            return matched, status
        matched, cur_freq, rates = gpu_mod.gpu_lock_matches(state.freq_hz)
        status = {"gpu_cur_freq_hz": cur_freq, **rates}
        return matched, status
    if state.axis == "emc":
        if state.freq_hz is None:
            matched, status = default_emc_matches()
            return matched, status
        matched, status = emc_mod.emc_lock_matches(state.freq_hz)
        return matched, status
    raise ValueError(f"Unknown axis: {state.axis}")


def apply_combo_state(state: ComboState) -> None:
    if state.cpu_hz is None:
        cpu_mod.unlock_cpu_freq_all()
    else:
        cpu_mod.set_cpu_freq_all(state.cpu_hz)

    if state.gpu_hz is None:
        gpu_mod.unlock_gpu_freq()
    else:
        gpu_mod.set_gpu_freq(state.gpu_hz)

    if state.emc_hz is None:
        emc_mod.unlock_emc_freq()
    else:
        emc_mod.set_emc_freq(state.emc_hz)


def combo_matches(state: ComboState) -> tuple[bool, dict[str, object]]:
    cpu_ok, cpu_status = (
        default_cpu_matches()
        if state.cpu_hz is None
        else cpu_mod.cpu_lock_matches(state.cpu_hz)
    )
    if state.gpu_hz is None:
        gpu_ok, gpu_status = default_gpu_matches()
    else:
        gpu_ok, gpu_cur, gpu_rates = gpu_mod.gpu_lock_matches(state.gpu_hz)
        gpu_status = {"gpu_cur_freq_hz": gpu_cur, **gpu_rates}
    emc_ok, emc_status = (
        default_emc_matches()
        if state.emc_hz is None
        else emc_mod.emc_lock_matches(state.emc_hz)
    )
    status = {
        "cpu_status": cpu_status,
        "gpu_status": gpu_status,
        "emc_status": emc_status,
    }
    return cpu_ok and gpu_ok and emc_ok, status


def wait_for_state(
    matcher: Callable[[], tuple[bool, dict[str, object]]],
    *,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
) -> tuple[bool, int | None, int | None, dict[str, object]]:
    deadline = time.monotonic() + timeout_s
    first_match_ns: int | None = None
    stable_match_ns: int | None = None
    consecutive = 0
    last_status: dict[str, object] = {}
    while time.monotonic() < deadline:
        matched, status = matcher()
        now_ns = time.monotonic_ns()
        last_status = status
        if matched:
            if first_match_ns is None:
                first_match_ns = now_ns
            consecutive += 1
            if consecutive >= stable_samples:
                stable_match_ns = now_ns
                return True, first_match_ns, stable_match_ns, last_status
        else:
            consecutive = 0
        time.sleep(poll_ms / 1000.0)
    return False, first_match_ns, stable_match_ns, last_status


def measure_transition(
    apply_fn: Callable[[object], None],
    match_fn: Callable[[object], tuple[bool, dict[str, object]]],
    from_state: object,
    to_state: object,
    *,
    ensure_timeout_s: float,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
    warmup_sleep_s: float,
) -> dict[str, object]:
    apply_fn(from_state)
    ok, _, _, _ = wait_for_state(
        lambda: match_fn(from_state),
        timeout_s=ensure_timeout_s,
        poll_ms=poll_ms,
        stable_samples=stable_samples,
    )
    if not ok:
        raise RuntimeError(f"Failed to settle source state: {from_state}")

    if warmup_sleep_s > 0:
        time.sleep(warmup_sleep_s)

    t0 = time.monotonic_ns()
    apply_fn(to_state)
    t1 = time.monotonic_ns()
    ok, first_match_ns, stable_match_ns, status = wait_for_state(
        lambda: match_fn(to_state),
        timeout_s=timeout_s,
        poll_ms=poll_ms,
        stable_samples=stable_samples,
    )

    write_overhead_ms = (t1 - t0) / 1e6
    feedback_ms = None if first_match_ns is None else (first_match_ns - t0) / 1e6
    stable_ms = None if stable_match_ns is None else (stable_match_ns - t0) / 1e6
    return {
        "success": int(ok),
        "write_overhead_ms": write_overhead_ms,
        "feedback_ms": feedback_ms,
        "stable_ms": stable_ms,
        "status_repr": repr(status),
    }


def build_single_axis_states(
    axis: str,
    freqs_hz: list[int],
    *,
    include_default: bool,
) -> list[SingleAxisState]:
    states: list[SingleAxisState] = []
    if include_default:
        states.append(SingleAxisState(axis=axis, freq_hz=None))
    states.extend(SingleAxisState(axis=axis, freq_hz=f) for f in freqs_hz)
    return states


def build_combo_states(
    cpu_freqs: list[int],
    gpu_freqs: list[int],
    emc_freqs: list[int],
    *,
    include_default: bool,
) -> list[ComboState]:
    cpu_levels: list[int | None] = ([None] if include_default else []) + cpu_freqs
    gpu_levels: list[int | None] = ([None] if include_default else []) + gpu_freqs
    emc_levels: list[int | None] = ([None] if include_default else []) + emc_freqs
    states: list[ComboState] = []
    for cpu_hz in cpu_levels:
        for gpu_hz in gpu_levels:
            for emc_hz in emc_levels:
                states.append(ComboState(cpu_hz=cpu_hz, gpu_hz=gpu_hz, emc_hz=emc_hz))
    return states


def save_results_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "mode",
        "from_label",
        "to_label",
        "success",
        "write_overhead_ms",
        "feedback_ms",
        "stable_ms",
        "status_repr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_heatmap(
    png_path: Path,
    labels: list[str],
    rows: list[dict[str, object]],
    value_key: str,
    title: str,
) -> None:
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for row in rows:
        if not row["success"]:
            continue
        value = row[value_key]
        if value in (None, ""):
            continue
        matrix[label_to_idx[row["from_label"]], label_to_idx[row["to_label"]]] = float(value)

    fig_w = max(8, len(labels) * 0.9)
    fig_h = max(6, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, interpolation="nearest", cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_key.replace("_", " "))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if math.isnan(matrix[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > np.nanmean(matrix) else "black",
                fontsize=9,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("thor_measurements/freq_switch_overhead"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cpu", "gpu", "emc", "combo"),
        default=("cpu", "gpu", "emc"),
    )
    parser.add_argument("--cpu-freq", action="append", default=[])
    parser.add_argument("--gpu-freq", action="append", default=[])
    parser.add_argument("--emc-freq", action="append", default=[])
    parser.add_argument("--include-default", dest="include_default", action="store_true", default=True)
    parser.add_argument("--no-default", dest="include_default", action="store_false")
    parser.add_argument("--poll-ms", type=float, default=5.0)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--ensure-timeout-s", type=float, default=3.0)
    parser.add_argument("--warmup-sleep-s", type=float, default=0.02)
    parser.add_argument(
        "--combo-transition-mode",
        choices=("from-default", "pairwise"),
        default="from-default",
    )
    return parser.parse_args()


def resolved_cpu_freqs(args: argparse.Namespace) -> list[int]:
    requested = [parse_freq_hz(v) for v in args.cpu_freq] if args.cpu_freq else [
        648_000_000,
        972_000_000,
        1_242_000_000,
        1_566_000_000,
        1_836_000_000,
        2_160_000_000,
        2_430_000_000,
        2_601_000_000,
    ]
    try:
        available = cpu_mod.available_cpu_freqs()
        return [cpu_mod.nearest_available(v, available) for v in requested]
    except Exception:
        return requested


def resolved_gpu_freqs(args: argparse.Namespace) -> list[int]:
    requested = [parse_freq_hz(v) for v in args.gpu_freq] if args.gpu_freq else [
        504_000_000,
        702_000_000,
        900_000_000,
        1_107_000_000,
        1_305_000_000,
        1_503_000_000,
    ]
    try:
        available = gpu_mod.available_gpu_freqs()
        return [gpu_mod.nearest_available(v, available) for v in requested]
    except Exception:
        return requested


def resolved_emc_freqs(args: argparse.Namespace) -> list[int]:
    requested = [parse_freq_hz(v) for v in args.emc_freq] if args.emc_freq else [
        665_600_000,
        2_750_000_000,
        3_200_000_000,
        4_266_000_000,
    ]
    try:
        available = emc_mod.available_emc_freqs()
        return [emc_mod.nearest_available(v, available) for v in requested]
    except Exception:
        return requested


def run_single_axis_mode(
    out_dir: Path,
    states: list[SingleAxisState],
    *,
    axis: str,
    ensure_timeout_s: float,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
    warmup_sleep_s: float,
) -> None:
    print(f"[INFO] Measuring {axis} transitions: {len(states)} states")
    rows: list[dict[str, object]] = []
    for from_state in states:
        for to_state in states:
            print(f"[INFO] {axis}: {from_state.label} -> {to_state.label}")
            result = measure_transition(
                apply_single_axis_state,
                single_axis_matches,
                from_state,
                to_state,
                ensure_timeout_s=ensure_timeout_s,
                timeout_s=timeout_s,
                poll_ms=poll_ms,
                stable_samples=stable_samples,
                warmup_sleep_s=warmup_sleep_s,
            )
            rows.append(
                {
                    "mode": axis,
                    "from_label": from_state.label,
                    "to_label": to_state.label,
                    **result,
                }
            )

    mode_dir = out_dir / axis
    mode_dir.mkdir(parents=True, exist_ok=True)
    csv_path = mode_dir / "transition_results.csv"
    save_results_csv(csv_path, rows)
    labels = [state.label for state in states]
    for metric in ("write_overhead_ms", "feedback_ms", "stable_ms"):
        save_matrix_heatmap(
            mode_dir / f"{metric}.png",
            labels,
            rows,
            metric,
            title=f"{axis.upper()} switch overhead: {metric}",
        )


def run_combo_mode(
    out_dir: Path,
    states: list[ComboState],
    *,
    transition_mode: str,
    ensure_timeout_s: float,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
    warmup_sleep_s: float,
) -> None:
    print(f"[INFO] Measuring combo transitions: {len(states)} states, mode={transition_mode}")
    rows: list[dict[str, object]] = []
    default_state = ComboState(cpu_hz=None, gpu_hz=None, emc_hz=None)

    if transition_mode == "pairwise":
        pairs = [(a, b) for a in states for b in states]
    else:
        pairs = [(default_state, state) for state in states]
        pairs += [(state, default_state) for state in states if state != default_state]

    for from_state, to_state in pairs:
        print(f"[INFO] combo: {from_state.label} -> {to_state.label}")
        result = measure_transition(
            apply_combo_state,
            combo_matches,
            from_state,
            to_state,
            ensure_timeout_s=ensure_timeout_s,
            timeout_s=timeout_s,
            poll_ms=poll_ms,
            stable_samples=stable_samples,
            warmup_sleep_s=warmup_sleep_s,
        )
        rows.append(
            {
                "mode": "combo",
                "from_label": from_state.label,
                "to_label": to_state.label,
                **result,
            }
        )

    mode_dir = out_dir / "combo"
    mode_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(mode_dir / "transition_results.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    need_cpu = any(mode in {"cpu", "combo"} for mode in args.modes)
    need_gpu = any(mode in {"gpu", "combo"} for mode in args.modes)
    need_emc = any(mode in {"emc", "combo"} for mode in args.modes)

    cpu_freqs: list[int] = resolved_cpu_freqs(args) if need_cpu else []
    gpu_freqs: list[int] = resolved_gpu_freqs(args) if need_gpu else []
    emc_freqs: list[int] = resolved_emc_freqs(args) if need_emc else []

    if cpu_freqs:
        print("[INFO] CPU_FREQS=", " ".join(format_state_freq(v) for v in cpu_freqs))
    if gpu_freqs:
        print("[INFO] GPU_FREQS=", " ".join(format_state_freq(v) for v in gpu_freqs))
    if emc_freqs:
        print("[INFO] EMC_FREQS=", " ".join(format_state_freq(v) for v in emc_freqs))

    try:
        for mode in args.modes:
            if mode == "cpu":
                run_single_axis_mode(
                    out_dir,
                    build_single_axis_states("cpu", cpu_freqs, include_default=args.include_default),
                    axis="cpu",
                    ensure_timeout_s=args.ensure_timeout_s,
                    timeout_s=args.timeout_s,
                    poll_ms=args.poll_ms,
                    stable_samples=args.stable_samples,
                    warmup_sleep_s=args.warmup_sleep_s,
                )
            elif mode == "gpu":
                run_single_axis_mode(
                    out_dir,
                    build_single_axis_states("gpu", gpu_freqs, include_default=args.include_default),
                    axis="gpu",
                    ensure_timeout_s=args.ensure_timeout_s,
                    timeout_s=args.timeout_s,
                    poll_ms=args.poll_ms,
                    stable_samples=args.stable_samples,
                    warmup_sleep_s=args.warmup_sleep_s,
                )
            elif mode == "emc":
                run_single_axis_mode(
                    out_dir,
                    build_single_axis_states("emc", emc_freqs, include_default=args.include_default),
                    axis="emc",
                    ensure_timeout_s=args.ensure_timeout_s,
                    timeout_s=args.timeout_s,
                    poll_ms=args.poll_ms,
                    stable_samples=args.stable_samples,
                    warmup_sleep_s=args.warmup_sleep_s,
                )
            elif mode == "combo":
                combo_states = build_combo_states(
                    cpu_freqs,
                    gpu_freqs,
                    emc_freqs,
                    include_default=args.include_default,
                )
                run_combo_mode(
                    out_dir,
                    combo_states,
                    transition_mode=args.combo_transition_mode,
                    ensure_timeout_s=args.ensure_timeout_s,
                    timeout_s=args.timeout_s,
                    poll_ms=args.poll_ms,
                    stable_samples=args.stable_samples,
                    warmup_sleep_s=args.warmup_sleep_s,
                )
    finally:
        cpu_mod.unlock_cpu_freq_all()
        gpu_mod.unlock_gpu_freq()
        emc_mod.unlock_emc_freq()


if __name__ == "__main__":
    main()
