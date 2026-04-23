#!/usr/bin/env python3
"""
Jetson Thor GPU frequency sweep helper.

This script:
  - locks gpu_gpc0/1/2 to selected BPMP clock rates,
  - verifies the requested/current GPU clock through sysfs/debugfs,
  - logs VDD_GPU power and frequency telemetry at a fixed interval,
  - optionally runs a benchmark command once per frequency,
  - writes per-frequency raw telemetry and a summary CSV.

Run inside a container with writable /sys/kernel/debug, or on the host.
For Docker, setting clocks requires debugfs to be mounted read/write.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable


GPU_DEVFREQ = Path("/sys/class/devfreq/gpu-gpc-0")
GPU_AVAILABLE_FREQS = GPU_DEVFREQ / "available_frequencies"
GPU_CUR_FREQ = GPU_DEVFREQ / "cur_freq"
GPU_DEBUG_CLKS = ("gpu_gpc0", "gpu_gpc1", "gpu_gpc2")
BPMP_CLK_ROOT = Path("/sys/kernel/debug/bpmp/debug/clk")
INA3221_ROOT = Path("/sys/bus/i2c/devices/2-0040/hwmon")
INA238_ROOT = Path("/sys/bus/i2c/devices/2-0044/hwmon")


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def read_int(path: Path, default: int = -1) -> int:
    try:
        return int(read_text(path))
    except ValueError:
        return default


def find_hwmon(root: Path, required_label: str | None = None) -> Path | None:
    if not root.is_dir():
        return None
    for hwmon in sorted(root.glob("hwmon*")):
        if not hwmon.is_dir():
            continue
        if required_label is None:
            return hwmon
        for label in hwmon.glob("in*_label"):
            if read_text(label) == required_label:
                return hwmon
    return None


def find_ina3221_channel(hwmon: Path, label_text: str) -> int | None:
    for label in sorted(hwmon.glob("in*_label")):
        if read_text(label) == label_text:
            name = label.name
            return int(name.removeprefix("in").removesuffix("_label"))
    return None


def read_power_from_channel(hwmon: Path | None, channel: int | None) -> float:
    if hwmon is None or channel is None:
        return float("nan")
    voltage_mv = read_int(hwmon / f"in{channel}_input", 0)
    current_ma = read_int(hwmon / f"curr{channel}_input", 0)
    return (voltage_mv * current_ma) / 1e6


def read_vin_power_w() -> float:
    ina238 = find_hwmon(INA238_ROOT)
    if ina238 is not None:
        power_uw = read_int(ina238 / "power1_input", -1)
        if power_uw >= 0:
            return power_uw / 1e6
    return float("nan")


def available_gpu_freqs() -> list[int]:
    freqs = [int(x) for x in read_text(GPU_AVAILABLE_FREQS).split()]
    if not freqs:
        raise RuntimeError(f"No GPU frequencies found at {GPU_AVAILABLE_FREQS}")
    return sorted(freqs)


def nearest_available(target_hz: int, freqs: list[int]) -> int:
    return min(freqs, key=lambda f: (abs(f - target_hz), f))


def build_freq_list(args: argparse.Namespace, available: list[int]) -> list[int | None]:
    out: list[int | None] = []
    if args.include_default:
        out.append(None)

    if args.freq:
        requested = [parse_freq_hz(v) for v in args.freq]
    else:
        requested = list(range(args.start_hz, args.stop_hz + 1, args.step_hz))
        if requested[-1] != args.stop_hz:
            requested.append(args.stop_hz)
    rounded = [nearest_available(v, available) for v in requested]
    for freq in rounded:
        if freq not in out:
            out.append(freq)
    return out


def parse_freq_hz(value: str) -> int:
    s = value.strip().lower()
    if s.endswith("ghz"):
        return int(float(s[:-3]) * 1e9)
    if s.endswith("mhz"):
        return int(float(s[:-3]) * 1e6)
    return int(s)


def format_freq(freq_hz: int | None) -> str:
    if freq_hz is None:
        return "default"
    if freq_hz >= 1_000_000_000:
        return f"{freq_hz / 1e9:.3f}GHz"
    return f"{freq_hz // 1_000_000}MHz"


def set_gpu_freq(freq_hz: int) -> None:
    for clk in GPU_DEBUG_CLKS:
        clk_dir = BPMP_CLK_ROOT / clk
        write_text(clk_dir / "mrq_rate_locked", "1")
        write_text(clk_dir / "rate", str(freq_hz))


def unlock_gpu_freq() -> None:
    for clk in GPU_DEBUG_CLKS:
        clk_dir = BPMP_CLK_ROOT / clk
        try:
            write_text(clk_dir / "mrq_rate_locked", "0")
            write_text(clk_dir / "rate", "0")
        except OSError:
            pass


def read_gpu_debug_rates() -> dict[str, int]:
    return {clk: read_int(BPMP_CLK_ROOT / clk / "rate") for clk in GPU_DEBUG_CLKS}


def locked_debug_rate(freq_hz: int, rates: dict[str, int]) -> int:
    return freq_hz if all(rate == freq_hz for rate in rates.values()) else read_int(GPU_CUR_FREQ)


def gpu_lock_matches(freq_hz: int) -> tuple[bool, int, dict[str, int]]:
    cur_freq = read_int(GPU_CUR_FREQ)
    rates = read_gpu_debug_rates()
    # BPMP debugfs is the source of truth for the requested clock lock. On Thor,
    # devfreq cur_freq can continue to report a different instantaneous/derived
    # value even after gpu_gpc0/1/2 rates are locked.
    matched = all(rate == freq_hz for rate in rates.values())
    return matched, cur_freq, rates


def lock_gpu_freq_with_retry(
    freq_hz: int,
    settle_s: float,
    retries: int,
    verify_s: float,
) -> tuple[int, dict[str, int], bool]:
    attempts = max(1, retries + 1)
    last_cur = -1
    last_rates: dict[str, int] = {}
    for attempt in range(1, attempts + 1):
        set_gpu_freq(freq_hz)
        time.sleep(settle_s)
        matched, cur_freq, rates = gpu_lock_matches(freq_hz)
        last_cur = cur_freq
        last_rates = rates
        print(
            f"[INFO] lock check {attempt}/{attempts}: "
            f"cur_freq={cur_freq}, debug_rates={rates}"
        )
        if matched and cur_freq != freq_hz:
            print(
                f"[WARN] devfreq cur_freq={cur_freq} differs from requested "
                f"{freq_hz}; accepting BPMP debug_rates as lock source"
            )
        if not matched:
            unlock_gpu_freq()
            time.sleep(0.2)
            continue

        deadline = time.monotonic() + verify_s
        stable = True
        while time.monotonic() < deadline:
            matched, cur_freq, rates = gpu_lock_matches(freq_hz)
            last_cur = cur_freq
            last_rates = rates
            if not matched:
                stable = False
                print(
                    f"[WARN] lock drift during verify: "
                    f"cur_freq={cur_freq}, debug_rates={rates}"
                )
                break
            if cur_freq != freq_hz:
                print(
                    f"[WARN] devfreq cur_freq={cur_freq} differs during verify; "
                    f"BPMP debug_rates remain locked"
                )
            time.sleep(0.1)
        if stable:
            return last_cur, last_rates, True

        unlock_gpu_freq()
        time.sleep(0.2)

    return last_cur, last_rates, False


def sample_row(ina3221: Path | None, gpu_ch: int | None, cpu_ch: int | None) -> dict[str, object]:
    rates = read_gpu_debug_rates()
    return {
        "ts_ns": time.monotonic_ns(),
        "vdd_gpu_w": read_power_from_channel(ina3221, gpu_ch),
        "vdd_cpu_soc_mss_w": read_power_from_channel(ina3221, cpu_ch),
        "vin_w": read_vin_power_w(),
        "gpu_cur_freq_hz": read_int(GPU_CUR_FREQ),
        "gpu_gpc0_hz": rates["gpu_gpc0"],
        "gpu_gpc1_hz": rates["gpu_gpc1"],
        "gpu_gpc2_hz": rates["gpu_gpc2"],
    }


def telemetry_loop(
    csv_path: Path,
    stop_flag: list[bool],
    interval_ms: float,
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> None:
    fields = [
        "ts_ns",
        "vdd_gpu_w",
        "vdd_cpu_soc_mss_w",
        "vin_w",
        "gpu_cur_freq_hz",
        "gpu_gpc0_hz",
        "gpu_gpc1_hz",
        "gpu_gpc2_hz",
    ]
    interval_s = interval_ms / 1000.0
    next_t = time.monotonic()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while not stop_flag[0]:
            writer.writerow(sample_row(ina3221, gpu_ch, cpu_ch))
            f.flush()
            next_t += interval_s
            time.sleep(max(0.0, next_t - time.monotonic()))


def run_telemetry_for_duration(
    csv_path: Path,
    duration_s: float,
    interval_ms: float,
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> None:
    stop = [False]

    def handle_stop(*_: object) -> None:
        stop[0] = True

    old_int = signal.signal(signal.SIGINT, handle_stop)
    end_t = time.monotonic() + duration_s
    fields = [
        "ts_ns",
        "vdd_gpu_w",
        "vdd_cpu_soc_mss_w",
        "vin_w",
        "gpu_cur_freq_hz",
        "gpu_gpc0_hz",
        "gpu_gpc1_hz",
        "gpu_gpc2_hz",
    ]
    interval_s = interval_ms / 1000.0
    next_t = time.monotonic()
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while time.monotonic() < end_t and not stop[0]:
                writer.writerow(sample_row(ina3221, gpu_ch, cpu_ch))
                f.flush()
                next_t += interval_s
                time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        signal.signal(signal.SIGINT, old_int)


def summarize_csv(csv_path: Path, requested_hz: int | None, actual_hz: int) -> dict[str, object]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if len(rows) < 2:
        return {"samples": len(rows), "requested_hz": requested_hz, "actual_hz": actual_hz}

    def values(name: str) -> list[float]:
        out = []
        for row in rows:
            try:
                v = float(row[name])
            except (ValueError, KeyError):
                continue
            if v == v:
                out.append(v)
        return out

    ts = values("ts_ns")
    gpu_p = values("vdd_gpu_w")
    gpu_f = values("gpu_cur_freq_hz")
    duration_s = (ts[-1] - ts[0]) / 1e9 if len(ts) >= 2 else 0.0
    energy_j = trapezoid_energy(rows, "vdd_gpu_w")
    return {
        "requested_hz": requested_hz if requested_hz is not None else -1,
        "actual_hz": actual_hz,
        "samples": len(rows),
        "duration_s": duration_s,
        "gpu_power_mean_w": mean(gpu_p),
        "gpu_power_p95_w": percentile(gpu_p, 95),
        "gpu_power_max_w": max(gpu_p) if gpu_p else float("nan"),
        "gpu_energy_j": energy_j,
        "gpu_freq_mean_hz": mean(gpu_f),
        "raw_csv": str(csv_path),
    }


def load_telemetry(csv_path: Path) -> list[dict[str, float]]:
    rows = []
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = float("nan")
        rows.append(parsed)
    return rows


def load_nvtx_ranges(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []

    stacks: dict[str, list[int]] = defaultdict(list)
    ranges = []
    for row in csv.reader(csv_path.open(encoding="utf-8")):
        if len(row) < 2:
            continue
        try:
            ts_ns = int(row[0])
        except ValueError:
            continue
        event = row[1]
        if event.endswith("_START"):
            label = event[: -len("_START")]
            stacks[label].append(ts_ns)
        elif event.endswith("_END"):
            label = event[: -len("_END")]
            if stacks[label]:
                start_ns = stacks[label].pop()
                ranges.append(
                    {
                        "label": label,
                        "start_ns": start_ns,
                        "end_ns": ts_ns,
                        "duration_ms": (ts_ns - start_ns) / 1e6,
                    }
                )
    return sorted(ranges, key=lambda r: int(r["start_ns"]))


def integrate_between(
    telemetry_rows: list[dict[str, float]], start_ns: int, end_ns: int, power_col: str
) -> tuple[float, float, int]:
    seg = [r for r in telemetry_rows if start_ns <= r["ts_ns"] <= end_ns]
    if len(seg) < 2:
        return float("nan"), float("nan"), len(seg)
    energy = 0.0
    for prev, cur in zip(seg, seg[1:]):
        p0 = prev.get(power_col, float("nan"))
        p1 = cur.get(power_col, float("nan"))
        if p0 != p0 or p1 != p1:
            continue
        energy += ((p0 + p1) * 0.5) * ((cur["ts_ns"] - prev["ts_ns"]) * 1e-9)
    duration_s = (end_ns - start_ns) * 1e-9
    avg_power = energy / duration_s if duration_s > 0 else float("nan")
    return energy, avg_power, len(seg)


def write_phase_metrics(
    telemetry_csv: Path,
    nvtx_csv: Path,
    out_dir: Path,
    requested_hz: int | None,
    actual_hz: int,
) -> list[dict[str, object]]:
    telemetry_rows = load_telemetry(telemetry_csv)
    ranges = load_nvtx_ranges(nvtx_csv)
    phase_labels = {
        "VLA/get_action",
        "VLA/backbone",
        "VLA/ViT",
        "VLA/LLM",
        "VLA/action_head",
        "VLA/action_head/inference",
        "VLA/action_head/DiT_step",
        "VLA/action_head/TensorRT",
        "VLA/action_head/TensorRT_enqueue",
    }
    rows = []
    inference_id = 0
    for r in ranges:
        label = str(r["label"])
        if label == "VLA/get_action":
            inference_id += 1
        if label not in phase_labels:
            continue
        start_ns = int(r["start_ns"])
        end_ns = int(r["end_ns"])
        e_gpu, p_gpu, samples = integrate_between(
            telemetry_rows, start_ns, end_ns, "vdd_gpu_w"
        )
        e_cpu, p_cpu, _ = integrate_between(
            telemetry_rows, start_ns, end_ns, "vdd_cpu_soc_mss_w"
        )
        e_vin, p_vin, _ = integrate_between(telemetry_rows, start_ns, end_ns, "vin_w")
        rows.append(
            {
                "requested_hz": requested_hz,
                "actual_hz": actual_hz,
                "inference_id": inference_id,
                "phase": label,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "latency_ms": r["duration_ms"],
                "telemetry_samples": samples,
                "gpu_energy_j": e_gpu,
                "gpu_avg_power_w": p_gpu,
                "cpu_soc_mss_energy_j": e_cpu,
                "cpu_soc_mss_avg_power_w": p_cpu,
                "vin_energy_j": e_vin,
                "vin_avg_power_w": p_vin,
            }
        )

    phase_csv = out_dir / "phase_metrics.csv"
    if rows:
        with phase_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if int(row["inference_id"]) > 1:
                grouped[str(row["phase"])].append(row)
        summary_rows = []
        for phase, phase_rows in sorted(grouped.items()):
            summary_rows.append(
                {
                    "requested_hz": requested_hz,
                    "actual_hz": actual_hz,
                    "phase": phase,
                    "n": len(phase_rows),
                    "latency_ms_mean": mean(float(r["latency_ms"]) for r in phase_rows),
                    "gpu_avg_power_w_mean": mean(
                        float(r["gpu_avg_power_w"]) for r in phase_rows
                    ),
                    "gpu_energy_j_mean": mean(float(r["gpu_energy_j"]) for r in phase_rows),
                    "vin_avg_power_w_mean": mean(
                        float(r["vin_avg_power_w"]) for r in phase_rows
                    ),
                    "vin_energy_j_mean": mean(float(r["vin_energy_j"]) for r in phase_rows),
                }
            )
        if summary_rows:
            with (out_dir / "phase_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
    return rows


def trapezoid_energy(rows: list[dict[str, str]], power_col: str) -> float:
    energy = 0.0
    prev_t = None
    prev_p = None
    for row in rows:
        try:
            t = float(row["ts_ns"])
            p = float(row[power_col])
        except (ValueError, KeyError):
            continue
        if p != p:
            continue
        if prev_t is not None and prev_p is not None:
            energy += ((prev_p + p) * 0.5) * ((t - prev_t) * 1e-9)
        prev_t = t
        prev_p = p
    return energy


def mean(values_: Iterable[float]) -> float:
    vals = list(values_)
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values_: Iterable[float], pct: float) -> float:
    vals = sorted(values_)
    if not vals:
        return float("nan")
    idx = (len(vals) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def run_command(command: list[str], env: dict[str, str]) -> int:
    return subprocess.call(command, env=env)


def run_one_frequency(
    freq_hz: int | None,
    args: argparse.Namespace,
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> dict[str, object]:
    out_dir = args.out_dir / f"gpufreq_{format_freq(freq_hz)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "telemetry_raw.csv"
    nvtx_csv = out_dir / "nvtx_ranges.csv"
    nvtx_csv.unlink(missing_ok=True)

    if freq_hz is None:
        print("[INFO] Running default GPU governor baseline")
        unlock_gpu_freq()
        time.sleep(args.settle_s)
        cur_freq = read_int(GPU_CUR_FREQ)
        rates = read_gpu_debug_rates()
        actual = -1
        lock_ok = True
        (out_dir / "lock_status.txt").write_text(
            f"lock_ok=1\nmode=default\nrequested_hz=-1\nactual_hz=-1\n"
            f"cur_freq_hz={cur_freq}\ndebug_rates={rates}\n",
            encoding="utf-8",
        )
    else:
        print(f"[INFO] Locking GPU to {freq_hz} Hz ({format_freq(freq_hz)})")
        cur_freq, rates, lock_ok = lock_gpu_freq_with_retry(
            freq_hz,
            args.settle_s,
            args.lock_retries,
            args.lock_verify_s,
        )
        actual = locked_debug_rate(freq_hz, rates)
        if not lock_ok:
            msg = (
                f"GPU lock failed for {freq_hz} Hz after {args.lock_retries + 1} attempts. "
                f"cur_freq={cur_freq}, debug_rates={rates}"
            )
            (out_dir / "lock_status.txt").write_text(
                f"lock_ok=0\nrequested_hz={freq_hz}\nactual_hz={actual}\n"
                f"cur_freq_hz={cur_freq}\ndebug_rates={rates}\nmessage={msg}\n",
                encoding="utf-8",
            )
            if args.skip_failed_locks:
                print(f"[WARN] {msg}; skipping benchmark")
                return {
                    "requested_hz": freq_hz,
                    "actual_hz": actual,
                    "cur_freq_hz": cur_freq,
                    "lock_ok": 0,
                    "lock_message": msg,
                    "raw_csv": "",
                    "nvtx_csv": "",
                    "phase_metrics_csv": "",
                    "phase_summary_csv": "",
                }
            raise RuntimeError(msg)

        (out_dir / "lock_status.txt").write_text(
            f"lock_ok=1\nrequested_hz={freq_hz}\nactual_hz={actual}\n"
            f"cur_freq_hz={cur_freq}\ndebug_rates={rates}\n",
            encoding="utf-8",
        )

    if args.command:
        import threading

        stop = [False]
        thread = threading.Thread(
            target=telemetry_loop,
            args=(raw_csv, stop, args.interval_ms, ina3221, gpu_ch, cpu_ch),
            daemon=True,
        )
        thread.start()
        env = os.environ.copy()
        env["THOR_GPU_FREQ_HZ"] = str(freq_hz if freq_hz is not None else "default")
        env["THOR_TELEMETRY_CSV"] = str(raw_csv)
        env["NVTX_RANGES_CSV"] = str(nvtx_csv)
        start_ns = time.monotonic_ns()
        rc = run_command(args.command, env)
        end_ns = time.monotonic_ns()
        stop[0] = True
        thread.join(timeout=2.0)
        (out_dir / "command_status.txt").write_text(
            f"returncode={rc}\nstart_ns={start_ns}\nend_ns={end_ns}\n",
            encoding="utf-8",
        )
    else:
        run_telemetry_for_duration(
            raw_csv, args.duration_s, args.interval_ms, ina3221, gpu_ch, cpu_ch
        )

    summary = summarize_csv(raw_csv, freq_hz, actual)
    summary["lock_ok"] = 1 if lock_ok else 0
    summary["initial_cur_freq_hz"] = cur_freq
    summary["mode"] = "default" if freq_hz is None else "locked"
    phase_rows = write_phase_metrics(raw_csv, nvtx_csv, out_dir, freq_hz, actual)
    summary["phase_metrics_csv"] = str(out_dir / "phase_metrics.csv") if phase_rows else ""
    summary["phase_summary_csv"] = str(out_dir / "phase_summary.csv") if phase_rows else ""
    summary["nvtx_csv"] = str(nvtx_csv) if nvtx_csv.exists() else ""
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Jetson Thor GPU frequency and log power.")
    parser.add_argument("--out-dir", type=Path, default=Path("thor_measurements/gpu_sweep"))
    parser.add_argument("--interval-ms", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--start-hz", type=int, default=405_000_000)
    parser.add_argument("--stop-hz", type=int, default=1_575_000_000)
    parser.add_argument("--step-hz", type=int, default=100_000_000)
    parser.add_argument("--freq", action="append", help="Explicit frequency, e.g. 801MHz or 1305000000.")
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Run one unlocked/default governor baseline before locked frequency runs.",
    )
    parser.add_argument(
        "--lock-retries",
        type=int,
        default=3,
        help="Number of extra attempts when GPU frequency lock verification fails.",
    )
    parser.add_argument(
        "--lock-verify-s",
        type=float,
        default=0.5,
        help="Seconds to keep verifying the GPU clock after an initial successful lock.",
    )
    parser.add_argument(
        "--skip-failed-locks",
        action="store_true",
        help="Skip a frequency instead of failing the whole sweep when lock verification fails.",
    )
    parser.add_argument("--unlock-at-end", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional benchmark command after '--'. Telemetry runs while this command executes.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    available = available_gpu_freqs()
    freqs = build_freq_list(args, available)
    print("[INFO] Frequencies:")
    for freq in freqs:
        if freq is None:
            print("  default governor")
        else:
            print(f"  {freq} ({format_freq(freq)})")
    if args.dry_run:
        return 0

    ina3221 = find_hwmon(INA3221_ROOT, "VDD_GPU")
    gpu_ch = find_ina3221_channel(ina3221, "VDD_GPU") if ina3221 else None
    cpu_ch = find_ina3221_channel(ina3221, "VDD_CPU_SOC_MSS") if ina3221 else None
    print(f"[INFO] INA3221={ina3221}, GPU channel={gpu_ch}, CPU channel={cpu_ch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.csv"
    summary_rows = []
    try:
        for freq in freqs:
            summary_rows.append(run_one_frequency(freq, args, ina3221, gpu_ch, cpu_ch))
    finally:
        if args.unlock_at_end:
            print("[INFO] Unlocking GPU clocks")
            unlock_gpu_freq()

    fieldnames = sorted({k for row in summary_rows for k in row})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[INFO] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
