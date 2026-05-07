#!/usr/bin/env python3
"""
Jetson Thor EMC frequency sweep helper.

This script:
  - locks EMC via BPMP debugfs using mrq_rate_locked/state/bwmgr_halt,
  - optionally locks GPU to a fixed frequency while sweeping EMC,
  - verifies the requested EMC clock through debugfs,
  - logs rail power telemetry and EMC/GPU rates at a fixed interval,
  - optionally runs a benchmark command once per EMC frequency,
  - writes per-frequency raw telemetry and a summary CSV,
  - derives phase latency / power / energy from NVTX CSV logs.

Notes:
  - The visible hwmon rails do not expose a dedicated "EMC-only" power rail in the
    same way that VDD_GPU is exposed. For EMC sweeps, system-level VIN and
    VDD_CPU_SOC_MSS are usually the most useful power signals to inspect.
  - Run inside a container with writable /sys/kernel/debug, or on the host.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Iterable


BPMP_DEBUG_ROOT = Path("/sys/kernel/debug/bpmp/debug")
BPMP_CLK_ROOT = BPMP_DEBUG_ROOT / "clk"
BWMGR_ROOT = BPMP_DEBUG_ROOT / "bwmgr"
EMC_ROOT = BPMP_CLK_ROOT / "emc"
GPU_DEVFREQ = Path("/sys/class/devfreq/gpu-gpc-0")
GPU_CUR_FREQ = GPU_DEVFREQ / "cur_freq"
GPU_DEBUG_CLKS = ("gpu_gpc0", "gpu_gpc1", "gpu_gpc2")
INA3221_ROOT = Path("/sys/bus/i2c/devices/2-0040/hwmon")
INA238_ROOT = Path("/sys/bus/i2c/devices/2-0044/hwmon")

EMC_DVFS = EMC_ROOT / "dvfs"
EMC_RATE = EMC_ROOT / "rate"
EMC_MRQ_LOCKED = EMC_ROOT / "mrq_rate_locked"
EMC_STATE = EMC_ROOT / "state"
BWMGR_HALT = BWMGR_ROOT / "bwmgr_halt"


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
            return int(label.name.removeprefix("in").removesuffix("_label"))
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


def available_emc_freqs() -> list[int]:
    freqs: list[int] = []
    for line in read_text(EMC_DVFS).splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            freqs.append(int(parts[1]))
        except ValueError:
            continue
    if not freqs:
        raise RuntimeError(f"No EMC frequencies found at {EMC_DVFS}")
    return sorted(dict.fromkeys(freqs))


def nearest_available(target_hz: int, freqs: list[int]) -> int:
    return min(freqs, key=lambda f: (abs(f - target_hz), f))


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


def set_emc_freq(freq_hz: int) -> None:
    write_text(EMC_MRQ_LOCKED, "1")
    write_text(EMC_STATE, "1")
    write_text(BWMGR_HALT, "1")
    write_text(EMC_RATE, str(freq_hz))


def unlock_emc_freq() -> None:
    for path, value in (
        (EMC_MRQ_LOCKED, "0"),
        (EMC_STATE, "0"),
        (BWMGR_HALT, "0"),
    ):
        try:
            write_text(path, value)
        except OSError:
            pass


def read_emc_status() -> dict[str, int]:
    return {
        "rate_hz": read_int(EMC_RATE),
        "mrq_rate_locked": read_int(EMC_MRQ_LOCKED),
        "state": read_int(EMC_STATE),
        "bwmgr_halt": read_int(BWMGR_HALT),
    }


def emc_lock_matches(freq_hz: int) -> tuple[bool, dict[str, int]]:
    status = read_emc_status()
    matched = (
        status["rate_hz"] == freq_hz
        and status["mrq_rate_locked"] == 1
        and status["state"] == 1
        and status["bwmgr_halt"] == 1
    )
    return matched, status


def lock_emc_freq_with_retry(
    freq_hz: int,
    settle_s: float,
    retries: int,
    verify_s: float,
) -> tuple[dict[str, int], bool]:
    attempts = max(1, retries + 1)
    last_status: dict[str, int] = {}
    for attempt in range(1, attempts + 1):
        set_emc_freq(freq_hz)
        time.sleep(settle_s)
        matched, status = emc_lock_matches(freq_hz)
        last_status = status
        print(f"[INFO] EMC lock check {attempt}/{attempts}: {status}")
        if not matched:
            unlock_emc_freq()
            time.sleep(0.2)
            continue

        deadline = time.monotonic() + verify_s
        stable = True
        while time.monotonic() < deadline:
            matched, status = emc_lock_matches(freq_hz)
            last_status = status
            if not matched:
                stable = False
                print(f"[WARN] EMC lock drift during verify: {status}")
                break
            time.sleep(0.1)
        if stable:
            return last_status, True

        unlock_emc_freq()
        time.sleep(0.2)

    return last_status, False


def sample_row(
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> dict[str, object]:
    gpu_rates = read_gpu_debug_rates()
    emc = read_emc_status()
    return {
        "ts_ns": time.monotonic_ns(),
        "vdd_gpu_w": read_power_from_channel(ina3221, gpu_ch),
        "vdd_cpu_soc_mss_w": read_power_from_channel(ina3221, cpu_ch),
        "vin_w": read_vin_power_w(),
        "gpu_cur_freq_hz": read_int(GPU_CUR_FREQ),
        "gpu_gpc0_hz": gpu_rates["gpu_gpc0"],
        "gpu_gpc1_hz": gpu_rates["gpu_gpc1"],
        "gpu_gpc2_hz": gpu_rates["gpu_gpc2"],
        "emc_rate_hz": emc["rate_hz"],
        "emc_mrq_rate_locked": emc["mrq_rate_locked"],
        "emc_state": emc["state"],
        "bwmgr_halt": emc["bwmgr_halt"],
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
        "emc_rate_hz",
        "emc_mrq_rate_locked",
        "emc_state",
        "bwmgr_halt",
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
        "emc_rate_hz",
        "emc_mrq_rate_locked",
        "emc_state",
        "bwmgr_halt",
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
    vals = [v for v in values_ if v == v]
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


def summarize_csv(
    csv_path: Path,
    requested_hz: int | None,
    actual_hz: int,
    fixed_gpu_hz: int | None,
) -> dict[str, object]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if len(rows) < 2:
        return {
            "samples": len(rows),
            "requested_hz": requested_hz if requested_hz is not None else -1,
            "actual_hz": actual_hz,
        }

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
    cpu_p = values("vdd_cpu_soc_mss_w")
    vin_p = values("vin_w")
    emc_r = values("emc_rate_hz")
    gpu_f = values("gpu_cur_freq_hz")
    duration_s = (ts[-1] - ts[0]) / 1e9 if len(ts) >= 2 else 0.0
    return {
        "requested_hz": requested_hz if requested_hz is not None else -1,
        "actual_hz": actual_hz,
        "fixed_gpu_hz": fixed_gpu_hz if fixed_gpu_hz is not None else -1,
        "samples": len(rows),
        "duration_s": duration_s,
        "vin_power_mean_w": mean(vin_p),
        "vin_power_p95_w": percentile(vin_p, 95),
        "vin_power_max_w": max(vin_p) if vin_p else float("nan"),
        "vin_energy_j": trapezoid_energy(rows, "vin_w"),
        "gpu_power_mean_w": mean(gpu_p),
        "gpu_power_p95_w": percentile(gpu_p, 95),
        "gpu_power_max_w": max(gpu_p) if gpu_p else float("nan"),
        "gpu_energy_j": trapezoid_energy(rows, "vdd_gpu_w"),
        "cpu_soc_mss_power_mean_w": mean(cpu_p),
        "cpu_soc_mss_power_p95_w": percentile(cpu_p, 95),
        "cpu_soc_mss_power_max_w": max(cpu_p) if cpu_p else float("nan"),
        "cpu_soc_mss_energy_j": trapezoid_energy(rows, "vdd_cpu_soc_mss_w"),
        "emc_rate_mean_hz": mean(emc_r),
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
                "requested_hz": requested_hz if requested_hz is not None else -1,
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
                    "requested_hz": requested_hz if requested_hz is not None else -1,
                    "actual_hz": actual_hz,
                    "phase": phase,
                    "n": len(phase_rows),
                    "latency_ms_mean": mean(float(r["latency_ms"]) for r in phase_rows),
                    "gpu_avg_power_w_mean": mean(float(r["gpu_avg_power_w"]) for r in phase_rows),
                    "gpu_energy_j_mean": mean(float(r["gpu_energy_j"]) for r in phase_rows),
                    "vin_avg_power_w_mean": mean(float(r["vin_avg_power_w"]) for r in phase_rows),
                    "vin_energy_j_mean": mean(float(r["vin_energy_j"]) for r in phase_rows),
                }
            )
        if summary_rows:
            with (out_dir / "phase_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
    return rows


def run_command(command: list[str], env: dict[str, str]) -> int:
    return subprocess.call(command, env=env)


def run_one_frequency(
    freq_hz: int | None,
    args: argparse.Namespace,
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> dict[str, object]:
    suffix = format_freq(freq_hz)
    if args.fixed_gpu_hz is not None:
        suffix += f"_gpufreq_{format_freq(args.fixed_gpu_hz)}"
    out_dir = args.out_dir / f"emcfreq_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "telemetry_raw.csv"
    nvtx_csv = out_dir / "nvtx_ranges.csv"
    nvtx_csv.unlink(missing_ok=True)

    if args.fixed_gpu_hz is not None:
        print(f"[INFO] Locking GPU to fixed {args.fixed_gpu_hz} Hz ({format_freq(args.fixed_gpu_hz)})")
        set_gpu_freq(args.fixed_gpu_hz)
        time.sleep(args.settle_s)

    if freq_hz is None:
        print("[INFO] Running default EMC governor baseline")
        unlock_emc_freq()
        time.sleep(args.settle_s)
        status = read_emc_status()
        actual = -1
        lock_ok = True
        (out_dir / "lock_status.txt").write_text(
            "lock_ok=1\nmode=default\nrequested_hz=-1\nactual_hz=-1\n"
            f"status={status}\nfixed_gpu_hz={args.fixed_gpu_hz if args.fixed_gpu_hz is not None else -1}\n",
            encoding="utf-8",
        )
    else:
        print(f"[INFO] Locking EMC to {freq_hz} Hz ({format_freq(freq_hz)})")
        status, lock_ok = lock_emc_freq_with_retry(
            freq_hz,
            args.settle_s,
            args.lock_retries,
            args.lock_verify_s,
        )
        actual = status.get("rate_hz", -1)
        if not lock_ok:
            msg = (
                f"EMC lock failed for {freq_hz} Hz after {args.lock_retries + 1} attempts. "
                f"status={status}"
            )
            (out_dir / "lock_status.txt").write_text(
                "lock_ok=0\n"
                f"requested_hz={freq_hz}\nactual_hz={actual}\nstatus={status}\n"
                f"fixed_gpu_hz={args.fixed_gpu_hz if args.fixed_gpu_hz is not None else -1}\n"
                f"message={msg}\n",
                encoding="utf-8",
            )
            if args.skip_failed_locks:
                print(f"[WARN] {msg}; skipping benchmark")
                return {
                    "requested_hz": freq_hz,
                    "actual_hz": actual,
                    "fixed_gpu_hz": args.fixed_gpu_hz if args.fixed_gpu_hz is not None else -1,
                    "lock_ok": 0,
                    "lock_message": msg,
                    "raw_csv": "",
                    "nvtx_csv": "",
                    "phase_metrics_csv": "",
                    "phase_summary_csv": "",
                }
            raise RuntimeError(msg)

        (out_dir / "lock_status.txt").write_text(
            "lock_ok=1\n"
            f"requested_hz={freq_hz}\nactual_hz={actual}\nstatus={status}\n"
            f"fixed_gpu_hz={args.fixed_gpu_hz if args.fixed_gpu_hz is not None else -1}\n",
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
        env["THOR_EMC_FREQ_HZ"] = str(freq_hz if freq_hz is not None else "default")
        env["THOR_FIXED_GPU_HZ"] = str(args.fixed_gpu_hz if args.fixed_gpu_hz is not None else "default")
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

    summary = summarize_csv(raw_csv, freq_hz, actual, args.fixed_gpu_hz)
    summary["lock_ok"] = 1 if lock_ok else 0
    summary["mode"] = "default" if freq_hz is None else "locked"
    phase_rows = write_phase_metrics(raw_csv, nvtx_csv, out_dir, freq_hz, actual)
    summary["phase_metrics_csv"] = str(out_dir / "phase_metrics.csv") if phase_rows else ""
    summary["phase_summary_csv"] = str(out_dir / "phase_summary.csv") if phase_rows else ""
    summary["nvtx_csv"] = str(nvtx_csv) if nvtx_csv.exists() else ""
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Jetson Thor EMC frequency and log power.")
    parser.add_argument("--out-dir", type=Path, default=Path("thor_measurements/emc_sweep"))
    parser.add_argument("--interval-ms", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--start-hz", type=int, default=665_600_000)
    parser.add_argument("--stop-hz", type=int, default=4_266_000_000)
    parser.add_argument("--step-hz", type=int, default=500_000_000)
    parser.add_argument("--freq", action="append", help="Explicit EMC frequency, e.g. 2.75GHz or 4266000000.")
    parser.add_argument("--fixed-gpu-hz", type=parse_freq_hz, default=None, help="Optional fixed GPU frequency while sweeping EMC.")
    parser.add_argument("--include-default", action="store_true", help="Run one unlocked/default EMC governor baseline before locked runs.")
    parser.add_argument("--lock-retries", type=int, default=3)
    parser.add_argument("--lock-verify-s", type=float, default=0.5)
    parser.add_argument("--skip-failed-locks", action="store_true")
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

    available = available_emc_freqs()
    freqs = build_freq_list(args, available)
    print("[INFO] EMC frequencies:")
    for freq in freqs:
        if freq is None:
            print("  default governor")
        else:
            print(f"  {freq} ({format_freq(freq)})")
    if args.fixed_gpu_hz is not None:
        print(f"[INFO] Fixed GPU frequency: {args.fixed_gpu_hz} ({format_freq(args.fixed_gpu_hz)})")
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
            print("[INFO] Unlocking EMC clocks")
            unlock_emc_freq()
            if args.fixed_gpu_hz is not None:
                print("[INFO] Unlocking fixed GPU clocks")
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
