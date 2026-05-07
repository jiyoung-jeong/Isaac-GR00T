#!/usr/bin/env python3
"""
Jetson Thor CPU/GPU/EMC full-combination sweep helper.

This script:
  - runs one all-default baseline first (optional),
  - can also treat default as one selectable level for CPU / GPU / EMC,
  - then iterates every requested CPU x GPU x EMC frequency combination,
  - verifies CPU/GPU/EMC locks before running the benchmark,
  - logs power telemetry during each run,
  - derives phase latency / power / energy from NVTX CSV logs,
  - writes one per-combination folder plus a top-level summary CSV.

The telemetry / phase-analysis behavior intentionally matches the existing
single-axis sweep scripts so we can compare results directly.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import signal
import threading
import time
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment_scripts import thor_cpufreq_power_sweep as cpu_mod
from deployment_scripts import thor_emcfreq_power_sweep as emc_mod
from deployment_scripts import thor_gpufreq_power_sweep as gpu_mod


DEFAULT_CPU_FREQS = [
    "648MHz",
    "756MHz",
    "864MHz",
    "972MHz",
    "1.08GHz",
    "1.188GHz",
    "1.296GHz",
    "1.404GHz",
    "1.512GHz",
    "1.620GHz",
    "1.728GHz",
    "1.836GHz",
    "1.944GHz",
    "2.052GHz",
    "2.160GHz",
    "2.268GHz",
    "2.376GHz",
    "2.484GHz",
    "2.601GHz",
]

DEFAULT_GPU_FREQS = [
    "504MHz",
    "603MHz",
    "702MHz",
    "801MHz",
    "900MHz",
    "999MHz",
    "1.107GHz",
    "1.206GHz",
    "1.305GHz",
    "1.404GHz",
    "1.503GHz",
    "1.575GHz",
]

DEFAULT_EMC_FREQS = [
    "665.6MHz",
    "1.2GHz",
    "1.6GHz",
    "2.133GHz",
    "2.75GHz",
    "3.2GHz",
    "3.75GHz",
    "4.266GHz",
]


def build_explicit_list(
    requested: list[str] | None,
    defaults: list[str],
    available: list[int],
    parser_fn,
) -> list[int]:
    raw = requested if requested else defaults
    rounded = [gpu_mod.nearest_available(parser_fn(v), available) for v in raw]
    out: list[int] = []
    for freq in rounded:
        if freq not in out:
            out.append(freq)
    return out


def format_combo(cpu_hz: int | None, gpu_hz: int | None, emc_hz: int | None) -> str:
    if cpu_hz is None and gpu_hz is None and emc_hz is None:
        return "default"
    return "_".join(
        [
            f"cpufreq_{cpu_mod.format_freq(cpu_hz)}",
            f"gpufreq_{gpu_mod.format_freq(gpu_hz)}",
            f"emcfreq_{emc_mod.format_freq(emc_hz)}",
        ]
    )


def unlock_all() -> None:
    cpu_mod.unlock_cpu_freq_all()
    gpu_mod.unlock_gpu_freq()
    emc_mod.unlock_emc_freq()


def sample_row(
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> dict[str, object]:
    return cpu_mod.sample_row(ina3221, gpu_ch, cpu_ch)


def run_combo(
    cpu_hz: int | None,
    gpu_hz: int | None,
    emc_hz: int | None,
    args: argparse.Namespace,
    ina3221: Path | None,
    gpu_ch: int | None,
    cpu_ch: int | None,
) -> dict[str, object]:
    combo_name = format_combo(cpu_hz, gpu_hz, emc_hz)
    out_dir = args.out_dir / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "telemetry_raw.csv"
    nvtx_csv = out_dir / "nvtx_ranges.csv"
    nvtx_csv.unlink(missing_ok=True)

    gpu_actual = -1
    emc_actual = -1
    cpu_actual = -1
    gpu_lock_ok = True
    emc_lock_ok = True
    cpu_lock_ok = True
    gpu_status: dict[str, object] = {}
    emc_status: dict[str, object] = {}
    cpu_status: dict[str, object] = {}

    if cpu_hz is None and gpu_hz is None and emc_hz is None:
        print("[INFO] Running all-default governor baseline")
        unlock_all()
        time.sleep(args.settle_s)
        gpu_status = {
            "cur_freq_hz": gpu_mod.read_int(gpu_mod.GPU_CUR_FREQ),
            "debug_rates": gpu_mod.read_gpu_debug_rates(),
        }
        emc_status = emc_mod.read_emc_status()
        cpu_status = cpu_mod.read_cpu_policy_status()
    else:
        if gpu_hz is not None:
            print(f"[INFO] Locking GPU to {gpu_hz} Hz ({gpu_mod.format_freq(gpu_hz)})")
            gpu_cur, gpu_rates, gpu_lock_ok = gpu_mod.lock_gpu_freq_with_retry(
                gpu_hz,
                args.settle_s,
                args.lock_retries,
                args.lock_verify_s,
            )
            gpu_actual = gpu_hz if gpu_lock_ok else gpu_cur
            gpu_status = {"cur_freq_hz": gpu_cur, "debug_rates": gpu_rates}
        else:
            gpu_mod.unlock_gpu_freq()
            time.sleep(args.settle_s)
            gpu_status = {
                "cur_freq_hz": gpu_mod.read_int(gpu_mod.GPU_CUR_FREQ),
                "debug_rates": gpu_mod.read_gpu_debug_rates(),
            }

        if emc_hz is not None:
            print(f"[INFO] Locking EMC to {emc_hz} Hz ({emc_mod.format_freq(emc_hz)})")
            emc_status, emc_lock_ok = emc_mod.lock_emc_freq_with_retry(
                emc_hz,
                args.settle_s,
                args.lock_retries,
                args.lock_verify_s,
            )
            emc_actual = emc_status.get("rate_hz", -1)
        else:
            emc_mod.unlock_emc_freq()
            time.sleep(args.settle_s)
            emc_status = emc_mod.read_emc_status()

        if cpu_hz is not None:
            print(f"[INFO] Locking CPU to {cpu_hz} Hz ({cpu_mod.format_freq(cpu_hz)})")
            cpu_status, cpu_lock_ok = cpu_mod.lock_cpu_freq_with_retry(
                cpu_hz,
                args.settle_s,
                args.lock_retries,
                args.lock_verify_s,
            )
            cpu_actual = cpu_hz
        else:
            cpu_mod.unlock_cpu_freq_all()
            time.sleep(args.settle_s)
            cpu_status = cpu_mod.read_cpu_policy_status()

    lock_ok = gpu_lock_ok and emc_lock_ok and cpu_lock_ok
    lock_text = "\n".join(
        [
            f"lock_ok={1 if lock_ok else 0}",
            f"mode={'default' if combo_name == 'default' else 'locked_combo'}",
            f"requested_cpu_hz={cpu_hz if cpu_hz is not None else -1}",
            f"actual_cpu_hz={cpu_actual}",
            f"requested_gpu_hz={gpu_hz if gpu_hz is not None else -1}",
            f"actual_gpu_hz={gpu_actual}",
            f"requested_emc_hz={emc_hz if emc_hz is not None else -1}",
            f"actual_emc_hz={emc_actual}",
            f"cpu_lock_ok={1 if cpu_lock_ok else 0}",
            f"gpu_lock_ok={1 if gpu_lock_ok else 0}",
            f"emc_lock_ok={1 if emc_lock_ok else 0}",
            f"cpu_status={cpu_status}",
            f"gpu_status={gpu_status}",
            f"emc_status={emc_status}",
        ]
    )
    (out_dir / "lock_status.txt").write_text(lock_text + "\n", encoding="utf-8")

    if not lock_ok:
        msg = (
            "One or more frequency locks failed: "
            f"cpu_lock_ok={cpu_lock_ok}, gpu_lock_ok={gpu_lock_ok}, emc_lock_ok={emc_lock_ok}"
        )
        if args.skip_failed_locks:
            print(f"[WARN] {msg}; skipping benchmark")
            return {
                "mode": "default" if combo_name == "default" else "locked_combo",
                "lock_ok": 0,
                "cpu_lock_ok": 1 if cpu_lock_ok else 0,
                "gpu_lock_ok": 1 if gpu_lock_ok else 0,
                "emc_lock_ok": 1 if emc_lock_ok else 0,
                "requested_cpu_hz": cpu_hz if cpu_hz is not None else -1,
                "actual_cpu_hz": cpu_actual,
                "requested_gpu_hz": gpu_hz if gpu_hz is not None else -1,
                "actual_gpu_hz": gpu_actual,
                "requested_emc_hz": emc_hz if emc_hz is not None else -1,
                "actual_emc_hz": emc_actual,
                "combo_name": combo_name,
                "raw_csv": "",
                "nvtx_csv": "",
                "phase_metrics_csv": "",
                "phase_summary_csv": "",
                "lock_message": msg,
            }
        raise RuntimeError(msg)

    if args.command:
        stop = [False]
        thread = threading.Thread(
            target=cpu_mod.telemetry_loop,
            args=(raw_csv, stop, args.interval_ms, ina3221, gpu_ch, cpu_ch),
            daemon=True,
        )
        thread.start()
        env = os.environ.copy()
        env["THOR_CPU_FREQ_HZ"] = str(cpu_hz if cpu_hz is not None else "default")
        env["THOR_GPU_FREQ_HZ"] = str(gpu_hz if gpu_hz is not None else "default")
        env["THOR_EMC_FREQ_HZ"] = str(emc_hz if emc_hz is not None else "default")
        env["THOR_TELEMETRY_CSV"] = str(raw_csv)
        env["NVTX_RANGES_CSV"] = str(nvtx_csv)
        start_ns = time.monotonic_ns()
        rc = cpu_mod.run_command(args.command, env)
        end_ns = time.monotonic_ns()
        stop[0] = True
        thread.join(timeout=2.0)
        (out_dir / "command_status.txt").write_text(
            f"returncode={rc}\nstart_ns={start_ns}\nend_ns={end_ns}\n",
            encoding="utf-8",
        )
    else:
        cpu_mod.run_telemetry_for_duration(
            raw_csv,
            args.duration_s,
            args.interval_ms,
            ina3221,
            gpu_ch,
            cpu_ch,
        )

    summary = cpu_mod.summarize_csv(raw_csv, cpu_hz, cpu_actual, gpu_hz, emc_hz)
    summary["mode"] = "default" if combo_name == "default" else "locked_combo"
    summary["lock_ok"] = 1
    summary["cpu_lock_ok"] = 1 if cpu_lock_ok else 0
    summary["gpu_lock_ok"] = 1 if gpu_lock_ok else 0
    summary["emc_lock_ok"] = 1 if emc_lock_ok else 0
    summary["combo_name"] = combo_name
    summary["requested_cpu_hz"] = cpu_hz if cpu_hz is not None else -1
    summary["actual_cpu_hz"] = cpu_actual
    summary["requested_gpu_hz"] = gpu_hz if gpu_hz is not None else -1
    summary["actual_gpu_hz"] = gpu_actual
    summary["requested_emc_hz"] = emc_hz if emc_hz is not None else -1
    summary["actual_emc_hz"] = emc_actual
    phase_rows = cpu_mod.write_phase_metrics(raw_csv, nvtx_csv, out_dir, cpu_hz, cpu_actual)
    summary["phase_metrics_csv"] = str(out_dir / "phase_metrics.csv") if phase_rows else ""
    summary["phase_summary_csv"] = str(out_dir / "phase_summary.csv") if phase_rows else ""
    summary["nvtx_csv"] = str(nvtx_csv) if nvtx_csv.exists() else ""
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep every CPU x GPU x EMC frequency combination on Jetson Thor."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("thor_measurements/combo_sweep"))
    parser.add_argument("--interval-ms", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--cpu-freq", action="append", help="Explicit CPU frequency, e.g. 1.62GHz")
    parser.add_argument("--gpu-freq", action="append", help="Explicit GPU frequency, e.g. 1.305GHz")
    parser.add_argument("--emc-freq", action="append", help="Explicit EMC frequency, e.g. 3.2GHz")
    parser.add_argument("--include-default", action="store_true", help="Run one all-default baseline before the locked combinations.")
    parser.add_argument(
        "--include-axis-defaults",
        action="store_true",
        help="Include default/unlocked as one selectable level for each of CPU, GPU, and EMC in the Cartesian sweep.",
    )
    parser.add_argument("--lock-retries", type=int, default=3)
    parser.add_argument("--lock-verify-s", type=float, default=0.5)
    parser.add_argument("--skip-failed-locks", action="store_true")
    parser.add_argument("--unlock-at-end", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional benchmark command after '--'. Telemetry runs while this command executes.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()

    try:
        cpu_available = cpu_mod.available_cpu_freqs()
        gpu_available = gpu_mod.available_gpu_freqs()
        emc_available = emc_mod.available_emc_freqs()
    except Exception:
        if not args.dry_run:
            raise
        cpu_available = [cpu_mod.parse_freq_hz(v) for v in (args.cpu_freq or DEFAULT_CPU_FREQS)]
        gpu_available = [gpu_mod.parse_freq_hz(v) for v in (args.gpu_freq or DEFAULT_GPU_FREQS)]
        emc_available = [emc_mod.parse_freq_hz(v) for v in (args.emc_freq or DEFAULT_EMC_FREQS)]

    cpu_freqs = build_explicit_list(args.cpu_freq, DEFAULT_CPU_FREQS, cpu_available, cpu_mod.parse_freq_hz)
    gpu_freqs = build_explicit_list(args.gpu_freq, DEFAULT_GPU_FREQS, gpu_available, gpu_mod.parse_freq_hz)
    emc_freqs = build_explicit_list(args.emc_freq, DEFAULT_EMC_FREQS, emc_available, emc_mod.parse_freq_hz)

    combos: list[tuple[int | None, int | None, int | None]] = []
    seen: set[tuple[int | None, int | None, int | None]] = set()
    if args.include_default:
        combos.append((None, None, None))
        seen.add((None, None, None))

    cpu_options: list[int | None] = ([None] if args.include_axis_defaults else []) + cpu_freqs
    gpu_options: list[int | None] = ([None] if args.include_axis_defaults else []) + gpu_freqs
    emc_options: list[int | None] = ([None] if args.include_axis_defaults else []) + emc_freqs

    for gpu_hz in gpu_options:
        for emc_hz in emc_options:
            for cpu_hz in cpu_options:
                combo = (cpu_hz, gpu_hz, emc_hz)
                if combo in seen:
                    continue
                seen.add(combo)
                combos.append(combo)
    if args.max_combos is not None:
        combos = combos[: args.max_combos]

    print(f"[INFO] CPU frequencies ({len(cpu_freqs)}): {[cpu_mod.format_freq(v) for v in cpu_freqs]}")
    print(f"[INFO] GPU frequencies ({len(gpu_freqs)}): {[gpu_mod.format_freq(v) for v in gpu_freqs]}")
    print(f"[INFO] EMC frequencies ({len(emc_freqs)}): {[emc_mod.format_freq(v) for v in emc_freqs]}")
    print(f"[INFO] Include axis defaults: {args.include_axis_defaults}")
    print(f"[INFO] Total runs: {len(combos)}")
    if args.dry_run:
        for idx, (cpu_hz, gpu_hz, emc_hz) in enumerate(combos, start=1):
            print(
                f"  [{idx:04d}] "
                f"CPU={cpu_mod.format_freq(cpu_hz)} "
                f"GPU={gpu_mod.format_freq(gpu_hz)} "
                f"EMC={emc_mod.format_freq(emc_hz)}"
            )
        return 0

    ina3221 = cpu_mod.find_hwmon(cpu_mod.INA3221_ROOT, "VDD_GPU")
    gpu_ch = cpu_mod.find_ina3221_channel(ina3221, "VDD_GPU") if ina3221 else None
    cpu_ch = cpu_mod.find_ina3221_channel(ina3221, "VDD_CPU_SOC_MSS") if ina3221 else None
    print(f"[INFO] INA3221={ina3221}, GPU channel={gpu_ch}, CPU channel={cpu_ch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    try:
        for idx, combo in enumerate(combos, start=1):
            cpu_hz, gpu_hz, emc_hz = combo
            print(
                f"[INFO] Running combo {idx}/{len(combos)}: "
                f"CPU={cpu_mod.format_freq(cpu_hz)} "
                f"GPU={gpu_mod.format_freq(gpu_hz)} "
                f"EMC={emc_mod.format_freq(emc_hz)}"
            )
            summary_rows.append(run_combo(cpu_hz, gpu_hz, emc_hz, args, ina3221, gpu_ch, cpu_ch))
    finally:
        if args.unlock_at_end:
            print("[INFO] Unlocking CPU/GPU/EMC frequency constraints")
            unlock_all()

    summary_path = args.out_dir / "summary.csv"
    fieldnames = sorted({k for row in summary_rows for k in row})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[INFO] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
