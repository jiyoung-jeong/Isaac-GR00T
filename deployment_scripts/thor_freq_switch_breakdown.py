#!/usr/bin/env python3
"""
Break down Jetson Thor manual frequency switch timing.

This is intended for phase-level DVFS decisions.  It separates the time spent in
individual sysfs/debugfs writes, feedback reads, post-write waiting, and the
"usable" time where the requested state has been observed for N consecutive
polls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment_scripts import thor_cpufreq_power_sweep as cpu_mod
from deployment_scripts import thor_emcfreq_power_sweep as emc_mod
from deployment_scripts import thor_gpufreq_power_sweep as gpu_mod


@dataclass(frozen=True)
class State:
    axis: str
    freq_hz: int | None

    @property
    def label(self) -> str:
        return format_freq(self.freq_hz)


@dataclass(frozen=True)
class WriteOp:
    name: str
    path: Path
    value: str


def parse_freq_hz(value: str) -> int:
    s = value.strip().lower()
    if s.endswith("ghz"):
        return int(float(s[:-3]) * 1e9)
    if s.endswith("mhz"):
        return int(float(s[:-3]) * 1e6)
    if s.endswith("khz"):
        return int(float(s[:-3]) * 1e3)
    return int(s)


def format_freq(freq_hz: int | None) -> str:
    if freq_hz is None:
        return "default"
    if freq_hz >= 1_000_000_000:
        return f"{freq_hz / 1e9:.3f}GHz"
    return f"{freq_hz // 1_000_000}MHz"


def timed_read_int(path: Path) -> tuple[int, int, int]:
    start = time.monotonic_ns()
    value = int(path.read_text(encoding="utf-8").strip())
    end = time.monotonic_ns()
    return value, start, end


def timed_write(op: WriteOp) -> dict[str, object]:
    start = time.monotonic_ns()
    op.path.write_text(op.value, encoding="utf-8")
    end = time.monotonic_ns()
    return {
        "kind": "write",
        "name": op.name,
        "path": str(op.path),
        "value": op.value,
        "start_ns": start,
        "end_ns": end,
        "duration_ms": (end - start) / 1e6,
    }


def cpu_write_plan(freq_hz: int | None) -> list[WriteOp]:
    ops: list[WriteOp] = []
    for policy in cpu_mod.cpu_policies():
        if freq_hz is None:
            target_min = cpu_mod.read_int(policy / "cpuinfo_min_freq")
            target_max = cpu_mod.read_int(policy / "cpuinfo_max_freq")
        else:
            target_min = target_max = freq_hz // 1000

        cur_min = cpu_mod.read_int(policy / "scaling_min_freq")
        cur_max = cpu_mod.read_int(policy / "scaling_max_freq")
        min_op = WriteOp(f"{policy.name}:scaling_min_freq", policy / "scaling_min_freq", str(target_min))
        max_op = WriteOp(f"{policy.name}:scaling_max_freq", policy / "scaling_max_freq", str(target_max))

        # Avoid transient min > max when moving upward, and max < min when moving downward.
        if target_min > cur_max:
            ordered = [max_op, min_op]
        elif target_max < cur_min:
            ordered = [min_op, max_op]
        else:
            ordered = [min_op, max_op]
        ops.extend(ordered)
    return ops


def gpu_write_plan(freq_hz: int | None, runtime_write_mode: str) -> list[WriteOp]:
    ops: list[WriteOp] = []
    if freq_hz is None:
        for clk in gpu_mod.GPU_DEBUG_CLKS:
            clk_dir = gpu_mod.BPMP_CLK_ROOT / clk
            ops.append(WriteOp(f"{clk}:mrq_rate_locked", clk_dir / "mrq_rate_locked", "0"))
            ops.append(WriteOp(f"{clk}:rate", clk_dir / "rate", "0"))
        return ops

    for clk in gpu_mod.GPU_DEBUG_CLKS:
        clk_dir = gpu_mod.BPMP_CLK_ROOT / clk
        if runtime_write_mode == "full":
            ops.append(WriteOp(f"{clk}:mrq_rate_locked", clk_dir / "mrq_rate_locked", "1"))
        ops.append(WriteOp(f"{clk}:rate", clk_dir / "rate", str(freq_hz)))
    return ops


def emc_write_plan(freq_hz: int | None, runtime_write_mode: str) -> list[WriteOp]:
    if freq_hz is None:
        return [
            WriteOp("emc:mrq_rate_locked", emc_mod.EMC_MRQ_LOCKED, "0"),
            WriteOp("emc:state", emc_mod.EMC_STATE, "0"),
            WriteOp("bwmgr:bwmgr_halt", emc_mod.BWMGR_HALT, "0"),
        ]

    ops: list[WriteOp] = []
    if runtime_write_mode == "full":
        ops.extend(
            [
                WriteOp("emc:mrq_rate_locked", emc_mod.EMC_MRQ_LOCKED, "1"),
                WriteOp("emc:state", emc_mod.EMC_STATE, "1"),
                WriteOp("bwmgr:bwmgr_halt", emc_mod.BWMGR_HALT, "1"),
            ]
        )
    ops.append(WriteOp("emc:rate", emc_mod.EMC_RATE, str(freq_hz)))
    return ops


def write_plan(state: State, runtime_write_mode: str) -> list[WriteOp]:
    if state.axis == "cpu":
        return cpu_write_plan(state.freq_hz)
    if state.axis == "gpu":
        return gpu_write_plan(state.freq_hz, runtime_write_mode)
    if state.axis == "emc":
        return emc_write_plan(state.freq_hz, runtime_write_mode)
    raise ValueError(f"unknown axis: {state.axis}")


def read_cpu_status() -> tuple[dict[str, object], list[dict[str, object]], float]:
    status: dict[str, object] = {}
    events: list[dict[str, object]] = []
    total_ns = 0
    for policy in cpu_mod.cpu_policies():
        row: dict[str, int] = {}
        for key, rel in (
            ("min_khz", "scaling_min_freq"),
            ("max_khz", "scaling_max_freq"),
            ("cur_khz", "cpuinfo_cur_freq"),
        ):
            value, start, end = timed_read_int(policy / rel)
            total_ns += end - start
            row[key] = value
            events.append(
                {
                    "kind": "read",
                    "name": f"{policy.name}:{rel}",
                    "path": str(policy / rel),
                    "value": value,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ms": (end - start) / 1e6,
                }
            )
        status[policy.name] = row
    return status, events, total_ns / 1e6


def read_gpu_status() -> tuple[dict[str, object], list[dict[str, object]], float]:
    status: dict[str, object] = {}
    events: list[dict[str, object]] = []
    total_ns = 0
    paths = [("gpu_cur_freq_hz", gpu_mod.GPU_CUR_FREQ)]
    for clk in gpu_mod.GPU_DEBUG_CLKS:
        paths.append((f"{clk}_rate_hz", gpu_mod.BPMP_CLK_ROOT / clk / "rate"))
        paths.append((f"{clk}_mrq_rate_locked", gpu_mod.BPMP_CLK_ROOT / clk / "mrq_rate_locked"))
    for name, path in paths:
        value, start, end = timed_read_int(path)
        total_ns += end - start
        status[name] = value
        events.append(
            {
                "kind": "read",
                "name": name,
                "path": str(path),
                "value": value,
                "start_ns": start,
                "end_ns": end,
                "duration_ms": (end - start) / 1e6,
            }
        )
    return status, events, total_ns / 1e6


def read_emc_status() -> tuple[dict[str, object], list[dict[str, object]], float]:
    status: dict[str, object] = {}
    events: list[dict[str, object]] = []
    total_ns = 0
    paths = [
        ("rate_hz", emc_mod.EMC_RATE),
        ("mrq_rate_locked", emc_mod.EMC_MRQ_LOCKED),
        ("state", emc_mod.EMC_STATE),
        ("bwmgr_halt", emc_mod.BWMGR_HALT),
    ]
    for name, path in paths:
        value, start, end = timed_read_int(path)
        total_ns += end - start
        status[name] = value
        events.append(
            {
                "kind": "read",
                "name": name,
                "path": str(path),
                "value": value,
                "start_ns": start,
                "end_ns": end,
                "duration_ms": (end - start) / 1e6,
            }
        )
    return status, events, total_ns / 1e6


def read_status(axis: str) -> tuple[dict[str, object], list[dict[str, object]], float]:
    if axis == "cpu":
        return read_cpu_status()
    if axis == "gpu":
        return read_gpu_status()
    if axis == "emc":
        return read_emc_status()
    raise ValueError(f"unknown axis: {axis}")


def matches(state: State, status: dict[str, object]) -> bool:
    if state.axis == "cpu":
        if state.freq_hz is None:
            for policy in cpu_mod.cpu_policies():
                row = status[policy.name]
                if not isinstance(row, dict):
                    return False
                if row["min_khz"] != cpu_mod.read_int(policy / "cpuinfo_min_freq"):
                    return False
                if row["max_khz"] != cpu_mod.read_int(policy / "cpuinfo_max_freq"):
                    return False
            return True
        target_khz = state.freq_hz // 1000
        return all(
            isinstance(row, dict)
            and row.get("min_khz") == target_khz
            and row.get("max_khz") == target_khz
            for row in status.values()
        )
    if state.axis == "gpu":
        if state.freq_hz is None:
            return all(status.get(f"{clk}_mrq_rate_locked") == 0 for clk in gpu_mod.GPU_DEBUG_CLKS)
        return all(
            status.get(f"{clk}_rate_hz") == state.freq_hz
            and status.get(f"{clk}_mrq_rate_locked") == 1
            for clk in gpu_mod.GPU_DEBUG_CLKS
        )
    if state.axis == "emc":
        if state.freq_hz is None:
            return (
                status.get("mrq_rate_locked") == 0
                and status.get("state") == 0
                and status.get("bwmgr_halt") == 0
            )
        return (
            status.get("rate_hz") == state.freq_hz
            and status.get("mrq_rate_locked") == 1
            and status.get("state") == 1
            and status.get("bwmgr_halt") == 1
        )
    raise ValueError(f"unknown axis: {state.axis}")


def apply_state(state: State, runtime_write_mode: str) -> list[dict[str, object]]:
    return [timed_write(op) for op in write_plan(state, runtime_write_mode)]


def wait_for_state(
    state: State,
    *,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
) -> tuple[bool, int | None, int | None, dict[str, object], list[dict[str, object]], float, int]:
    deadline = time.monotonic() + timeout_s
    first_match_ns: int | None = None
    stable_match_ns: int | None = None
    consecutive = 0
    last_status: dict[str, object] = {}
    events: list[dict[str, object]] = []
    read_total_ms = 0.0
    polls = 0

    while time.monotonic() < deadline:
        poll_start_ns = time.monotonic_ns()
        status, read_events, read_ms = read_status(state.axis)
        poll_end_ns = time.monotonic_ns()
        polls += 1
        read_total_ms += read_ms
        last_status = status
        matched = matches(state, status)
        events.append(
            {
                "kind": "poll",
                "matched": matched,
                "poll_index": polls,
                "start_ns": poll_start_ns,
                "end_ns": poll_end_ns,
                "duration_ms": (poll_end_ns - poll_start_ns) / 1e6,
                "read_total_ms": read_ms,
                "status": status,
            }
        )
        events.extend(read_events)

        if matched:
            if first_match_ns is None:
                first_match_ns = poll_end_ns
            consecutive += 1
            if consecutive >= stable_samples:
                stable_match_ns = poll_end_ns
                return True, first_match_ns, stable_match_ns, last_status, events, read_total_ms, polls
        else:
            consecutive = 0

        time.sleep(poll_ms / 1000.0)

    return False, first_match_ns, stable_match_ns, last_status, events, read_total_ms, polls


def measure_transition(
    from_state: State,
    to_state: State,
    *,
    ensure_timeout_s: float,
    timeout_s: float,
    poll_ms: float,
    stable_samples: int,
    warmup_sleep_s: float,
    runtime_write_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pre_events = apply_state(from_state, "full")
    ok, _, _, _, ensure_events, _, _ = wait_for_state(
        from_state,
        timeout_s=ensure_timeout_s,
        poll_ms=poll_ms,
        stable_samples=stable_samples,
    )
    if not ok:
        raise RuntimeError(f"failed to settle source state: {from_state}")

    if warmup_sleep_s > 0:
        time.sleep(warmup_sleep_s)

    t0 = time.monotonic_ns()
    write_events = apply_state(to_state, runtime_write_mode)
    writes_done_ns = time.monotonic_ns()
    ok, first_match_ns, stable_match_ns, status, read_events, read_total_ms, polls = wait_for_state(
        to_state,
        timeout_s=timeout_s,
        poll_ms=poll_ms,
        stable_samples=stable_samples,
    )

    write_total_ms = (writes_done_ns - t0) / 1e6
    per_write_ms = sum(float(e["duration_ms"]) for e in write_events)
    feedback_ms = math.nan if first_match_ns is None else (first_match_ns - t0) / 1e6
    post_write_to_first_match_ms = (
        math.nan if first_match_ns is None else (first_match_ns - writes_done_ns) / 1e6
    )
    stable_ms = math.nan if stable_match_ns is None else (stable_match_ns - t0) / 1e6
    stable_extra_ms = (
        math.nan
        if first_match_ns is None or stable_match_ns is None
        else (stable_match_ns - first_match_ns) / 1e6
    )

    row = {
        "axis": to_state.axis,
        "from_label": from_state.label,
        "to_label": to_state.label,
        "success": int(ok),
        "write_total_ms": write_total_ms,
        "per_write_sum_ms": per_write_ms,
        "write_op_count": len(write_events),
        "feedback_ms": feedback_ms,
        "post_write_to_first_match_ms": post_write_to_first_match_ms,
        "stable_extra_ms": stable_extra_ms,
        "usable_stable_ms": stable_ms,
        "poll_count": polls,
        "read_total_ms": read_total_ms,
        "read_mean_per_poll_ms": read_total_ms / polls if polls else math.nan,
        "last_status_json": json.dumps(status, sort_keys=True),
    }
    events: list[dict[str, object]] = []
    for phase, phase_events in (
        ("settle_source_write", pre_events),
        ("settle_source_poll", ensure_events),
        ("transition_write", write_events),
        ("transition_poll", read_events),
    ):
        for event in phase_events:
            event = dict(event)
            event["phase"] = phase
            event["axis"] = to_state.axis
            event["from_label"] = from_state.label
            event["to_label"] = to_state.label
            event["t0_ns"] = t0
            events.append(event)
    return row, events


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "axis",
        "from_label",
        "to_label",
        "success",
        "write_total_ms",
        "per_write_sum_ms",
        "write_op_count",
        "feedback_ms",
        "post_write_to_first_match_ms",
        "stable_extra_ms",
        "usable_stable_ms",
        "poll_count",
        "read_total_ms",
        "read_mean_per_poll_ms",
        "last_status_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_events_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")


def available_freqs(axis: str) -> list[int]:
    if axis == "cpu":
        return cpu_mod.available_cpu_freqs()
    if axis == "gpu":
        return gpu_mod.available_gpu_freqs()
    if axis == "emc":
        return emc_mod.available_emc_freqs()
    raise ValueError(f"unknown axis: {axis}")


def default_freqs(axis: str) -> list[int]:
    if axis == "cpu":
        requested = [
            648_000_000,
            972_000_000,
            1_242_000_000,
            1_566_000_000,
            1_836_000_000,
            2_160_000_000,
            2_430_000_000,
            2_601_000_000,
        ]
    elif axis == "gpu":
        requested = [
            504_000_000,
            702_000_000,
            900_000_000,
            1_107_000_000,
            1_305_000_000,
            1_503_000_000,
        ]
    elif axis == "emc":
        requested = [
            665_600_000,
            2_750_000_000,
            3_200_000_000,
            4_266_000_000,
        ]
    else:
        raise ValueError(f"unknown axis: {axis}")
    avail = available_freqs(axis)
    out: list[int] = []
    for freq in requested:
        rounded = min(avail, key=lambda f: (abs(f - freq), f))
        if rounded not in out:
            out.append(rounded)
    return out


def build_states(axis: str, freq_args: list[str], include_default: bool) -> list[State]:
    freqs = [parse_freq_hz(v) for v in freq_args] if freq_args else default_freqs(axis)
    if freq_args:
        avail = available_freqs(axis)
        freqs = [min(avail, key=lambda f: (abs(f - freq), f)) for freq in freqs]
    states: list[State] = []
    if include_default:
        states.append(State(axis, None))
    for freq in freqs:
        state = State(axis, freq)
        if state not in states:
            states.append(state)
    return states


def snapshot_initial() -> dict[str, object]:
    cpu, _, _ = read_cpu_status()
    gpu, _, _ = read_gpu_status()
    emc, _, _ = read_emc_status()
    return {"cpu": cpu, "gpu": gpu, "emc": emc}


def restore_initial(snapshot: dict[str, object]) -> None:
    cpu_status = snapshot.get("cpu", {})
    if isinstance(cpu_status, dict):
        for policy in cpu_mod.cpu_policies():
            row = cpu_status.get(policy.name)
            if isinstance(row, dict):
                (policy / "scaling_min_freq").write_text(str(row["min_khz"]), encoding="utf-8")
                (policy / "scaling_max_freq").write_text(str(row["max_khz"]), encoding="utf-8")

    gpu_status = snapshot.get("gpu", {})
    if isinstance(gpu_status, dict):
        for clk in gpu_mod.GPU_DEBUG_CLKS:
            rate = int(gpu_status.get(f"{clk}_rate_hz", 0))
            locked = int(gpu_status.get(f"{clk}_mrq_rate_locked", 0))
            clk_dir = gpu_mod.BPMP_CLK_ROOT / clk
            (clk_dir / "mrq_rate_locked").write_text(str(locked), encoding="utf-8")
            (clk_dir / "rate").write_text(str(rate), encoding="utf-8")

    emc_status = snapshot.get("emc", {})
    if isinstance(emc_status, dict):
        emc_mod.EMC_MRQ_LOCKED.write_text(str(emc_status.get("mrq_rate_locked", 0)), encoding="utf-8")
        emc_mod.EMC_STATE.write_text(str(emc_status.get("state", 0)), encoding="utf-8")
        emc_mod.BWMGR_HALT.write_text(str(emc_status.get("bwmgr_halt", 0)), encoding="utf-8")
        if int(emc_status.get("mrq_rate_locked", 0)):
            emc_mod.EMC_RATE.write_text(str(emc_status.get("rate_hz", 0)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("thor_measurements/freq_switch_breakdown"))
    parser.add_argument("--modes", nargs="+", choices=("cpu", "gpu", "emc"), default=("cpu", "gpu", "emc"))
    parser.add_argument("--cpu-freq", action="append", default=[])
    parser.add_argument("--gpu-freq", action="append", default=[])
    parser.add_argument("--emc-freq", action="append", default=[])
    parser.add_argument("--include-default", action="store_true", default=False)
    parser.add_argument("--pairwise", action="store_true", help="measure all from/to pairs; default is adjacent")
    parser.add_argument("--poll-ms", type=float, default=2.0)
    parser.add_argument("--stable-samples", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--ensure-timeout-s", type=float, default=3.0)
    parser.add_argument("--warmup-sleep-s", type=float, default=0.02)
    parser.add_argument(
        "--runtime-write-mode",
        choices=("full", "rate-only"),
        default="full",
        help="for GPU/EMC fixed states, write all lock controls or only rate during transition",
    )
    parser.add_argument(
        "--restore",
        choices=("initial", "none"),
        default="initial",
        help="restore initial CPU/GPU/EMC lock state after the run",
    )
    return parser.parse_args()


def axis_freq_args(args: argparse.Namespace, axis: str) -> list[str]:
    return {
        "cpu": args.cpu_freq,
        "gpu": args.gpu_freq,
        "emc": args.emc_freq,
    }[axis]


def transition_pairs(states: list[State], pairwise: bool) -> list[tuple[State, State]]:
    if pairwise:
        return [(src, dst) for src in states for dst in states if src != dst]
    return list(zip(states, states[1:]))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    events_path = args.out_dir / "transition_events.jsonl"
    if events_path.exists():
        events_path.unlink()

    initial = snapshot_initial()
    (args.out_dir / "initial_state.json").write_text(json.dumps(initial, indent=2, sort_keys=True), encoding="utf-8")

    try:
        for axis in args.modes:
            states = build_states(axis, axis_freq_args(args, axis), args.include_default)
            pairs = transition_pairs(states, args.pairwise)
            print(f"[INFO] {axis}: states={','.join(s.label for s in states)} pairs={len(pairs)}")
            for src, dst in pairs:
                print(f"[INFO] {axis}: {src.label} -> {dst.label}")
                row, events = measure_transition(
                    src,
                    dst,
                    ensure_timeout_s=args.ensure_timeout_s,
                    timeout_s=args.timeout_s,
                    poll_ms=args.poll_ms,
                    stable_samples=args.stable_samples,
                    warmup_sleep_s=args.warmup_sleep_s,
                    runtime_write_mode=args.runtime_write_mode,
                )
                summary_rows.append(row)
                append_events_jsonl(events_path, events)
                save_csv(args.out_dir / "transition_breakdown.csv", summary_rows)
    finally:
        if args.restore == "initial":
            restore_initial(initial)


if __name__ == "__main__":
    main()
