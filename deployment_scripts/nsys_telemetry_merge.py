#!/usr/bin/env python3
"""
nsys-rep에서 NVTX 구간(Total_Policy_Inference)을 파싱하고, telemetry_raw.csv와 시간 정렬 후
구간별 infer_ms / energy(E_gpu, E_vin 등)를 계산해 CSV로 출력.

사용 조건: 동일 런에서 (1) telemetry 수집 (2) 서버가 nvtx_ranges.csv 기록 (3) nsys profile로 서버 실행
→ nvtx_ranges.csv의 첫 inference START/END(monotonic ns)와 nsys 첫 구간 start/end로 시간축 정렬.

사용 예:
  python3 deployment_scripts/nsys_telemetry_merge.py \\
    --run-dir /path/to/thor_gr00t_server_YYYYMMDD_HHMMSS \\
    --nsys-rep /path/to/robocasa_profile.nsys-rep
  # 또는
  python3 deployment_scripts/nsys_telemetry_merge.py \\
    --telemetry-csv /path/to/telemetry_raw.csv \\
    --nvtx-csv /path/to/nvtx_ranges.csv \\
    --nsys-rep /path/to/report.nsys-rep \\
    --output-dir /path/to/output
"""
import argparse
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# nsys NVTX eventType: 59 = NvtxPushPopRange, 60 = NvtxStartEndRange
NVTX_RANGE_TYPES = (59, 60)
# 우리가 쓰는 구간 이름 (inference_service.py에서 nvtx.range_push 로 기록)
# nsys export/summary 에선 ":Total_Policy_Inference" 로 저장될 수 있음
NVTX_RANGE_NAMES = ("Total_Policy_Inference", ":Total_Policy_Inference")


def export_nsys_to_sqlite(nsys_rep: Path, out_sqlite: Path) -> bool:
    """nsys export --type=sqlite 로 .nsys-rep → .sqlite 생성. 성공 여부 반환."""
    nsys_rep = nsys_rep.resolve()
    out_sqlite = out_sqlite.resolve()
    # nsys export: -o output before input file; force-overwrite for temp file reuse
    cmd = [
        "nsys", "export", "--type=sqlite",
        "--force-overwrite", "true",
        "-o", str(out_sqlite), str(nsys_rep),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            cwd=str(nsys_rep.parent),
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
            print(f"[WARN] nsys export failed (exit {result.returncode})", file=sys.stderr)
            if stderr:
                print(f"  stderr: {stderr}", file=sys.stderr)
            if stdout:
                print(f"  stdout: {stdout}", file=sys.stderr)
            return False
        return out_sqlite.exists()
    except FileNotFoundError as e:
        print(f"[WARN] nsys not found (install Nsight Systems): {e}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("[WARN] nsys export timed out (120s)", file=sys.stderr)
        return False


def get_nvtx_ranges_from_sqlite(sqlite_path: Path) -> List[Tuple[int, int]]:
    """
    NVTX_EVENTS 테이블에서 Total_Policy_Inference 구간 (start, end) ns 목록을 시작 시각 순으로 반환.
    start/end는 nsys trace timestamp (nanoseconds).
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        # NVTX_EVENTS: start, end, eventType, text, textId (textId -> StringIds)
        # nsys에 따라 value가 "Total_Policy_Inference" 또는 ":Total_Policy_Inference"
        for name in NVTX_RANGE_NAMES:
            for query, params in [
                (
                    """
                    SELECT n.start, n.end FROM NVTX_EVENTS n
                    JOIN StringIds s ON n.textId = s.id
                    WHERE n.eventType IN (59, 60) AND n.end IS NOT NULL AND s.value = ?
                    ORDER BY n.start
                    """,
                    (name,),
                ),
                (
                    """
                    SELECT start, end FROM NVTX_EVENTS
                    WHERE eventType IN (59, 60) AND end IS NOT NULL AND text = ?
                    ORDER BY start
                    """,
                    (name,),
                ),
            ]:
                try:
                    cur = conn.execute(query, params)
                    rows = cur.fetchall()
                    if rows:
                        return [(int(r[0]), int(r[1])) for r in rows if r[0] is not None and r[1] is not None]
                except sqlite3.OperationalError:
                    continue
        # fallback: 모든 완료된 NVTX 구간 (이름 필터 실패 시)
        cur = conn.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE eventType IN (59, 60) AND end IS NOT NULL ORDER BY start"
        )
        rows = cur.fetchall()
        return [(int(r[0]), int(r[1])) for r in rows if r[0] is not None and r[1] is not None]
    finally:
        conn.close()


def load_nvtx_csv_for_alignment(nvtx_csv: Path) -> Tuple[List[float], List[float]]:
    """nvtx_ranges.csv에서 POLICY_INFER_START / POLICY_INFER_END 의 ts_ns (monotonic) 목록 반환."""
    if not nvtx_csv.exists():
        return [], []
    df = pd.read_csv(nvtx_csv, header=None, names=["ts_ns", "event"])
    df = df[df["event"].isin(["POLICY_INFER_START", "POLICY_INFER_END"])].sort_values("ts_ns")
    starts = df[df["event"] == "POLICY_INFER_START"]["ts_ns"].values.astype(np.float64)
    ends = df[df["event"] == "POLICY_INFER_END"]["ts_ns"].values.astype(np.float64)
    return starts.tolist(), ends.tolist()


def align_nsys_to_monotonic(
    nsys_ranges: List[Tuple[int, int]],
    mono_starts: List[float],
    mono_ends: List[float],
) -> Optional[Tuple[List[Tuple[float, float]], float, float]]:
    """
    첫 번째 inference 구간으로 nsys timestamp -> monotonic_ns 선형 보정.
    반환: (monotonic (start_ns, end_ns) 목록, scale a, offset b) 또는 None.
    mono = a * nsys + b
    """
    if not nsys_ranges or not mono_starts or not mono_ends:
        return None
    if len(mono_starts) != len(mono_ends) or len(nsys_ranges) < 1:
        return None
    # 첫 구간으로 두 점 (nsys_start, mono_start), (nsys_end, mono_end)
    nsys_s1, nsys_e1 = nsys_ranges[0]
    mono_s1 = float(mono_starts[0])
    mono_e1 = float(mono_ends[0])
    if nsys_e1 == nsys_s1:
        return None
    a = (mono_e1 - mono_s1) / (nsys_e1 - nsys_s1)
    b = mono_s1 - a * nsys_s1
    mono_ranges = [(a * s + b, a * e + b) for s, e in nsys_ranges]
    return (mono_ranges, a, b)


def compute_energy_and_duration(
    telemetry_df: pd.DataFrame,
    ranges_mono: List[Tuple[float, float]],
    ranges_nsys_ns: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    """
    각 구간에 대해:
    - duration_ms: nsys 원본 구간 길이 (ranges_nsys_ns 있으면 사용, 없으면 mono 구간 길이)
    - E_*: telemetry를 monotonic 구간 [s_ns, e_ns]에서 적분
    """
    results = []
    for i, (s_ns, e_ns) in enumerate(ranges_mono):
        if ranges_nsys_ns and i < len(ranges_nsys_ns):
            nsys_s, nsys_e = ranges_nsys_ns[i]
            dur_ms = (nsys_e - nsys_s) * 1e-6  # nsys가 보여주는 구간 길이 그대로
        else:
            dur_ms = (e_ns - s_ns) * 1e-6
        seg = telemetry_df[(telemetry_df["ts_ns"] >= s_ns) & (telemetry_df["ts_ns"] <= e_ns)].sort_values("ts_ns")
        if len(seg) < 2:
            results.append({
                "inference_id": i + 1,
                "duration_ms": dur_ms,
                "E_gpu_J": np.nan,
                "E_cpu_J": np.nan,
                "E_vin_J": np.nan,
            })
            continue
        ts = seg["ts_ns"].values.astype(np.float64)
        dt = np.diff(ts) * 1e-9
        E_gpu = np.sum(seg["vdd_gpu_W"].values[:-1] * dt)
        E_cpu = np.sum(seg["vdd_cpu_soc_mss_W"].values[:-1] * dt)
        E_vin = np.sum(seg["vin_W"].values[:-1] * dt)
        results.append({
            "inference_id": i + 1,
            "duration_ms": dur_ms,
            "E_gpu_J": float(E_gpu),
            "E_cpu_J": float(E_cpu),
            "E_vin_J": float(E_vin),
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Merge nsys-rep NVTX ranges with telemetry CSV to get infer_ms and energy per inference."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="Thor run directory (contains telemetry_raw.csv, nvtx_ranges.csv)")
    g.add_argument(
        "--telemetry-csv",
        type=Path,
        help="Path to telemetry_raw.csv (use with --nvtx-csv and --output-dir)",
    )
    parser.add_argument("--nvtx-csv", type=Path, help="Path to nvtx_ranges.csv (for alignment)")
    parser.add_argument("--nsys-rep", type=Path, required=True, help="Path to .nsys-rep file")
    parser.add_argument("--output-dir", type=Path, help="Output directory for inference_energy_nsys.csv (with --telemetry-csv)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
        telemetry_csv = run_dir / "telemetry_raw.csv"
        nvtx_csv = run_dir / "nvtx_ranges.csv"
        output_dir = run_dir
    else:
        if not args.telemetry_csv or not args.output_dir:
            parser.error("--telemetry-csv and --output-dir required when not using --run-dir")
        telemetry_csv = args.telemetry_csv
        nvtx_csv = args.nvtx_csv or Path()
        output_dir = args.output_dir

    if not telemetry_csv.exists():
        print(f"[ERROR] Telemetry not found: {telemetry_csv}", file=sys.stderr)
        return 1
    if not args.nsys_rep.exists():
        print(f"[ERROR] nsys-rep not found: {args.nsys_rep}", file=sys.stderr)
        return 1

    df = pd.read_csv(telemetry_csv)
    if "ts_ns" not in df.columns:
        print("[ERROR] telemetry_raw.csv must have ts_ns column", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        tmp_sqlite = Path(f.name)
    try:
        if not export_nsys_to_sqlite(args.nsys_rep, tmp_sqlite):
            return 1
        nsys_ranges = get_nvtx_ranges_from_sqlite(tmp_sqlite)
    finally:
        tmp_sqlite.unlink(missing_ok=True)

    if not nsys_ranges:
        print("[WARN] No NVTX ranges found in nsys-rep (look for Total_Policy_Inference).", file=sys.stderr)
        return 1

    mono_starts, mono_ends = load_nvtx_csv_for_alignment(nvtx_csv)
    aligned = align_nsys_to_monotonic(nsys_ranges, mono_starts, mono_ends)
    if aligned is None:
        print(
            "[WARN] Could not align nsys time to telemetry (need nvtx_ranges.csv with POLICY_INFER_START/END from same run).",
            file=sys.stderr,
        )
        return 1

    ranges_mono, scale, offset = aligned
    energies = compute_energy_and_duration(df, ranges_mono, nsys_ranges)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(energies)
    out_csv = output_dir / "inference_energy_nsys.csv"
    out_df.to_csv(out_csv, index=False)
    # compare_thor_runs에서 freq/W를 nsys 구간 기준으로 쓰도록 정렬된 구간 저장
    ranges_csv = output_dir / "inference_ranges_nsys_mono.csv"
    ranges_df = pd.DataFrame(
        [{"inference_id": i + 1, "start_ns": s, "end_ns": e} for i, (s, e) in enumerate(ranges_mono)]
    )
    ranges_df.to_csv(ranges_csv, index=False)
    print(f"[INFO] Wrote {out_csv} (N={len(energies)} inferences from nsys NVTX + telemetry)")
    print(f"[INFO] Wrote {ranges_csv} (monotonic ranges for freq/power in compare_thor_runs)")
    print(f"  duration_ms: min={out_df['duration_ms'].min():.1f}, max={out_df['duration_ms'].max():.1f}, mean={out_df['duration_ms'].mean():.1f}")
    print(f"  E_gpu_J: sum={out_df['E_gpu_J'].sum():.2f}, E_vin_J: sum={out_df['E_vin_J'].sum():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
