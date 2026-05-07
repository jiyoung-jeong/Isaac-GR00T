#!/usr/bin/env python3
"""
Thor 측정 런 여러 개를 비교: inference 구간(첫 번째 제외) 기준 GPU 주파수, 전력, inference 시간 요약.
- inference 시간: inference_energy_nsys.csv 가 있으면 nsys 구간 기준(첫 번째 제외), 없으면 NVTX CSV 기준.
- GPU freq / 전력: inference 2~N 구간에 해당하는 텔레메트리 샘플만 사용.
- 기본으로 compare_inference.png, compare_power.png 를 --base(또는 --output-dir)에 저장.
사용 예:
  python3 deployment_scripts/compare_thor_runs.py \\
    thor_gr00t_server_20260304_065357_gpufreq_801MB \\
    ...
  python3 deployment_scripts/compare_thor_runs.py --no-plot run1 run2  # 표만 출력
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_telemetry(run_dir: Path):
    p = run_dir / "telemetry_raw.csv"
    if not p.exists():
        return None, None, None
    df = pd.read_csv(p)
    df["t_s"] = (df["ts_ns"] - df["ts_ns"].iloc[0]) * 1e-9
    return df, run_dir / "markers.csv", run_dir / "nvtx_ranges.csv"


def get_inference_ranges(nvtx_path: Path):
    """Return list of (start_ns, end_ns) for each inference from NVTX POLICY_INFER_START/END."""
    if not nvtx_path.exists():
        return []
    nv = pd.read_csv(nvtx_path, header=None, names=["ts_ns", "event"])
    nv = nv[nv["event"].isin(["POLICY_INFER_START", "POLICY_INFER_END"])].sort_values("ts_ns")
    starts = nv[nv["event"] == "POLICY_INFER_START"]["ts_ns"].values
    ends = nv[nv["event"] == "POLICY_INFER_END"]["ts_ns"].values
    if len(starts) != len(ends) or len(starts) == 0:
        return []
    return list(zip(starts, ends))


def inference_durations_excl_first(ranges: list) -> tuple:
    """Durations in ms; (all_durations, durations_excluding_first)."""
    if not ranges:
        return [], []
    durs_all = [(e - s) * 1e-6 for s, e in ranges]
    durs_excl = durs_all[1:] if len(durs_all) > 1 else []
    return durs_all, durs_excl


def get_phase_durations_per_inference(nvtx_path: Path):
    """
    nvtx_ranges.csv에서 inference별 ViT/LLM/Action 구간 duration(ms) 반환.
    반환: list of dict {"vit_ms", "llm_ms", "action_ms"} (없는 구간은 nan).
    """
    if not nvtx_path.exists():
        return []
    nv = pd.read_csv(nvtx_path, header=None, names=["ts_ns", "event"])
    pol_starts = nv[nv["event"] == "POLICY_INFER_START"].sort_values("ts_ns")["ts_ns"].values
    pol_ends = nv[nv["event"] == "POLICY_INFER_END"].sort_values("ts_ns")["ts_ns"].values
    if len(pol_starts) != len(pol_ends) or len(pol_starts) == 0:
        return []
    phases = [
        ("vit_ms", "Backbone_ViT_START", "Backbone_ViT_END"),
        ("llm_ms", "Backbone_LLM_START", "Backbone_LLM_END"),
        ("action_ms", "ACTION_HEAD_START", "ACTION_HEAD_END"),
    ]
    result = []
    for i in range(len(pol_starts)):
        s_ns, e_ns = pol_starts[i], pol_ends[i]
        win = nv[(nv["ts_ns"] >= s_ns) & (nv["ts_ns"] <= e_ns)].sort_values("ts_ns")
        row = {"vit_ms": float("nan"), "llm_ms": float("nan"), "action_ms": float("nan")}
        for key, start_ev, end_ev in phases:
            st = win[win["event"] == start_ev]["ts_ns"].values
            en = win[win["event"] == end_ev]["ts_ns"].values
            if len(st) > 0 and len(en) > 0:
                row[key] = (en[0] - st[0]) * 1e-6
        result.append(row)
    return result


def load_nsys_durations_excl_first(run_dir: Path) -> tuple:
    """
    inference_energy_nsys.csv 가 있으면 (infer_1st_ms, durs_excl_list, n_infer) 반환.
    없으면 (nan, [], 0).
    """
    p = run_dir / "inference_energy_nsys.csv"
    if not p.exists():
        return float("nan"), [], 0
    try:
        en = pd.read_csv(p)
        if "inference_id" not in en.columns or "duration_ms" not in en.columns or len(en) == 0:
            return float("nan"), [], 0
        first = en[en["inference_id"] == 1]
        infer_1st_ms = float(first["duration_ms"].iloc[0]) if len(first) else float("nan")
        rest = en[en["inference_id"] >= 2]["duration_ms"].values
        return infer_1st_ms, rest.tolist(), len(en)
    except Exception:
        return float("nan"), [], 0


def stats_during_inferences(df: pd.DataFrame, ranges: list):
    """
    Compute GPU freq and power stats from telemetry samples that fall inside the given
    (start_ns, end_ns) ranges. ranges is typically inference 2..N (first excluded).
    """
    if not ranges:
        return {}
    mask = pd.Series(False, index=df.index)
    for s_ns, e_ns in ranges:
        mask |= (df["ts_ns"] >= s_ns) & (df["ts_ns"] <= e_ns)
    seg = df.loc[mask]
    if len(seg) == 0:
        return {}
    gpu_hz = seg["gpu_freq_hz"]
    gpu_hz = gpu_hz[gpu_hz > 0]
    return {
        "gpu_freq_GHz_mean": (gpu_hz / 1e9).mean() if len(gpu_hz) else float("nan"),
        "gpu_freq_GHz_max": (gpu_hz / 1e9).max() if len(gpu_hz) else float("nan"),
        "vdd_gpu_W_mean": seg["vdd_gpu_W"].mean(),
        "vdd_gpu_W_min": seg["vdd_gpu_W"].min(),
        "vin_W_mean": seg["vin_W"].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Thor run dirs: GPU freq, power, inference duration")
    parser.add_argument("dirs", nargs="+", help="Run directory names (e.g. thor_gr00t_server_*_gpufreq_801MB)")
    parser.add_argument("--base", default=".", help="Base path containing run dirs (default: current dir)")
    parser.add_argument("--plot", action="store_true", default=True, help="Save comparison plots (default: True)")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Do not save plots")
    parser.add_argument("--output-dir", default=None, help="Directory to save PNGs (default: same as --base)")
    parser.add_argument("--tsv", action="store_true", help="Print tab-separated table for Excel paste (each column → each cell)")
    args = parser.parse_args()
    base = Path(args.base)
    out_dir = Path(args.output_dir) if args.output_dir else base

    rows = []
    for name in args.dirs:
        run_dir = base / name
        if not run_dir.is_dir():
            print(f"[WARN] Not a directory: {run_dir}", file=sys.stderr)
            continue
        df, mk_path, nvtx_path = load_telemetry(run_dir)
        if df is None:
            print(f"[WARN] No telemetry: {run_dir}", file=sys.stderr)
            rows.append({"run": name, "error": "no telemetry"})
            continue

        ranges = get_inference_ranges(nvtx_path)
        # inference 시간: nsys 기반 CSV 있으면 사용(첫 번째 제외), 없으면 NVTX CSV 기준
        infer_1st_nsys, durs_excl_nsys, n_infer_nsys = load_nsys_durations_excl_first(run_dir)
        if durs_excl_nsys:
            infer_1st_ms = infer_1st_nsys
            durs_excl = durs_excl_nsys
            n_infer_all = n_infer_nsys
        else:
            durs_all, durs_excl = inference_durations_excl_first(ranges)
            infer_1st_ms = float(durs_all[0]) if durs_all else float("nan")
            n_infer_all = len(durs_all) if durs_all else 0
        # GPU freq / 전력: 항상 NVTX 구간(2~N) 사용 (nsys 정렬 구간은 런마다 스케일 차이로 전력 추세가 뒤틀릴 수 있음)
        ranges_excl_first = ranges[1:] if len(ranges) > 1 else []
        s = stats_during_inferences(df, ranges_excl_first) if ranges_excl_first else {}

        # ViT/LLM/Action 구간 평균(2~N) — nvtx_ranges.csv의 Backbone_ViT, Backbone_LLM, ACTION_HEAD
        phase_list = get_phase_durations_per_inference(nvtx_path)
        phase_excl = phase_list[1:] if len(phase_list) > 1 else []
        if phase_excl:
            vit_vals = [p["vit_ms"] for p in phase_excl]
            llm_vals = [p["llm_ms"] for p in phase_excl]
            action_vals = [p["action_ms"] for p in phase_excl]
            vit_mean = float(np.nanmean(vit_vals)) if any(np.isfinite(vit_vals)) else float("nan")
            llm_mean = float(np.nanmean(llm_vals)) if any(np.isfinite(llm_vals)) else float("nan")
            action_mean = float(np.nanmean(action_vals)) if any(np.isfinite(action_vals)) else float("nan")
        else:
            vit_mean = llm_mean = action_mean = float("nan")

        if durs_excl:
            row = {
                "run": name,
                "gpu_freq_GHz_mean": s.get("gpu_freq_GHz_mean", float("nan")),
                "gpu_freq_GHz_max": s.get("gpu_freq_GHz_max", float("nan")),
                "vdd_gpu_W_mean": s.get("vdd_gpu_W_mean", float("nan")),
                "vdd_gpu_W_min": s.get("vdd_gpu_W_min", float("nan")),
                "vin_W_mean": s.get("vin_W_mean", float("nan")),
                "infer_1st_ms": infer_1st_ms,
                "infer_ms_mean": sum(durs_excl) / len(durs_excl),
                "infer_ms_min": min(durs_excl),
                "infer_ms_max": max(durs_excl),
                "vit_ms_mean": vit_mean,
                "llm_ms_mean": llm_mean,
                "action_ms_mean": action_mean,
                "n_infer": n_infer_all,
                "n_infer_excl": len(durs_excl),
            }
        else:
            row = {
                "run": name,
                "gpu_freq_GHz_mean": s.get("gpu_freq_GHz_mean", float("nan")),
                "gpu_freq_GHz_max": s.get("gpu_freq_GHz_max", float("nan")),
                "vdd_gpu_W_mean": s.get("vdd_gpu_W_mean", float("nan")),
                "vdd_gpu_W_min": s.get("vdd_gpu_W_min", float("nan")),
                "vin_W_mean": s.get("vin_W_mean", float("nan")),
                "infer_1st_ms": infer_1st_ms,
                "infer_ms_mean": float("nan"),
                "infer_ms_min": float("nan"),
                "infer_ms_max": float("nan"),
                "vit_ms_mean": vit_mean,
                "llm_ms_mean": llm_mean,
                "action_ms_mean": action_mean,
                "n_infer": n_infer_all,
                "n_infer_excl": 0,
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    if getattr(args, "tsv", False):
        print("\n--- For Excel (copy below and paste; tab-separated) ---", file=sys.stderr)
        print(out.to_csv(index=False))

    # 그래프: inference / power 비교 (에러 행 제외)
    if "error" in out.columns:
        valid = out.loc[out["error"].isna()].copy()
    else:
        valid = out.copy()
    if args.plot and len(valid) > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = valid["run"].astype(str).tolist()
        # 런 이름이 길면 끝 40자만 (gpufreq 등 식별용)
        labels = [r if len(r) <= 42 else "..." + r[-39:] for r in runs]
        x = np.arange(len(runs))
        w = 0.35

        # 1) Inference 시간
        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.8), 5))
        infer_mean = valid["infer_ms_mean"].values
        infer_1st = valid["infer_1st_ms"].values
        mask_mean = ~np.isnan(infer_mean)
        mask_1st = ~np.isnan(infer_1st)
        if np.any(mask_mean):
            ax.bar(x - w / 2, np.where(mask_mean, infer_mean, 0), width=w, label="infer_ms (mean, excl 1st)", color="C0")
        if np.any(mask_1st):
            ax.bar(x + w / 2, np.where(mask_1st, infer_1st, 0), width=w, label="infer_1st_ms", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Inference time comparison (Thor runs)")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plot_path = out_dir / "compare_inference.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved {plot_path}")

        # 2) 전력
        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.8), 5))
        vdd_gpu = valid["vdd_gpu_W_mean"].values
        vin = valid["vin_W_mean"].values
        mask_gpu = ~np.isnan(vdd_gpu)
        mask_vin = ~np.isnan(vin)
        if np.any(mask_gpu):
            ax.bar(x - w / 2, np.where(mask_gpu, vdd_gpu, 0), width=w, label="VDD_GPU_W (mean)", color="C0")
        if np.any(mask_vin):
            ax.bar(x + w / 2, np.where(mask_vin, vin, 0), width=w, label="VIN_W (mean)", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Power (W)")
        ax.set_title("Power comparison during inference (Thor runs)")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plot_path = out_dir / "compare_power.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
