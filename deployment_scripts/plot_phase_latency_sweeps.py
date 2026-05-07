#!/usr/bin/env python3
"""
Plot phase latency (ViT/LLM/Action) sweeps from nvtx_ranges.csv, excluding 1st inference.

Phases:
  - Backbone_ViT_START/END   -> vit_ms
  - Backbone_LLM_START/END   -> llm_ms
  - ACTION_HEAD_START/END    -> action_ms

Per inference i:
  phase_ms = (end_ns - start_ns) * 1e-6
We summarize per run as mean over inferences 2..N.

Generates 3 sweep plots (3 panels each: ViT/LLM/Action):
  - sweep_gpufreq_phases.png
  - sweep_emcfreq_phases.png
  - sweep_emcfreq_phases__gpufreq_{fixed}.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]
GPUFREQ_TOKENS: List[Tuple[str, float]] = [("801MHz", 0.801), ("1.305GHz", 1.305), ("1.575GHz", 1.575)]
EMC_TOKENS: List[Tuple[str, float]] = [("665MHz", 0.665), ("2.75GHz", 2.75), ("3.2GHz", 3.2), ("4.26GHz", 4.26)]


def find_one_or_none(result_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(result_dir.glob(pattern))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        return matches[-1]  # latest lexicographically
    return matches[0]


def _pair_duration_ms(win: pd.DataFrame, start_ev: str, end_ev: str) -> float:
    st = win[win["event"] == start_ev]["ts_ns"].values
    en = win[win["event"] == end_ev]["ts_ns"].values
    if len(st) == 0 or len(en) == 0:
        return float("nan")
    return float((en[0] - st[0]) * 1e-6)


def mean_phase_ms(nvtx_csv: Path) -> Dict[str, float]:
    nv = pd.read_csv(nvtx_csv, header=None, names=["ts_ns", "event"])
    # infer windows
    s_all = nv[nv["event"] == "POLICY_INFER_START"].sort_values("ts_ns")["ts_ns"].values
    e_all = nv[nv["event"] == "POLICY_INFER_END"].sort_values("ts_ns")["ts_ns"].values
    if len(s_all) == 0 or len(s_all) != len(e_all):
        raise ValueError(f"bad POLICY_INFER start/end pairs in {nvtx_csv}")

    # exclude 1st inference
    s_all = s_all[1:]
    e_all = e_all[1:]
    if len(s_all) == 0:
        raise ValueError(f"no inferences after excluding 1st in {nvtx_csv}")

    vit = []
    llm = []
    act = []
    for s_ns, e_ns in zip(s_all, e_all):
        win = nv[(nv["ts_ns"] >= s_ns) & (nv["ts_ns"] <= e_ns)].sort_values("ts_ns")
        vit.append(_pair_duration_ms(win, "Backbone_ViT_START", "Backbone_ViT_END"))
        llm.append(_pair_duration_ms(win, "Backbone_LLM_START", "Backbone_LLM_END"))
        act.append(_pair_duration_ms(win, "ACTION_HEAD_START", "ACTION_HEAD_END"))

    def _mean(xs: List[float]) -> float:
        arr = np.array(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    return {"vit_ms": _mean(vit), "llm_ms": _mean(llm), "action_ms": _mean(act)}


def plot(out: Path, title: str, xlabel: str, x: np.ndarray, xlabels: List[str], series: List[Tuple[str, List[float], List[float], List[float]]]):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for label, vit, llm, act in series:
        axes[0].plot(x, vit, marker="o", linewidth=1.8, label=label)
        axes[1].plot(x, llm, marker="o", linewidth=1.8, label=label)
        axes[2].plot(x, act, marker="o", linewidth=1.8, label=label)

    axes[0].set_ylabel("ViT mean (ms)\n(mean over i>=2)")
    axes[1].set_ylabel("LLM mean (ms)\n(mean over i>=2)")
    axes[2].set_ylabel("Action mean (ms)\n(mean over i>=2)")
    axes[2].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xlabels)
    fig.suptitle(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result")
    ap.add_argument("--out-dir", default="/home/Thor/compare_plots_means_sweeps")
    ap.add_argument("--fixed-gpufreq", default="1.305GHz")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)

    # 1) GPUfreq sweep
    x = np.array([v for _, v in GPUFREQ_TOKENS], dtype=float)
    xlabels = [t for t, _ in GPUFREQ_TOKENS]
    series = []
    for backend in BACKENDS:
        vit, llm, act = [], [], []
        for token, _ in GPUFREQ_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_gpufreq_{token}/nvtx_ranges.csv")
            if p is None:
                vit.append(float("nan")); llm.append(float("nan")); act.append(float("nan"))
            else:
                m = mean_phase_ms(p)
                vit.append(m["vit_ms"]); llm.append(m["llm_ms"]); act.append(m["action_ms"])
        series.append((backend, vit, llm, act))
    plot(
        out_dir / "sweep_gpufreq_phases.png",
        "Phase latency vs GPU freq (exclude 1st inference)",
        "GPU frequency (GHz)",
        x,
        xlabels,
        series,
    )

    # 2) EMCfreq sweep (no gpufreq constraint)
    x2 = np.array([v for _, v in EMC_TOKENS], dtype=float)
    x2labels = [t for t, _ in EMC_TOKENS]
    series2 = []
    for backend in BACKENDS:
        vit, llm, act = [], [], []
        for token, _ in EMC_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}/nvtx_ranges.csv")
            if p is None:
                vit.append(float("nan")); llm.append(float("nan")); act.append(float("nan"))
            else:
                m = mean_phase_ms(p)
                vit.append(m["vit_ms"]); llm.append(m["llm_ms"]); act.append(m["action_ms"])
        series2.append((backend, vit, llm, act))
    plot(
        out_dir / "sweep_emcfreq_phases.png",
        "Phase latency vs EMC freq (exclude 1st inference)",
        "EMC frequency (GHz)",
        x2,
        x2labels,
        series2,
    )

    # 3) EMCfreq sweep with fixed GPU freq
    fixed = args.fixed_gpufreq
    series3 = []
    for backend in BACKENDS:
        vit, llm, act = [], [], []
        for token, _ in EMC_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}_gpufreq_{fixed}/nvtx_ranges.csv")
            if p is None:
                vit.append(float("nan")); llm.append(float("nan")); act.append(float("nan"))
            else:
                m = mean_phase_ms(p)
                vit.append(m["vit_ms"]); llm.append(m["llm_ms"]); act.append(m["action_ms"])
        series3.append((backend, vit, llm, act))
    plot(
        out_dir / f"sweep_emcfreq_phases__gpufreq_{fixed}.png",
        f"Phase latency vs EMC freq @ GPU freq {fixed} (exclude 1st inference)",
        "EMC frequency (GHz)",
        x2,
        x2labels,
        series3,
    )


if __name__ == "__main__":
    main()

