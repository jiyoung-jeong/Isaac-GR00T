#!/usr/bin/env python3
"""
Plot phase latency (ViT/LLM/Action) as stacked bars (exclude 1st inference).

For each sweep point and backend, compute mean phase duration(ms) over inferences 2..N
from nvtx_ranges.csv and draw stacked bars: ViT + LLM + Action.

Generates:
  - stacked_gpufreq_phases.png
  - stacked_emcfreq_phases.png
  - stacked_emcfreq_phases__gpufreq_{fixed}.png
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

PHASES = [
    ("vit_ms", "Backbone_ViT_START", "Backbone_ViT_END", "ViT", "C2"),
    ("llm_ms", "Backbone_LLM_START", "Backbone_LLM_END", "LLM", "C0"),
    ("action_ms", "ACTION_HEAD_START", "ACTION_HEAD_END", "Action", "C1"),
]


def find_one_or_none(result_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(result_dir.glob(pattern))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        return matches[-1]
    return matches[0]


def find_no_suffix_nvtx(result_dir: Path, backend: str) -> Optional[Path]:
    """
    Find nvtx_ranges.csv for a run dir that ends with _{backend} (no suffix),
    e.g. thor_gr00t_server_..._pytorch/nvtx_ranges.csv
    """
    matches = [
        p
        for p in result_dir.glob(f"thor_gr00t_server_*_{backend}/nvtx_ranges.csv")
        if p.parent.name.endswith("_" + backend)
    ]
    return matches[-1] if matches else None


def _pair_duration_ms(win: pd.DataFrame, start_ev: str, end_ev: str) -> float:
    st = win[win["event"] == start_ev]["ts_ns"].values
    en = win[win["event"] == end_ev]["ts_ns"].values
    if len(st) == 0 or len(en) == 0:
        return float("nan")
    return float((en[0] - st[0]) * 1e-6)


def mean_phases_ms(nvtx_csv: Path) -> Dict[str, float]:
    nv = pd.read_csv(nvtx_csv, header=None, names=["ts_ns", "event"])
    s_all = nv[nv["event"] == "POLICY_INFER_START"].sort_values("ts_ns")["ts_ns"].values
    e_all = nv[nv["event"] == "POLICY_INFER_END"].sort_values("ts_ns")["ts_ns"].values
    if len(s_all) == 0 or len(s_all) != len(e_all):
        raise ValueError(f"bad POLICY_INFER start/end pairs in {nvtx_csv}")
    s_all = s_all[1:]
    e_all = e_all[1:]
    if len(s_all) == 0:
        raise ValueError(f"no inferences after excluding 1st in {nvtx_csv}")

    vals: Dict[str, List[float]] = {k: [] for k, *_ in PHASES}
    for s_ns, e_ns in zip(s_all, e_all):
        win = nv[(nv["ts_ns"] >= s_ns) & (nv["ts_ns"] <= e_ns)].sort_values("ts_ns")
        for key, start_ev, end_ev, *_ in PHASES:
            vals[key].append(_pair_duration_ms(win, start_ev, end_ev))

    out: Dict[str, float] = {}
    for key in vals:
        arr = np.array(vals[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[key] = float(np.mean(arr)) if arr.size else float("nan")
    return out


def plot_stacked(
    out: Path,
    title: str,
    xlabels: List[str],
    phase_by_backend: Dict[str, List[Dict[str, float]]],
):
    # phase_by_backend[backend] is list aligned with xlabels
    n_x = len(xlabels)
    n_b = len(BACKENDS)
    group_w = 0.8
    bar_w = group_w / n_b
    x = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(max(10, n_x * 1.4), 6))
    for bi, backend in enumerate(BACKENDS):
        series = phase_by_backend.get(backend, [])
        if len(series) != n_x:
            series = (series + [{"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan}] * n_x)[:n_x]

        x0 = x - group_w / 2 + bi * bar_w + bar_w / 2
        bottom = np.zeros(n_x, dtype=float)
        for key, *_rest, label, color in [(p[0], p[1], p[2], p[3], p[4]) for p in PHASES]:
            vals = np.array([float(d.get(key, np.nan)) for d in series], dtype=float)
            vals0 = np.nan_to_num(vals, nan=0.0)
            seg_bottom = bottom.copy()
            ax.bar(x0, vals0, width=bar_w * 0.95, bottom=bottom, color=color, alpha=0.9, label=label if bi == 0 else None)
            # Segment value annotations (ViT/LLM/Action ms)
            # - To avoid clutter, skip near-zero segments.
            for i in range(n_x):
                v = float(vals0[i])
                if not np.isfinite(v) or v <= 0.15:
                    continue
                y = float(seg_bottom[i] + v / 2.0)
                # Text bbox for readability on top of colored bars
                ax.text(
                    x0[i],
                    y,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25, edgecolor="none"),
                )
            bottom += vals0

        # Total value annotation on top of each stacked bar
        y_off = max(0.6, float(np.nanmax(bottom)) * 0.015) if np.isfinite(np.nanmax(bottom)) else 0.6
        for i in range(n_x):
            tot = float(bottom[i])
            if not np.isfinite(tot) or tot <= 0:
                continue
            ax.text(
                x0[i],
                tot + y_off,
                f"{tot:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_ylabel("Mean phase time (ms) (sum = ViT + LLM + Action)\n(mean over inferences i>=2)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")

    # Legend for phases + backend color mapping via separate legend
    phase_handles, phase_labels = ax.get_legend_handles_labels()
    ax.legend(phase_handles, phase_labels, loc="upper right", title="Phase (stack)")

    # Add backend labels under each grouped bar cluster
    y_min = 0
    for bi, backend in enumerate(BACKENDS):
        ax.text(0, 0, "")  # keep deterministic layout
    # second legend: backend positions (proxy handles)
    import matplotlib.patches as mpatches

    proxies = [mpatches.Patch(facecolor="white", edgecolor="black", label=b) for b in BACKENDS]
    ax.legend(handles=phase_handles + proxies, loc="upper left", fontsize=9, framealpha=0.9)

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
    xlabels = ["default"] + [t for t, _ in GPUFREQ_TOKENS]
    phase_by_backend: Dict[str, List[Dict[str, float]]] = {}
    for backend in BACKENDS:
        vals = []
        # default (no fixed freq)
        p0 = find_no_suffix_nvtx(result_dir, backend)
        vals.append(mean_phases_ms(p0) if p0 is not None else {"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
        for token, _ in GPUFREQ_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_gpufreq_{token}/nvtx_ranges.csv")
            if p is None:
                vals.append({"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
            else:
                vals.append(mean_phases_ms(p))
        phase_by_backend[backend] = vals
    plot_stacked(
        out_dir / "stacked_gpufreq_phases.png",
        "Stacked phase latency vs GPU freq (exclude 1st inference)",
        xlabels,
        phase_by_backend,
    )

    # 2) EMCfreq sweep (no gpufreq constraint)
    xlabels2 = ["default"] + [t for t, _ in EMC_TOKENS]
    phase_by_backend2: Dict[str, List[Dict[str, float]]] = {}
    for backend in BACKENDS:
        vals = []
        p0 = find_no_suffix_nvtx(result_dir, backend)
        vals.append(mean_phases_ms(p0) if p0 is not None else {"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
        for token, _ in EMC_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}/nvtx_ranges.csv")
            if p is None:
                vals.append({"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
            else:
                vals.append(mean_phases_ms(p))
        phase_by_backend2[backend] = vals
    plot_stacked(
        out_dir / "stacked_emcfreq_phases.png",
        "Stacked phase latency vs EMC freq (exclude 1st inference)",
        xlabels2,
        phase_by_backend2,
    )

    # 3) EMCfreq sweep with fixed GPU freq
    fixed = args.fixed_gpufreq
    phase_by_backend3: Dict[str, List[Dict[str, float]]] = {}
    for backend in BACKENDS:
        vals = []
        p0 = find_no_suffix_nvtx(result_dir, backend)
        vals.append(mean_phases_ms(p0) if p0 is not None else {"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
        for token, _ in EMC_TOKENS:
            p = find_one_or_none(result_dir, f"thor_gr00t_server_*_{backend}_emcfreq_{token}_gpufreq_{fixed}/nvtx_ranges.csv")
            if p is None:
                vals.append({"vit_ms": np.nan, "llm_ms": np.nan, "action_ms": np.nan})
            else:
                vals.append(mean_phases_ms(p))
        phase_by_backend3[backend] = vals
    plot_stacked(
        out_dir / f"stacked_emcfreq_phases__gpufreq_{fixed}.png",
        f"Stacked phase latency vs EMC freq @ GPU freq {fixed} (exclude 1st inference)",
        xlabels2,
        phase_by_backend3,
    )


if __name__ == "__main__":
    main()

