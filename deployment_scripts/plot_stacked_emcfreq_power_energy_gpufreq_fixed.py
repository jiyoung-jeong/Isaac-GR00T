#!/usr/bin/env python3
"""
Stacked ViT / LLM / Action breakdown for EMCfreq sweep @ fixed gpufreq (default: 1.305GHz).

Uses:
  - nvtx_ranges.csv: POLICY + phase START/END
  - telemetry_raw.csv: vdd_gpu_W, vin_W vs ts_ns

Metrics (per inference i>=2, then mean across inferences):
  - Energy (J) in each phase: trapezoid sum(P * dt) like thor_server_measure_and_analyze.sh
  - Power stack segments (W): for each phase, (phase energy) / (POLICY duration). These sum to
    ~(full POLICY energy)/T = time-averaged power over the policy, matching the total label.
  - Total power label (W): integrate(P) over full POLICY / policy duration (same trapezoid rule).

Outputs (under --out-dir):
  - stacked_emcfreq_gpu_vin_power__gpufreq_{fixed}.png  (2 panels: GPU + VIN; stacks sum ≈ total)
  - stacked_emcfreq_energy__gpufreq_{fixed}.png  (2 panels: E_gpu, E_vin)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKENDS = ["pytorch", "torchcompile", "tensorRT"]
EMC_TOKENS = [("665MHz", 0.665), ("2.75GHz", 2.75), ("3.2GHz", 3.2), ("4.26GHz", 4.26)]

PHASE_SPECS = [
    ("prebackbone", "POLICY_INFER_START", "BACKBONE_START", "Pre-backbone", "C3"),
    ("vit", "Backbone_ViT_START", "Backbone_ViT_END", "ViT", "C2"),
    ("llm", "Backbone_LLM_START", "Backbone_LLM_END", "LLM", "C0"),
    ("action", "ACTION_HEAD_START", "ACTION_HEAD_END", "Action", "C1"),
]


def find_one_or_none(result_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(result_dir.glob(pattern))
    return matches[-1] if matches else None


def find_no_suffix_nvtx(result_dir: Path, backend: str) -> Optional[Path]:
    matches = [
        p
        for p in result_dir.glob(f"thor_gr00t_server_*_{backend}/nvtx_ranges.csv")
        if p.parent.name.endswith("_" + backend)
    ]
    return matches[-1] if matches else None


def phase_bounds_ns(win: pd.DataFrame, start_ev: str, end_ev: str) -> Optional[Tuple[int, int]]:
    st = win[win["event"] == start_ev]["ts_ns"].values
    en = win[win["event"] == end_ev]["ts_ns"].values
    if len(st) == 0 or len(en) == 0:
        return None
    return int(st[0]), int(en[0])


def integrate_power_j(df: pd.DataFrame, s_ns: int, e_ns: int, col: str) -> float:
    seg = df[(df["ts_ns"] >= s_ns) & (df["ts_ns"] <= e_ns)].sort_values("ts_ns")
    if len(seg) < 2:
        return 0.0
    ts = seg["ts_ns"].values.astype(np.float64)
    dt = np.diff(ts) * 1e-9
    p = seg[col].values[:-1].astype(float)
    return float(np.sum(p * dt))


def mean_policy_power_w(df: pd.DataFrame, pol_s: int, pol_e: int, col: str) -> float:
    """Time-averaged power (W) over [pol_s, pol_e] using trapezoid energy / duration."""
    t_sec = (pol_e - pol_s) * 1e-9
    if t_sec <= 0:
        return float("nan")
    e_j = integrate_power_j(df, pol_s, pol_e, col)
    return float(e_j / t_sec)


def collect_run_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    """Phase keys: *_p_gpu / *_p_vin are E_phase/T_policy (W), stackable to ~tot. Energy: *_e_*."""
    nv_path = run_dir / "nvtx_ranges.csv"
    te_path = run_dir / "telemetry_raw.csv"
    if not nv_path.exists() or not te_path.exists():
        return None
    nv = pd.read_csv(nv_path, header=None, names=["ts_ns", "event"])
    df = pd.read_csv(te_path)
    for c in ("vdd_gpu_W", "vin_W", "ts_ns"):
        if c not in df.columns:
            return None

    s_all = nv[nv["event"] == "POLICY_INFER_START"].sort_values("ts_ns")["ts_ns"].values
    e_all = nv[nv["event"] == "POLICY_INFER_END"].sort_values("ts_ns")["ts_ns"].values
    if len(s_all) < 2 or len(s_all) != len(e_all):
        return None
    s_all = s_all[1:]
    e_all = e_all[1:]

    acc_pg: Dict[str, List[float]] = {k: [] for k, _, _, _, _ in PHASE_SPECS}
    acc_pv: Dict[str, List[float]] = {k: [] for k, _, _, _, _ in PHASE_SPECS}
    acc_eg: Dict[str, List[float]] = {k: [] for k, _, _, _, _ in PHASE_SPECS}
    acc_ev: Dict[str, List[float]] = {k: [] for k, _, _, _, _ in PHASE_SPECS}
    tot_pg: List[float] = []
    tot_pv: List[float] = []

    for pol_s, pol_e in zip(s_all, e_all):
        pol_s = int(pol_s)
        pol_e = int(pol_e)
        t_sec = (pol_e - pol_s) * 1e-9
        win = nv[(nv["ts_ns"] >= pol_s) & (nv["ts_ns"] <= pol_e)].sort_values("ts_ns")
        tot_pg.append(mean_policy_power_w(df, pol_s, pol_e, "vdd_gpu_W"))
        tot_pv.append(mean_policy_power_w(df, pol_s, pol_e, "vin_W"))
        for key, start_ev, end_ev, *_ in PHASE_SPECS:
            b = phase_bounds_ns(win, start_ev, end_ev)
            if b is None:
                acc_pg[key].append(float("nan"))
                acc_pv[key].append(float("nan"))
                acc_eg[key].append(float("nan"))
                acc_ev[key].append(float("nan"))
                continue
            s_ns, e_ns = b
            eg = integrate_power_j(df, s_ns, e_ns, "vdd_gpu_W")
            ev = integrate_power_j(df, s_ns, e_ns, "vin_W")
            acc_eg[key].append(eg)
            acc_ev[key].append(ev)
            if t_sec > 0:
                acc_pg[key].append(float(eg / t_sec))
                acc_pv[key].append(float(ev / t_sec))
            else:
                acc_pg[key].append(float("nan"))
                acc_pv[key].append(float("nan"))

    def _mean(xs: List[float]) -> float:
        a = np.array(xs, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    out: Dict[str, float] = {}
    for key, *_r in PHASE_SPECS:
        out[f"{key}_p_gpu"] = _mean(acc_pg[key])
        out[f"{key}_p_vin"] = _mean(acc_pv[key])
        out[f"{key}_e_gpu"] = _mean(acc_eg[key])
        out[f"{key}_e_vin"] = _mean(acc_ev[key])
    out["tot_p_gpu"] = _mean(tot_pg)
    out["tot_p_vin"] = _mean(tot_pv)
    sum_eg = sum(out[f"{k}_e_gpu"] for k, *_ in PHASE_SPECS if np.isfinite(out[f"{k}_e_gpu"]))
    sum_ev = sum(out[f"{k}_e_vin"] for k, *_ in PHASE_SPECS if np.isfinite(out[f"{k}_e_vin"]))
    out["tot_e_gpu"] = float(sum_eg) if np.isfinite(sum_eg) else float("nan")
    out["tot_e_vin"] = float(sum_ev) if np.isfinite(sum_ev) else float("nan")
    return out  # type: ignore[return-value]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result")
    ap.add_argument("--out-dir", default="/home/Thor/compare_plots_means_sweeps")
    ap.add_argument("--fixed-gpufreq", default="1.305GHz")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    fixed = args.fixed_gpufreq

    xlabels = ["default"] + [t for t, _ in EMC_TOKENS]

    phase_by_backend: Dict[str, List[Dict[str, float]]] = {}
    for backend in BACKENDS:
        rows: List[Dict[str, float]] = []
        p0 = find_no_suffix_nvtx(result_dir, backend)
        if p0:
            m0 = collect_run_metrics(p0.parent)
            rows.append(m0 if m0 is not None else {})
        else:
            rows.append({})
        for token, _ in EMC_TOKENS:
            telem = find_one_or_none(
                result_dir,
                f"thor_gr00t_server_*_{backend}_emcfreq_{token}_gpufreq_{fixed}/telemetry_raw.csv",
            )
            if telem:
                m = collect_run_metrics(telem.parent)
                rows.append(m if m is not None else {})
            else:
                rows.append({})
        phase_by_backend[backend] = rows

    n_x = len(xlabels)
    n_b = len(BACKENDS)
    group_w = 0.8
    bar_w = group_w / n_b
    x = np.arange(n_x)
    phase_order = list(PHASE_SPECS)

    # GPU + VIN mean power: 2 panels (same layout as energy figure)
    fig_p, axes_p = plt.subplots(2, 1, figsize=(max(10, n_x * 1.4), 10), sharex=True)
    for ax_idx, (ax, suffix, tot_k, ylab) in enumerate(
        [
            (axes_p[0], "p_gpu", "tot_p_gpu", "W — GPU (segment = E_phase/T_policy)"),
            (axes_p[1], "p_vin", "tot_p_vin", "W — VIN (segment = E_phase/T_policy)"),
        ]
    ):
        for bi, backend in enumerate(BACKENDS):
            series = phase_by_backend.get(backend, [])
            if len(series) != n_x:
                series = (series + [{}] * n_x)[:n_x]
            x0 = x - group_w / 2 + bi * bar_w + bar_w / 2
            bottom = np.zeros(n_x, dtype=float)
            for key, _s, _e, label, color in phase_order:
                colname = f"{key}_{suffix}"
                vals = np.array([float(d.get(colname, np.nan)) for d in series], dtype=float)
                vals0 = np.nan_to_num(vals, nan=0.0)
                seg_bottom = bottom.copy()
                ax.bar(x0, vals0, width=bar_w * 0.95, bottom=bottom, color=color, alpha=0.9, label=label if bi == 0 else None)
                thr = (
                    0.03 * (float(np.nanmax(bottom + vals0)) + 1e-6)
                    if np.isfinite(np.nanmax(bottom + vals0))
                    else 0.15
                )
                for i in range(n_x):
                    v = float(vals0[i])
                    if not np.isfinite(v) or v <= max(0.15, thr):
                        continue
                    y = float(seg_bottom[i] + v / 2.0)
                    ax.text(
                        x0[i],
                        y,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25, edgecolor="none"),
                    )
                bottom += vals0

            y_off = max(0.02 * (float(np.nanmax(bottom)) + 1e-6), 0.02) if np.isfinite(np.nanmax(bottom)) else 0.05
            for i in range(n_x):
                tot = float(series[i].get(tot_k, np.nan))
                if not np.isfinite(tot):
                    continue
                ax.text(
                    x0[i],
                    float(bottom[i]) + y_off,
                    f"{tot:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75, edgecolor="none"),
                )

        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25, axis="y")
        if ax_idx == 0:
            phase_h, phase_lab = ax.get_legend_handles_labels()
            proxies = [mpatches.Patch(facecolor="white", edgecolor="black", label=b) for b in BACKENDS]
            ax.legend(handles=phase_h + proxies, loc="upper left", fontsize=9, framealpha=0.9)

    axes_p[1].set_xticks(x)
    axes_p[1].set_xticklabels(xlabels)
    fig_p.suptitle(
        f"Policy time-averaged power (W) vs EMC freq @ gpufreq {fixed}\n"
        f"(each segment = ∫P dt in phase / T_policy; stack sum ≈ total; trapezoid on telemetry; i≥2)"
    )
    plt.tight_layout()
    pw_out = out_dir / f"stacked_emcfreq_gpu_vin_power__gpufreq_{fixed}.png"
    pw_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pw_out, dpi=180)
    plt.close(fig_p)
    print(f"[OK] wrote {pw_out}")

    # Energy: 2 panels (E_gpu, E_vin)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_x * 1.4), 10), sharex=True)

    for ax_idx, (ax, suffix, tot_k, ylab) in enumerate(
        [
            (axes[0], "e_gpu", "tot_e_gpu", "Mean energy per inference (J) — GPU rail"),
            (axes[1], "e_vin", "tot_e_vin", "Mean energy per inference (J) — VIN"),
        ]
    ):
        for bi, backend in enumerate(BACKENDS):
            series = phase_by_backend.get(backend, [])
            if len(series) != n_x:
                series = (series + [{}] * n_x)[:n_x]
            x0 = x - group_w / 2 + bi * bar_w + bar_w / 2
            bottom = np.zeros(n_x, dtype=float)
            for key, _s, _e, label, color in phase_order:
                colname = f"{key}_{suffix}"
                vals = np.array([float(d.get(colname, np.nan)) for d in series], dtype=float)
                vals0 = np.nan_to_num(vals, nan=0.0)
                seg_bottom = bottom.copy()
                ax.bar(x0, vals0, width=bar_w * 0.95, bottom=bottom, color=color, alpha=0.9, label=label if bi == 0 else None)
                for i in range(n_x):
                    v = float(vals0[i])
                    if not np.isfinite(v) or v <= 0.05:
                        continue
                    y = float(seg_bottom[i] + v / 2.0)
                    ax.text(
                        x0[i],
                        y,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25, edgecolor="none"),
                    )
                bottom += vals0

            y_off = max(0.02 * (float(np.nanmax(bottom)) + 1e-6), 0.02) if np.isfinite(np.nanmax(bottom)) else 0.05
            for i in range(n_x):
                tot = float(series[i].get(tot_k, np.nan))
                if not np.isfinite(tot):
                    continue
                ax.text(
                    x0[i],
                    float(bottom[i]) + y_off,
                    f"{tot:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75, edgecolor="none"),
                )

        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25, axis="y")
        if ax_idx == 0:
            phase_h, phase_lab = ax.get_legend_handles_labels()
            proxies = [mpatches.Patch(facecolor="white", edgecolor="black", label=b) for b in BACKENDS]
            ax.legend(handles=phase_h + proxies, loc="upper left", fontsize=9, framealpha=0.9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels)
    fig.suptitle(
        f"Per-phase energy (J) vs EMC freq @ gpufreq {fixed}\n"
        f"(phase: ∫P dt in NVTX window; stack sum ≈ total; total label = sum of phase means, i>=2)"
    )
    plt.tight_layout()
    en_out = out_dir / f"stacked_emcfreq_energy__gpufreq_{fixed}.png"
    en_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(en_out, dpi=180)
    plt.close()
    print(f"[OK] wrote {en_out}")


if __name__ == "__main__":
    main()
