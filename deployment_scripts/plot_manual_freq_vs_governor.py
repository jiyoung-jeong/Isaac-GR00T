#!/usr/bin/env python3
"""
Plot manual-freq vs governor comparisons from precomputed CSV summaries.

Inputs:
  - manual_freq_phase_metrics.csv
  - manual_freq_vs_governor_summary.csv

Outputs:
  - manual_vs_governor_phase_compare_{backend}.png (for tensorRT/torchcompile/pytorch)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKENDS = ["tensorRT", "torchcompile", "pytorch"]
PHASES = ["vit", "llm", "action", "total"]
METRICS = [
    ("latency_ms", "Latency (ms)"),
    ("power_gpu_w", "GPU Power (W)"),
    ("power_vin_w", "VIN Power (W)"),
    ("energy_gpu_j", "GPU Energy (J)"),
    ("energy_vin_j", "VIN Energy (J)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        default="/home/Thor/compare_plots_means_sweeps/manual_freq_phase_metrics.csv",
    )
    ap.add_argument(
        "--summary-csv",
        default="/home/Thor/compare_plots_means_sweeps/manual_freq_vs_governor_summary.csv",
    )
    ap.add_argument("--out-dir", default="/home/Thor/compare_plots_means_sweeps")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    sm = pd.read_csv(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for backend in BACKENDS:
        g = df[(df["backend"] == backend) & (df["set_type"] == "governor")]
        if g.empty:
            print(f"[WARN] governor row missing for {backend}")
            continue
        g_row = g.iloc[0]

        s = sm[(sm["backend"] == backend) & (sm["target_metric"] == "wins_across_20_metrics")]
        if s.empty:
            print(f"[WARN] summary wins row missing for {backend}")
            continue
        best_run = str(s.iloc[0]["best_manual_run"])

        m = df[(df["backend"] == backend) & (df["set_type"] == "manual") & (df["run"] == best_run)]
        if m.empty:
            print(f"[WARN] best manual run row missing for {backend}: {best_run}")
            continue
        m_row = m.iloc[0]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        x = np.arange(len(PHASES))
        w = 0.36
        manual_label = f"manual best ({m_row['emcfreq']}, {m_row['gpufreq']})"

        for i, (suffix, ylabel) in enumerate(METRICS):
            ax = axes[i]
            g_vals = [float(g_row[f"{p}_{suffix}"]) for p in PHASES]
            m_vals = [float(m_row[f"{p}_{suffix}"]) for p in PHASES]
            ax.bar(x - w / 2, g_vals, width=w, label="governor", alpha=0.85)
            ax.bar(
                x + w / 2,
                m_vals,
                width=w,
                label=manual_label,
                alpha=0.85,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(["ViT", "LLM", "Action", "Total"])
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, axis="y", alpha=0.25)

        # Last panel: show percent deltas for TOTAL only
        ax = axes[5]
        labels = ["Latency", "GPU P", "VIN P", "GPU E", "VIN E"]
        deltas = []
        for suffix, _ in METRICS:
            gv = float(g_row[f"total_{suffix}"])
            mv = float(m_row[f"total_{suffix}"])
            d = (mv - gv) / gv * 100.0 if np.isfinite(gv) and gv != 0 else np.nan
            deltas.append(d)
        colors = ["C2" if (np.isfinite(v) and v < 0) else "C3" for v in deltas]
        ax.bar(np.arange(len(labels)), deltas, color=colors, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Delta vs governor (%)")
        ax.set_title("Total metric deltas\n(manual-best vs governor)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticklabels(labels, rotation=15)
        finite = np.array([v for v in deltas if np.isfinite(v)], dtype=float)
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            pad = max(2.0, 0.12 * (hi - lo + 1e-6))
            ax.set_ylim(lo - pad, hi + pad)

        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, framealpha=0.95)
        fig.suptitle(f"{backend}: governor vs manual-best", y=0.995)
        # Keep long run names out of the crowded top area.
        fig.text(0.01, 0.01, f"governor run: {g_row['run']}", ha="left", va="bottom", fontsize=9)
        fig.text(0.01, 0.035, f"manual run: {best_run}", ha="left", va="bottom", fontsize=9)
        fig.text(0.99, 0.01, f"selected manual: {manual_label}", ha="right", va="bottom", fontsize=9)
        plt.tight_layout(rect=[0, 0.07, 1, 0.84])
        out = out_dir / f"manual_vs_governor_phase_compare_{backend}.png"
        plt.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

