#!/usr/bin/env python3
"""Regenerate dense control text/view annotated figures as PNG and PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
TEXT_ORDER = ["8_words", "64_words", "128_words", "256_words"]
TEXT_X = [8, 64, 128, 256]
VIEW_ORDER = ["image_only", "both_views"]
DENOISE_ORDER = [1, 2, 4, 8]
SELECTIONS = ["best_latency", "best_energy", "best_tradeoff"]
TEXT_COLORS = {
    "best_latency": "#1f77b4",
    "best_energy": "#2ca02c",
    "best_tradeoff": "#d62728",
}
VIEW_COLORS = {"image_only": "#1f77b4", "both_views": "#ff7f0e"}
MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}
LINESTYLES = {1: "-", 2: "--", 4: "-.", 8: ":"}


def fmt_freq(row: pd.Series) -> str:
    return f"C {row['cpu_label']}\nG {row['gpu_label']}\nE {row['emc_label']}"


def save_all(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def make_text_annotated(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    # Separate the annotation boxes by denoising layer. The lines are close for
    # d=1 and d=2, so small symmetric offsets make the labels collide.
    offsets = {
        1: [(-10, -22), (-10, -22), (-10, -22), (-16, -22)],
        2: [(8, 14), (8, 14), (8, 14), (12, 14)],
        4: [(-8, 18), (-8, 18), (-8, 18), (-12, 18)],
        8: [(8, -20), (8, -20), (8, -20), (12, -20)],
    }

    for r, selection in enumerate(SELECTIONS):
        sub = df[df["selection_metric"] == selection].copy()
        for c, (metric_mean, metric_std, ylabel, suffix) in enumerate(
            [
                ("e2e_median_ms_mean", "e2e_median_ms_std", "Latency (ms)", "Latency"),
                ("vin_energy_j_mean", "vin_energy_j_std", "VIN Energy (J)", "VIN Energy"),
            ]
        ):
            ax = axes[r, c]
            all_y: list[float] = []
            for denoise in DENOISE_ORDER:
                dsub = sub[sub["denoising_steps"] == denoise].set_index("text_label").reindex(TEXT_ORDER).reset_index()
                y = dsub[metric_mean].to_numpy()
                yerr = dsub[metric_std].fillna(0).to_numpy()
                all_y.extend(list(y[np.isfinite(y)]))
                ax.plot(
                    TEXT_X,
                    y,
                    color=TEXT_COLORS[selection],
                    linestyle=LINESTYLES[denoise],
                    marker=MARKERS[denoise],
                    linewidth=2.0,
                    markersize=7,
                    label=f"d={denoise}",
                )
                ax.fill_between(TEXT_X, y - yerr, y + yerr, color=TEXT_COLORS[selection], alpha=0.10)
                for i, (_, row) in enumerate(dsub.iterrows()):
                    if pd.isna(row[metric_mean]):
                        continue
                    dx, dy = offsets[denoise][i]
                    ax.annotate(
                        fmt_freq(row),
                        xy=(TEXT_X[i], row[metric_mean]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": TEXT_COLORS[selection], "alpha": 0.75},
                    )
            if all_y:
                ymin, ymax = min(all_y), max(all_y)
                pad_low = (ymax - ymin) * 0.35 if ymax > ymin else ymax * 0.12
                pad_high = (ymax - ymin) * 0.20 if ymax > ymin else ymax * 0.12
                ax.set_ylim(ymin - pad_low, ymax + pad_high)
            ax.set_title(f"{selection}: {suffix}", fontsize=11)
            ax.set_xlabel("Text length (words)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 270)
            ax.set_xticks(TEXT_X)
            ax.grid(True, alpha=0.22)
            ax.legend(loc="upper left", ncol=4, frameon=True, fontsize=8)
    save_all(fig, out_path)
    plt.close(fig)


def make_view_annotated(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    x = np.arange(len(DENOISE_ORDER))
    width = 0.34
    handles = [
        Line2D([0], [0], color=VIEW_COLORS[v], lw=8, label=("1 view" if v == "image_only" else "2 views"))
        for v in VIEW_ORDER
    ]

    for r, selection in enumerate(SELECTIONS):
        sub = df[df["selection_metric"] == selection].copy()
        for c, (metric_mean, metric_std, ylabel, suffix) in enumerate(
            [
                ("e2e_median_ms_mean", "e2e_median_ms_std", "Latency (ms)", "Latency"),
                ("vin_energy_j_mean", "vin_energy_j_std", "VIN Energy (J)", "VIN Energy"),
            ]
        ):
            ax = axes[r, c]
            for i, view in enumerate(VIEW_ORDER):
                vsub = sub[sub["view_label"] == view].set_index("denoising_steps").reindex(DENOISE_ORDER).reset_index()
                xpos = x + (i - 0.5) * width
                vals = vsub[metric_mean].to_numpy()
                errs = vsub[metric_std].fillna(0).to_numpy()
                ax.bar(
                    xpos,
                    vals,
                    width=width,
                    color=VIEW_COLORS[view],
                    yerr=errs,
                    capsize=3,
                    label=("1 view" if view == "image_only" else "2 views"),
                )
                for j, (_, row) in enumerate(vsub.iterrows()):
                    if pd.isna(row[metric_mean]):
                        continue
                    offset = 1.8 if c == 0 else 1.0
                    ax.text(
                        xpos[j],
                        row[metric_mean] + errs[j] + offset,
                        fmt_freq(row),
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.75", "alpha": 0.97},
                    )
            ax.set_title(f"{selection}: {suffix}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([str(d) for d in DENOISE_ORDER])
            ax.set_xlabel("Denoising steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.22)
            if r == 0:
                ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    save_all(fig, out_path)
    plt.close(fig)


def main() -> None:
    text_df = pd.read_csv(BASE / "text_only_best_table.csv")
    view_df = pd.read_csv(BASE / "viewcount_only_best_table.csv")
    make_text_annotated(text_df, BASE / "text_only_main_figure_annotated_dense.png")
    make_view_annotated(view_df, BASE / "viewcount_main_figure_annotated_dense.png")


if __name__ == "__main__":
    main()
