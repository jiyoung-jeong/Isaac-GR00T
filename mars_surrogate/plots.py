from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_parity(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    out_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df[actual_col], df[pred_col], s=18, alpha=0.75)
    low = min(float(df[actual_col].min()), float(df[pred_col].min()))
    high = max(float(df[actual_col].max()), float(df[pred_col].max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_workload_decisions(decisions: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for steps, group in decisions.groupby("denoising_steps"):
        ordered = group.sort_values("text_length_target")
        ax.plot(
            ordered["text_length_target"],
            ordered["selected_actual_energy_j"],
            marker="o",
            label=f"steps={steps}",
        )
        for _, row in ordered.iterrows():
            label = f"{int(row['selected_cpu_hz']/1e6)}/{int(row['selected_gpu_hz']/1e6)}/{int(row['selected_emc_hz']/1e6)}"
            ax.annotate(label, (row["text_length_target"], row["selected_actual_energy_j"]), fontsize=7)
    ax.set_xlabel("text_length_target")
    ax.set_ylabel("selected measured energy (J/period)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_energy_regret(decisions: pd.DataFrame, out_path: str | Path) -> None:
    if {"text_length_target", "denoising_steps"}.issubset(decisions.columns):
        labels = [
            f"{int(row.text_length_target)}:{int(row.denoising_steps)}"
            for row in decisions.itertuples(index=False)
        ]
        xlabel = "text_length_target:denoising_steps"
    else:
        labels = [str(idx) for idx in range(len(decisions))]
        xlabel = "workload group"
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(labels)), 4.5))
    ax.bar(labels, decisions["energy_regret_pct"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("energy regret (%)")
    ax.tick_params(axis="x", labelrotation=70)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_margin_sweep_success_energy(comparison: pd.DataFrame, out_path: str | Path) -> None:
    margin_rows = comparison[comparison["selector_mode"] == "median_margin"].sort_values("safety_margin_ms")
    if margin_rows.empty:
        return
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()
    x = margin_rows["safety_margin_ms"]
    ax1.plot(x, margin_rows["strict_success_rate"], marker="o", color="tab:blue")
    ax2.plot(x, margin_rows["mean_energy_regret_pct"], marker="s", color="tab:red")
    ax1.set_xlabel("safety margin (ms)")
    ax1.set_ylabel("strict success rate", color="tab:blue")
    ax2.set_ylabel("mean energy regret (%)", color="tab:red")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_oracle_vs_selector_energy(decisions: pd.DataFrame, out_path: str | Path) -> None:
    labels = _workload_labels(decisions)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(labels)), 4.5))
    ax.bar(x - width / 2, decisions["oracle_actual_energy_j"], width, label="oracle")
    ax.bar(x + width / 2, decisions["selected_actual_energy_j"], width, label="selector")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("measured energy (J/period)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feasibility_confusion_matrix(confusion: list[list[int]], out_path: str | Path) -> None:
    matrix = np.asarray(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred miss", "pred feasible"])
    ax.set_yticklabels(["actual miss", "actual feasible"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_title("Strict Feasibility Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _workload_labels(decisions: pd.DataFrame) -> list[str]:
    if {"text_length_target", "denoising_steps"}.issubset(decisions.columns):
        return [f"{int(row.text_length_target)}:{int(row.denoising_steps)}" for row in decisions.itertuples(index=False)]
    return [str(idx) for idx in range(len(decisions))]
