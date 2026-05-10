#!/usr/bin/env python3
"""Regenerate the no-all-default Pareto latency/energy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DISPLAY_HZ = {
    "cpu": 2_601_000_000,
    "gpu": 1_575_000_000,
    "emc": 3_200_000_000,
}


def fmt_freq(hz: float) -> str:
    if hz < 0:
        return "default"
    if hz >= 1_000_000_000:
        return f"{hz / 1_000_000_000:.3f}".rstrip("0").rstrip(".") + "GHz"
    return f"{hz / 1_000_000:.0f}MHz"


def display_freq(row: pd.Series, key: str) -> str:
    requested = float(row[f"requested_{key}_hz"])
    if requested >= 0:
        return fmt_freq(requested)
    actual = float(row.get(f"actual_{key}_hz", -1))
    if actual > 0:
        return fmt_freq(actual)
    return fmt_freq(DEFAULT_DISPLAY_HZ[key])


def combo_freq_label(row: pd.Series, title: str) -> str:
    return "\n".join(
        [
            title,
            f"CPU {display_freq(row, 'cpu')}",
            f"GPU {display_freq(row, 'gpu')}",
            f"EMC {display_freq(row, 'emc')}",
        ]
    )


def label_box(ax, x: float, y: float, text: str, dx: float, dy: float, ha: str = "left") -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.55", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
    )


def main() -> None:
    summary = pd.read_csv(HERE / "all_nondefault_ranked.csv")
    pareto = pd.read_csv(HERE / "pareto_front.csv").sort_values("get_action_latency_ms")
    summary["edp"] = summary["get_action_latency_ms"] * summary["vin_energy_j"]

    best_latency = summary.loc[summary["get_action_latency_ms"].idxmin()]
    best_energy = summary.loc[summary["vin_energy_j"].idxmin()]
    best_tradeoff = summary.loc[summary["edp"].idxmin()]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(summary["get_action_latency_ms"], summary["vin_energy_j"], s=25, c="0.75", alpha=0.8)
    ax.plot(pareto["get_action_latency_ms"], pareto["vin_energy_j"], "-o", color="crimson", lw=2, ms=5)
    ax.scatter([best_latency["get_action_latency_ms"]], [best_latency["vin_energy_j"]], c="darkgreen", s=45, zorder=5)
    ax.scatter([best_energy["get_action_latency_ms"]], [best_energy["vin_energy_j"]], c="purple", s=45, zorder=5)
    ax.scatter([best_tradeoff["get_action_latency_ms"]], [best_tradeoff["vin_energy_j"]], c="darkorange", s=45, zorder=5)

    label_box(
        ax,
        best_latency["get_action_latency_ms"],
        best_latency["vin_energy_j"],
        combo_freq_label(best_latency, "Best latency"),
        dx=20,
        dy=104,
    )
    label_box(
        ax,
        best_tradeoff["get_action_latency_ms"],
        best_tradeoff["vin_energy_j"],
        combo_freq_label(best_tradeoff, "Best trade-off"),
        dx=96,
        dy=28,
    )
    label_box(
        ax,
        best_energy["get_action_latency_ms"],
        best_energy["vin_energy_j"],
        combo_freq_label(best_energy, "Best energy"),
        dx=138,
        dy=6,
    )

    ax.set_xlabel("VLA/get_action Latency (ms)")
    ax.set_ylabel("Total VIN Energy (J)")
    ax.set_title("Pareto Front: Latency vs Total VIN Energy (All-default removed)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "pareto_latency_energy.png", dpi=180)
    fig.savefig(HERE / "pareto_latency_energy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
