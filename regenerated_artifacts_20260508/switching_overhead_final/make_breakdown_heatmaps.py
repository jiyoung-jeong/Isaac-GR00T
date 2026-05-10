#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = SCRIPT_DIR / "source" / "transition_breakdown.csv"
DEFAULT_OLD_SOURCE_DIR = SCRIPT_DIR / "source"
DEFAULT_OUT_DIR = SCRIPT_DIR
AXIS_ORDER = ("cpu", "gpu", "emc")
AXIS_TITLES = {"cpu": "CPU breakdown", "gpu": "GPU breakdown", "emc": "EMC breakdown"}


def build_matrix(rows: pd.DataFrame, labels: list[str], value_col: str) -> np.ndarray:
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for row in rows.itertuples(index=False):
        if int(row.success) != 1:
            continue
        from_idx = label_to_idx[getattr(row, "from_label")]
        to_idx = label_to_idx[getattr(row, "to_label")]
        if from_idx == to_idx:
            continue
        matrix[from_idx, to_idx] = float(getattr(row, value_col))
    return matrix


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("from_label", "to_label"):
        out[col] = out[col].replace({"665MHz": "666MHz"})
    return out


def old_axis_csv(old_source_dir: Path, axis_name: str) -> Path:
    return old_source_dir / f"old_{axis_name}_transition_results.csv"


def color_limits(new_rows: pd.DataFrame, old_source_dir: Path, axis_name: str, value_col: str) -> tuple[float, float]:
    values = new_rows.loc[new_rows["success"] == 1, value_col].astype(float).tolist()
    old_csv = old_axis_csv(old_source_dir, axis_name)
    if old_csv.exists():
        old_rows = normalize_labels(pd.read_csv(old_csv))
        old_rows = old_rows[
            (old_rows["success"] == 1)
            & (old_rows["from_label"] != old_rows["to_label"])
            & (old_rows["from_label"] != "default")
            & (old_rows["to_label"] != "default")
        ]
        values.extend(old_rows["stable_ms"].astype(float).tolist())
    return float(np.nanmin(values)), float(np.nanmax(values))


def draw_heatmap(
    ax,
    rows: pd.DataFrame,
    axis_name: str,
    value_col: str,
    old_source_dir: Path,
) -> object:
    labels = list(dict.fromkeys(rows["from_label"].tolist() + rows["to_label"].tolist()))
    matrix = build_matrix(rows, labels, value_col)
    cmap = plt.get_cmap("viridis").copy()
    vmin, vmax = color_limits(rows, old_source_dir, axis_name, value_col)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            facecolor = "#f2f2f2" if math.isnan(value) else cmap(norm(value))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=facecolor, edgecolor="none"))

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(AXIS_TITLES[axis_name], fontsize=13, pad=8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("to", fontsize=10)
    ax.set_ylabel("from", fontsize=10)

    mean_val = float(np.nanmean(matrix))
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if math.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
            )
    return ScalarMappable(norm=norm, cmap=cmap)


def make_combined_heatmap(source_csv: Path, old_source_dir: Path, out_dir: Path, value_col: str) -> None:
    df = normalize_labels(pd.read_csv(source_csv))
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.1), dpi=180)

    for ax, axis_name in zip(axes, AXIS_ORDER):
        rows = df[df["axis"] == axis_name].copy()
        im = draw_heatmap(ax, rows, axis_name, value_col, old_source_dir)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.045)
        cbar.set_label(value_col.replace("usable_", "").replace("_", " "), fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(w_pad=2.6)
    fig.savefig(out_dir / "switching_overhead_breakdown_heatmaps.png", bbox_inches="tight")
    fig.savefig(out_dir / "switching_overhead_breakdown_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU/GPU/EMC switching-overhead breakdown heatmaps.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--old-source-dir", type=Path, default=DEFAULT_OLD_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--value-col", default="usable_stable_ms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_combined_heatmap(args.source_csv, args.old_source_dir, args.out_dir, args.value_col)
    print(args.out_dir)


if __name__ == "__main__":
    main()
