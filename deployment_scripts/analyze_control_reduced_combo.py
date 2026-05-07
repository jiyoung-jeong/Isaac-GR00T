#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd


VIEW_ORDER = ["image_only", "wrist_only", "both_views"]
VIEW_LABELS = {
    "image_only": "Front only (1 view)",
    "wrist_only": "Wrist only (1 view)",
    "both_views": "Front + wrist (2 views)",
}
FREQ_ORDERS = {
    "cpu_label": ["2.430GHz", "2.601GHz"],
    "gpu_label": ["default", "1.107GHz", "1.305GHz", "1.503GHz"],
    "emc_label": ["default", "2.750GHz", "3.200GHz", "4.266GHz"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze control reduced combo sweep results.")
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def sort_text_labels(labels: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, str]:
        try:
            return (int(label.split("_")[0]), label)
        except Exception:
            return (10**9, label)

    return sorted(labels, key=key)


def metric_best_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df.copy()
    if metric == "best_tradeoff":
        work["_score"] = work["e2e_median_ms"] * work["vin_energy_j"]
    elif metric == "best_latency":
        work["_score"] = work["e2e_median_ms"]
    elif metric == "best_energy":
        work["_score"] = work["vin_energy_j"]
    else:
        raise ValueError(metric)
    idx = work.groupby(["text_label", "view_label"])["_score"].idxmin()
    best = work.loc[idx].copy()
    best = best.sort_values(["text_label", "view_label"])
    return best


def add_default_gains(df: pd.DataFrame, best: pd.DataFrame, metric: str) -> pd.DataFrame:
    defaults = df[df["config_name"] == "default"][
        ["text_label", "view_label", "e2e_median_ms", "vin_energy_j"]
    ].rename(
        columns={
            "e2e_median_ms": "default_e2e_median_ms",
            "vin_energy_j": "default_vin_energy_j",
        }
    )
    merged = best.merge(defaults, on=["text_label", "view_label"], how="left")
    merged["latency_gain_pct"] = (
        (merged["default_e2e_median_ms"] - merged["e2e_median_ms"]) / merged["default_e2e_median_ms"] * 100.0
    )
    merged["energy_gain_pct"] = (
        (merged["default_vin_energy_j"] - merged["vin_energy_j"]) / merged["default_vin_energy_j"] * 100.0
    )
    merged["tradeoff_score"] = merged["e2e_median_ms"] * merged["vin_energy_j"]
    merged["default_tradeoff_score"] = merged["default_e2e_median_ms"] * merged["default_vin_energy_j"]
    merged["tradeoff_gain_pct"] = (
        (merged["default_tradeoff_score"] - merged["tradeoff_score"]) / merged["default_tradeoff_score"] * 100.0
    )
    merged["selection_metric"] = metric
    return merged


def pivot_value(df: pd.DataFrame, value_col: str, text_order: list[str], view_order: list[str]) -> pd.DataFrame:
    piv = df.pivot(index="view_label", columns="text_label", values=value_col)
    piv = piv.reindex(index=view_order, columns=text_order)
    return piv


def save_numeric_heatmap(
    pivot: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = ".1f",
    cmap_name: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(pivot.columns), 1.5 + 1.1 * len(pivot.index)))
    vals = pivot.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap_name, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("text length")
    ax.set_ylabel("view config")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    finite = vals[np.isfinite(vals)]
    threshold = np.median(finite) if finite.size else 0.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v <= threshold else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_categorical_heatmap(
    pivot: pd.DataFrame,
    title: str,
    value_label: str,
    out_path: Path,
    categories: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(pivot.columns), 1.5 + 1.1 * len(pivot.index)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    arr = pivot.copy()
    for col in arr.columns:
        arr[col] = arr[col].map(lambda x: cat_to_idx.get(x, np.nan))
    arr = arr.to_numpy(dtype=float)
    cmap = get_cmap("tab10", len(categories))
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5, 1), cmap.N)
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("text length")
    ax.set_ylabel("view config")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(categories)))
    cbar.ax.set_yticklabels(categories)
    cbar.set_label(value_label)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            label = pivot.iloc[i, j]
            if pd.isna(label):
                continue
            ax.text(j, i, str(label), ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metric_lineplot(
    merged_tables: dict[str, pd.DataFrame],
    metric_col: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    colors = {
        "best_latency": "#d62728",
        "best_energy": "#2ca02c",
        "best_tradeoff": "#1f77b4",
    }
    for ax, view_label in zip(axes, VIEW_ORDER):
        for selection_metric, df in merged_tables.items():
            subset = df[df["view_label"] == view_label].copy()
            subset["text_num"] = subset["text_label"].str.extract(r"(\d+)").astype(int)
            subset = subset.sort_values("text_num")
            ax.plot(
                subset["text_num"],
                subset[metric_col],
                marker="o",
                linewidth=2.2,
                label=selection_metric.replace("_", " "),
                color=colors[selection_metric],
            )
        ax.set_title(VIEW_LABELS.get(view_label, view_label))
        ax.set_xlabel("Text length (words)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_freq_selection_lines(
    merged_tables: dict[str, pd.DataFrame],
    freq_col: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    order = FREQ_ORDERS[freq_col]
    mapping = {v: i for i, v in enumerate(order)}
    colors = {
        "best_latency": "#d62728",
        "best_energy": "#2ca02c",
        "best_tradeoff": "#1f77b4",
    }
    for ax, view_label in zip(axes, VIEW_ORDER):
        for selection_metric, df in merged_tables.items():
            subset = df[df["view_label"] == view_label].copy()
            subset["text_num"] = subset["text_label"].str.extract(r"(\d+)").astype(int)
            subset = subset.sort_values("text_num")
            y = subset[freq_col].map(mapping)
            ax.plot(
                subset["text_num"],
                y,
                marker="o",
                linewidth=2.2,
                label=selection_metric.replace("_", " "),
                color=colors[selection_metric],
            )
        ax.set_title(VIEW_LABELS.get(view_label, view_label))
        ax.set_xlabel("Text length (words)")
        ax.grid(True, alpha=0.25)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
    axes[0].set_ylabel(freq_col.replace("_label", "").upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(out_dir: Path, merged_tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Control Scenario Reduced Combo Analysis", ""]
    for metric, df in merged_tables.items():
        lines.append(f"## {metric}")
        avg_latency_gain = df["latency_gain_pct"].mean()
        avg_energy_gain = df["energy_gain_pct"].mean()
        avg_tradeoff_gain = df["tradeoff_gain_pct"].mean()
        lines.append(f"- mean latency gain vs default: {avg_latency_gain:.2f}%")
        lines.append(f"- mean energy gain vs default: {avg_energy_gain:.2f}%")
        lines.append(f"- mean trade-off gain vs default: {avg_tradeoff_gain:.2f}%")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "selection_metric",
        "text_label",
        "view_label",
        "config_name",
        "cpu_label",
        "gpu_label",
        "emc_label",
        "e2e_median_ms",
        "vin_energy_j",
        "latency_gain_pct",
        "energy_gain_pct",
        "tradeoff_gain_pct",
    ]
    view = df[cols].copy()
    rounded = view.copy()
    for col in ["e2e_median_ms", "vin_energy_j", "latency_gain_pct", "energy_gain_pct", "tradeoff_gain_pct"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.2f}")
    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    text_order = sort_text_labels(df["text_label"].dropna().unique().tolist())
    view_order = [v for v in VIEW_ORDER if v in set(df["view_label"].dropna().tolist())]

    merged_tables: dict[str, pd.DataFrame] = {}
    combined_rows = []
    for metric in ["best_latency", "best_energy", "best_tradeoff"]:
        best = metric_best_rows(df, metric)
        merged = add_default_gains(df, best, metric)
        merged_tables[metric] = merged
        merged.to_csv(out_dir / f"{metric}_table.csv", index=False)
        combined_rows.append(merged)

        save_numeric_heatmap(
            pivot_value(merged, "e2e_median_ms", text_order, view_order),
            title=f"{metric}: e2e latency (ms)",
            cbar_label="e2e latency (ms)",
            out_path=out_dir / f"{metric}_latency_heatmap.png",
        )
        save_numeric_heatmap(
            pivot_value(merged, "vin_energy_j", text_order, view_order),
            title=f"{metric}: total VIN energy (J)",
            cbar_label="VIN energy (J)",
            out_path=out_dir / f"{metric}_energy_heatmap.png",
        )
        save_numeric_heatmap(
            pivot_value(merged, "latency_gain_pct", text_order, view_order),
            title=f"{metric}: latency gain vs default (%)",
            cbar_label="latency gain (%)",
            out_path=out_dir / f"{metric}_latency_gain_heatmap.png",
            cmap_name="RdYlGn",
        )
        save_numeric_heatmap(
            pivot_value(merged, "energy_gain_pct", text_order, view_order),
            title=f"{metric}: energy gain vs default (%)",
            cbar_label="energy gain (%)",
            out_path=out_dir / f"{metric}_energy_gain_heatmap.png",
            cmap_name="RdYlGn",
        )

        save_categorical_heatmap(
            pivot_value(merged, "cpu_label", text_order, view_order),
            title=f"{metric}: best CPU freq",
            value_label="CPU freq",
            out_path=out_dir / f"{metric}_cpu_freq_heatmap.png",
            categories=["default", "2.430GHz", "2.601GHz"],
        )
        save_categorical_heatmap(
            pivot_value(merged, "gpu_label", text_order, view_order),
            title=f"{metric}: best GPU freq",
            value_label="GPU freq",
            out_path=out_dir / f"{metric}_gpu_freq_heatmap.png",
            categories=["default", "1.107GHz", "1.305GHz", "1.503GHz"],
        )
        save_categorical_heatmap(
            pivot_value(merged, "emc_label", text_order, view_order),
            title=f"{metric}: best EMC freq",
            value_label="EMC freq",
            out_path=out_dir / f"{metric}_emc_freq_heatmap.png",
            categories=["default", "2.750GHz", "3.200GHz", "4.266GHz"],
        )

    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(out_dir / "best_by_input_all_metrics.csv", index=False)
    write_markdown_table(combined, out_dir / "paper_table.md")
    write_summary(out_dir, merged_tables)

    save_metric_lineplot(
        merged_tables,
        metric_col="e2e_median_ms",
        ylabel="Best e2e latency (ms)",
        out_path=out_dir / "best_latency_energy_tradeoff_line_latency.png",
    )
    save_metric_lineplot(
        merged_tables,
        metric_col="vin_energy_j",
        ylabel="Best total VIN energy (J)",
        out_path=out_dir / "best_latency_energy_tradeoff_line_energy.png",
    )
    save_metric_lineplot(
        merged_tables,
        metric_col="latency_gain_pct",
        ylabel="Latency gain vs default (%)",
        out_path=out_dir / "best_latency_energy_tradeoff_line_latency_gain.png",
    )
    save_metric_lineplot(
        merged_tables,
        metric_col="energy_gain_pct",
        ylabel="Energy gain vs default (%)",
        out_path=out_dir / "best_latency_energy_tradeoff_line_energy_gain.png",
    )
    save_freq_selection_lines(
        merged_tables,
        freq_col="cpu_label",
        out_path=out_dir / "best_freq_path_cpu.png",
    )
    save_freq_selection_lines(
        merged_tables,
        freq_col="gpu_label",
        out_path=out_dir / "best_freq_path_gpu.png",
    )
    save_freq_selection_lines(
        merged_tables,
        freq_col="emc_label",
        out_path=out_dir / "best_freq_path_emc.png",
    )


if __name__ == "__main__":
    main()
