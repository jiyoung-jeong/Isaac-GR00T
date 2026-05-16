#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mars_surrogate.evaluation import compare_selector_variants, evaluate_eco_selector, oracle_feasibility_report
from mars_surrogate.feasibility import StrictFeasibilityClassifier
from mars_surrogate.models import SurrogateModel
from mars_surrogate.plots import (
    plot_energy_regret,
    plot_feasibility_confusion_matrix,
    plot_margin_sweep_success_energy,
    plot_oracle_vs_selector_energy,
    plot_prediction_parity,
    plot_workload_decisions,
)
from mars_surrogate.schema import OPP_COLUMNS, validate_summary_df


def parse_group_cols(value: str) -> tuple[str, ...]:
    cols = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cols:
        raise ValueError("--group-cols must contain at least one column")
    return cols


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = validate_summary_df(pd.read_csv(args.csv))
    model = SurrogateModel.load(args.model_dir)
    try:
        feasibility_model = StrictFeasibilityClassifier.load(args.model_dir)
    except FileNotFoundError:
        feasibility_model = None
    candidates = pd.read_csv(args.candidate_csv)[OPP_COLUMNS]
    group_cols = parse_group_cols(args.group_cols)
    oracle_report, oracle_summary = oracle_feasibility_report(df, group_cols=group_cols)
    oracle_report.to_csv(out_dir / "oracle_feasibility_by_group.csv", index=False)
    write_json(out_dir / "oracle_feasibility_summary.json", oracle_summary)

    decisions, metrics = evaluate_eco_selector(
        df,
        model,
        candidates,
        group_cols=group_cols,
        feasibility=args.feasibility,
        safety_margin_ms=args.safety_margin_ms,
        selector_mode=args.selector_mode,
        feasibility_model=feasibility_model,
        feasible_prob_threshold=args.feasible_prob_threshold,
    )
    decisions.to_csv(out_dir / "eco_decision_eval.csv", index=False)
    write_json(out_dir / "eco_decision_metrics.json", metrics)

    comparison, comparison_payload = compare_selector_variants(
        df,
        model,
        candidates,
        feasibility_model=feasibility_model,
        group_cols=group_cols,
        feasibility=args.feasibility,
    )
    comparison.to_csv(out_dir / "selector_variant_comparison.csv", index=False)
    write_json(out_dir / "selector_variant_comparison.json", comparison_payload)
    selected_vs_oracle = decisions[
        [
            *group_cols,
            "selected_cpu_hz",
            "selected_gpu_hz",
            "selected_emc_hz",
            "oracle_cpu_hz",
            "oracle_gpu_hz",
            "oracle_emc_hz",
            "top1_match",
        ]
    ]
    selected_vs_oracle.to_csv(out_dir / "selected_vs_oracle_opp.csv", index=False)

    pred_df = df.copy()
    pred = model.predict(df)
    pred_df["pred_latency_ms"] = pred["pred_latency_ms"]
    pred_df["pred_median_latency_ms"] = pred["pred_median_latency_ms"]
    pred_df["pred_tail_latency_ms"] = pred["pred_tail_latency_ms"]
    pred_df["pred_energy_j"] = pred["pred_energy_j"]
    plot_prediction_parity(
        pred_df,
        actual_col=model.latency_target,
        pred_col="pred_latency_ms",
        out_path=out_dir / "pred_vs_actual_latency.png",
        title="Latency Prediction Parity",
        xlabel=f"actual {model.latency_target}",
        ylabel="predicted latency ms",
    )
    plot_prediction_parity(
        pred_df,
        actual_col=model.energy_target,
        pred_col="pred_energy_j",
        out_path=out_dir / "pred_vs_actual_energy.png",
        title="Energy Prediction Parity",
        xlabel=f"actual {model.energy_target}",
        ylabel="predicted energy J",
    )
    plot_prediction_parity(
        pred_df,
        actual_col=model.tail_target,
        pred_col="pred_tail_latency_ms",
        out_path=out_dir / "tail_pred_vs_actual.png",
        title="Tail Latency Prediction Parity",
        xlabel=f"actual {model.tail_target}",
        ylabel="predicted tail latency ms",
    )
    if {"text_length_target", "denoising_steps"}.issubset(decisions.columns):
        plot_workload_decisions(decisions, out_dir / "selected_energy_by_workload.png")
    plot_energy_regret(decisions, out_dir / "energy_regret_by_workload.png")
    plot_oracle_vs_selector_energy(decisions, out_dir / "oracle_vs_selector_energy.png")
    if not comparison.empty:
        plot_margin_sweep_success_energy(comparison, out_dir / "margin_sweep_success_energy.png")
    feasibility_metrics_path = args.model_dir / "feasibility_metrics.json"
    if feasibility_metrics_path.exists():
        feasibility_metrics = json.loads(feasibility_metrics_path.read_text(encoding="utf-8"))
        if "confusion_matrix" in feasibility_metrics:
            plot_feasibility_confusion_matrix(
                feasibility_metrics["confusion_matrix"],
                out_dir / "feasibility_confusion_matrix.png",
            )
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Eco OPP selection against measured oracle choices.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feasibility", choices=["strict", "max_latency", "median_latency"], default="strict")
    parser.add_argument("--selector-mode", choices=["median_margin", "tail", "risk"], default="median_margin")
    parser.add_argument("--feasible-prob-threshold", type=float, default=0.5)
    parser.add_argument("--safety-margin-ms", type=float, default=0.0)
    parser.add_argument("--group-cols", default="text_length_target,denoising_steps,num_views,fixed_period_ms")
    args = parser.parse_args()
    print(json.dumps(_json_safe(evaluate(args)), indent=2))


if __name__ == "__main__":
    main()
