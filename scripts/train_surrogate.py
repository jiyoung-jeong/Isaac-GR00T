#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mars_surrogate.calibration import calibrate_margin
from mars_surrogate.feasibility import StrictFeasibilityClassifier
from mars_surrogate.models import SurrogateModel
from mars_surrogate.schema import OPP_COLUMNS, validate_summary_df


def split_train_test(
    df: pd.DataFrame,
    *,
    split: str,
    test_size: float,
    random_state: int,
    workload_group_cols: tuple[str, ...] = ("text_length_target", "denoising_steps", "num_views"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "random_rows":
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        return train.copy(), test.copy()

    if split == "workload_holdout":
        group_cols = [column for column in workload_group_cols if column in df.columns]
        if not group_cols:
            raise ValueError("workload_holdout requires at least one workload group column")
    elif split == "opp_holdout":
        group_cols = OPP_COLUMNS
    else:
        raise ValueError(f"Unknown split mode: {split}")

    groups = df[group_cols].astype(str).agg("|".join, axis=1)
    if groups.nunique() < 2:
        raise ValueError(f"Split mode {split!r} requires at least two unique groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def apply_train_fraction(
    train_df: pd.DataFrame,
    *,
    train_fraction: float | None,
    random_state: int,
) -> pd.DataFrame:
    if train_fraction is None:
        return train_df
    if not 0 < train_fraction <= 1:
        raise ValueError("--train-fraction must be in (0, 1]")
    if train_fraction == 1:
        return train_df
    return train_df.sample(frac=train_fraction, random_state=random_state).copy()


def parse_group_cols(value: str) -> tuple[str, ...]:
    cols = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cols:
        raise ValueError("group columns must contain at least one column")
    return cols


def flattened_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "latency_mae_ms": metrics["latency"]["mae"],
        "latency_rmse_ms": metrics["latency"]["rmse"],
        "latency_mape": metrics["latency"]["mape"],
        "latency_r2": metrics["latency"]["r2"],
        "tail_latency_mae_ms": metrics["tail_latency"]["mae"],
        "tail_latency_rmse_ms": metrics["tail_latency"]["rmse"],
        "tail_latency_mape": metrics["tail_latency"]["mape"],
        "tail_latency_r2": metrics["tail_latency"]["r2"],
        "energy_mae_j": metrics["energy"]["mae"],
        "energy_rmse_j": metrics["energy"]["rmse"],
        "energy_mape": metrics["energy"]["mape"],
        "energy_r2": metrics["energy"]["r2"],
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    df = validate_summary_df(pd.read_csv(args.csv))
    train_df, test_df = split_train_test(
        df,
        split=args.split,
        test_size=args.test_size,
        random_state=args.random_state,
        workload_group_cols=parse_group_cols(args.workload_group_cols),
    )
    original_train = train_df
    train_df = apply_train_fraction(
        train_df,
        train_fraction=args.train_fraction,
        random_state=args.random_state,
    )

    model = SurrogateModel(
        model_type=args.model,
        log_target=args.log_target,
        random_state=args.random_state,
    )
    model.fit(
        train_df,
        val_df=test_df,
        latency_target=args.latency_target,
        tail_target=args.tail_target,
        energy_target=args.energy_target,
    )
    test_metrics = model.evaluate(
        test_df,
        latency_target=args.latency_target,
        tail_target=args.tail_target,
        energy_target=args.energy_target,
    )
    feasibility_model = StrictFeasibilityClassifier(
        model_type=args.feasibility_model,
        random_state=args.random_state,
    )
    feasibility_metrics = feasibility_model.fit(train_df, val_df=test_df)
    predictions = test_df.copy()
    pred = model.predict(test_df)
    predictions["pred_median_latency_ms"] = pred["pred_median_latency_ms"]
    predictions["pred_tail_latency_ms"] = pred["pred_tail_latency_ms"]
    predictions["pred_latency_ms"] = pred["pred_latency_ms"]
    predictions["pred_energy_j"] = pred["pred_energy_j"]
    predictions["pred_strict_feasible_prob"] = feasibility_model.predict_proba(test_df)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    feasibility_model.save(args.out)
    predictions.to_csv(args.out / "predictions_test.csv", index=False)
    pd.DataFrame({"index": train_df.index}).to_csv(args.out / "train_indices.csv", index=False)
    pd.DataFrame({"index": test_df.index}).to_csv(args.out / "test_indices.csv", index=False)

    metrics = {
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_train_before_fraction": int(len(original_train)),
        "n_test": int(len(test_df)),
        "split_mode": args.split,
        "train_fraction": args.train_fraction,
        "model_type": model.resolved_model_type_,
        "feasibility_model_type": feasibility_model.resolved_model_type_,
        "requested_model_type": args.model,
        "requested_feasibility_model_type": args.feasibility_model,
        "latency_target": args.latency_target,
        "tail_target": args.tail_target,
        "energy_target": args.energy_target,
        "log_target": args.log_target,
        "feature_columns": model.feature_columns,
        **flattened_metrics(test_metrics),
    }
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "feasibility_metrics.json", feasibility_metrics)
    margin_payload = {
        "median_to_max_gap": calibrate_margin(
            test_df,
            predictions,
            "median_to_max_gap",
            q=args.margin_quantile,
            per_denoising_step=True,
        ),
        "prediction_to_max_gap": calibrate_margin(
            test_df,
            predictions,
            "prediction_to_max_gap",
            q=args.margin_quantile,
            per_denoising_step=True,
        ),
    }
    write_json(args.out / "calibrated_margin.json", margin_payload)
    if model.model_selection_ is not None:
        write_json(args.out / "model_selection.json", model.model_selection_)
    if feasibility_model.model_selection_ is not None:
        write_json(args.out / "feasibility_model_selection.json", feasibility_model.model_selection_)
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
    parser = argparse.ArgumentParser(description="Train latency and energy surrogate regressors.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "random_forest", "gbr", "auto"], default="auto")
    parser.add_argument(
        "--feasibility-model",
        choices=["logistic_regression", "random_forest_classifier", "gbdt_classifier", "auto"],
        default="auto",
    )
    parser.add_argument("--latency-target", default="e2e_median_ms")
    parser.add_argument("--tail-target", default="e2e_max_ms")
    parser.add_argument("--energy-target", default="vin_energy_j_per_timed_iteration")
    parser.add_argument("--margin-quantile", type=float, default=0.95)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--split", choices=["random_rows", "workload_holdout", "opp_holdout"], default="random_rows")
    parser.add_argument("--workload-group-cols", default="text_length_target,denoising_steps,num_views")
    parser.add_argument("--train-fraction", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(_json_safe(train(args)), indent=2))


if __name__ == "__main__":
    main()
