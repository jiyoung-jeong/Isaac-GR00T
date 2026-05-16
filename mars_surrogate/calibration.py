from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calibrate_margin(
    validation_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    mode: str,
    *,
    q: float = 0.95,
    per_denoising_step: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    aligned = validation_df.reset_index(drop=True).copy()
    pred = predictions_df.reset_index(drop=True).copy()
    if len(aligned) != len(pred):
        raise ValueError("validation_df and predictions_df must have the same number of rows")

    if mode == "median_to_max_gap":
        gap = aligned["e2e_max_ms"].astype(float) - aligned["e2e_median_ms"].astype(float)
    elif mode == "prediction_to_max_gap":
        if "pred_median_latency_ms" not in pred.columns:
            raise ValueError("prediction_to_max_gap requires pred_median_latency_ms")
        gap = aligned["e2e_max_ms"].astype(float) - pred["pred_median_latency_ms"].astype(float)
    else:
        raise ValueError("mode must be one of {'median_to_max_gap', 'prediction_to_max_gap'}")

    gap = gap.clip(lower=0.0)
    payload: dict[str, Any] = {
        "mode": mode,
        "q": q,
        "margin_ms": float(np.quantile(gap, q)),
    }
    if per_denoising_step:
        margins = {}
        for step, idx in aligned.groupby("denoising_steps").groups.items():
            margins[str(step)] = float(np.quantile(gap.loc[idx], q))
        payload["per_denoising_step_margin_ms"] = margins
    return payload
