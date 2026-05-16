from __future__ import annotations

from mars_surrogate.calibration import calibrate_margin


def test_calibrated_margin(synthetic_summary_df):
    predictions = synthetic_summary_df[["e2e_median_ms"]].rename(
        columns={"e2e_median_ms": "pred_median_latency_ms"}
    )
    result = calibrate_margin(
        synthetic_summary_df,
        predictions,
        "prediction_to_max_gap",
        q=0.95,
        per_denoising_step=True,
    )
    assert result["margin_ms"] >= 0.0
    assert "per_denoising_step_margin_ms" in result
