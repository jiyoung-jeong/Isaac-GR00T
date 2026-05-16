from __future__ import annotations

from mars_surrogate.models import SurrogateModel


def test_surrogate_model_predicts_nonnegative(synthetic_summary_df):
    train = synthetic_summary_df.iloc[:-8]
    test = synthetic_summary_df.iloc[-8:]
    model = SurrogateModel(model_type="ridge", log_target=True)
    model.fit(train)
    pred = model.predict(test)
    assert (pred["pred_latency_ms"] >= 0).all()
    assert (pred["pred_median_latency_ms"] >= 0).all()
    assert (pred["pred_tail_latency_ms"] >= 0).all()
    assert (pred["pred_energy_j"] >= 0).all()
    assert "fixed_period_ms" not in model.feature_columns


def test_tail_model_prediction(synthetic_summary_df):
    model = SurrogateModel(model_type="ridge", log_target=True)
    model.fit(synthetic_summary_df, tail_target="e2e_max_ms")
    pred = model.predict(synthetic_summary_df.head(5))
    assert "pred_tail_latency_ms" in pred.columns
    assert pred["pred_tail_latency_ms"].notna().all()
