from __future__ import annotations

from mars_surrogate.feasibility import StrictFeasibilityClassifier, strict_feasible_labels


def test_strict_feasibility_label_construction(synthetic_summary_df):
    labels = strict_feasible_labels(synthetic_summary_df)
    expected = (synthetic_summary_df["fixed_period_deadline_miss_pct"] == 0.0).astype(int).to_numpy()
    assert labels.tolist() == expected.tolist()


def test_feasibility_classifier_predicts_probabilities(synthetic_summary_df):
    synthetic_summary_df = synthetic_summary_df.copy()
    synthetic_summary_df.loc[synthetic_summary_df.index[:4], "fixed_period_deadline_miss_pct"] = 100.0
    model = StrictFeasibilityClassifier(model_type="logistic_regression")
    model.fit(synthetic_summary_df)
    prob = model.predict_proba(synthetic_summary_df.head(4))
    assert len(prob) == 4
    assert ((prob >= 0.0) & (prob <= 1.0)).all()
    assert "fixed_period_ms" in model.feature_columns
    assert "fixed_period_deadline_miss_pct" not in model.feature_columns
