from __future__ import annotations

from mars_surrogate.features import FeatureBuilder
from mars_surrogate.schema import FORBIDDEN_FEATURE_COLUMNS


def test_feature_builder_fit_transform(synthetic_summary_df):
    builder = FeatureBuilder()
    features = builder.fit_transform(synthetic_summary_df)
    transformed = builder.transform(synthetic_summary_df.head(3))
    assert list(transformed.columns) == builder.get_feature_names()
    assert features.shape[1] == len(builder.get_feature_names())
    assert "fixed_period_ms" not in builder.get_feature_names()


def test_no_forbidden_columns_used_as_features(synthetic_summary_df):
    builder = FeatureBuilder()
    builder.fit_transform(synthetic_summary_df)
    forbidden = set(FORBIDDEN_FEATURE_COLUMNS)
    assert not forbidden.intersection(builder.get_feature_names())
