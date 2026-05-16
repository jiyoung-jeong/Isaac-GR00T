from __future__ import annotations

import numpy as np

from mars_surrogate.schema import validate_summary_df


def test_schema_validation(synthetic_summary_df):
    df = synthetic_summary_df.copy()
    df.loc[df.index[0], "e2e_median_ms"] = np.nan
    clean = validate_summary_df(df)
    assert len(clean) == len(df) - 1


def test_schema_validation_rejects_missing_required(synthetic_summary_df):
    df = synthetic_summary_df.drop(columns=["e2e_median_ms"])
    try:
        validate_summary_df(df)
    except ValueError as exc:
        assert "e2e_median_ms" in str(exc)
    else:
        raise AssertionError("validate_summary_df should reject missing required columns")
