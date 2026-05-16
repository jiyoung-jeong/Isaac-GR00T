from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype


WORKLOAD_COLUMNS = [
    "text_length_target",
    "actual_text_words",
    "input_token_count",
    "num_views",
    "denoising_steps",
]

OPP_COLUMNS = ["actual_cpu_hz", "actual_gpu_hz", "actual_emc_hz"]

REQUIRED_COLUMNS = [
    "text_length_target",
    "denoising_steps",
    "actual_cpu_hz",
    "actual_gpu_hz",
    "actual_emc_hz",
    "fixed_period_ms",
    "fixed_period_deadline_miss_pct",
    "e2e_median_ms",
    "e2e_max_ms",
    "vin_energy_j_per_timed_iteration",
]

OPTIONAL_USEFUL_COLUMNS = [
    "actual_text_words",
    "input_token_count",
    "num_views",
    "e2e_mean_ms",
    "fixed_period_deadline_misses",
    "gpu_energy_j_per_timed_iteration",
    "cpu_soc_mss_energy_j_per_timed_iteration",
    "config_name",
    "cpu_label",
    "gpu_label",
    "emc_label",
    "repeat_id",
]

FORBIDDEN_FEATURE_COLUMNS = [
    "config_name",
    "cpu_label",
    "gpu_label",
    "emc_label",
    "repeat_id",
    "fixed_period_deadline_misses",
    "fixed_period_deadline_miss_pct",
    "e2e_median_ms",
    "e2e_mean_ms",
    "e2e_max_ms",
    "vin_energy_j",
    "vin_energy_j_per_timed_iteration",
    "gpu_energy_j",
    "gpu_energy_j_per_timed_iteration",
    "cpu_soc_mss_energy_j",
    "cpu_soc_mss_energy_j_per_timed_iteration",
]

LOCK_COLUMNS = [
    "cpu_lock_ok",
    "gpu_lock_ok",
    "emc_lock_ok",
    "post_cpu_lock_ok",
    "post_gpu_lock_ok",
    "post_emc_lock_ok",
]

NUMERIC_COLUMNS = [
    *REQUIRED_COLUMNS,
    "actual_text_words",
    "input_token_count",
    "num_views",
    "e2e_mean_ms",
    "fixed_period_deadline_misses",
    "gpu_energy_j_per_timed_iteration",
    "cpu_soc_mss_energy_j_per_timed_iteration",
]


def validate_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a measured fixed-period summary DataFrame.

    Returns a copy with rows containing NaN in required columns dropped.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {missing}")

    clean = df.copy()
    for column in [c for c in NUMERIC_COLUMNS if c in clean.columns]:
        if not is_numeric_dtype(clean[column]):
            converted = pd.to_numeric(clean[column], errors="coerce")
            failed = converted.isna() & clean[column].notna()
            if failed.any():
                bad_count = int(failed.sum())
                raise ValueError(f"Column {column!r} must be numeric; {bad_count} values failed conversion")
            clean[column] = converted

    before = len(clean)
    clean = clean.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped = before - len(clean)
    if dropped:
        warnings.warn(f"Dropped {dropped} rows with NaN in required columns", RuntimeWarning)

    for column in LOCK_COLUMNS:
        if column in clean.columns and not _all_lock_ok(clean[column]):
            warnings.warn(f"Frequency lock column {column!r} contains non-ok values", RuntimeWarning)

    return clean


def ensure_no_forbidden_features(feature_columns: list[str]) -> None:
    forbidden = sorted(set(feature_columns).intersection(FORBIDDEN_FEATURE_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden columns used as model features: {forbidden}")


def _all_lock_ok(series: pd.Series) -> bool:
    if series.empty:
        return True
    normalized = series.dropna()
    if normalized.empty:
        return True
    if is_numeric_dtype(normalized):
        return bool((normalized.astype(float) != 0.0).all())
    text = normalized.astype(str).str.strip().str.lower()
    return bool(text.isin({"1", "true", "yes", "y", "ok"}).all())
