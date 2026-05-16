from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mars_surrogate.schema import ensure_no_forbidden_features


@dataclass
class FeatureBuilder:
    include_deadline_feature: bool = False
    feature_names_: list[str] = field(default_factory=list)
    optional_features_: list[str] = field(default_factory=list)
    has_num_views_: bool = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.optional_features_ = [
            column for column in ["actual_text_words", "input_token_count"] if column in df.columns
        ]
        self.has_num_views_ = "num_views" in df.columns
        features = self._build(df)
        self.feature_names_ = list(features.columns)
        ensure_no_forbidden_features(self.feature_names_)
        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        features = self._build(df)
        missing = [column for column in self.feature_names_ if column not in features.columns]
        if missing:
            raise ValueError(f"Input DataFrame cannot produce fitted feature columns: {missing}")
        return features[self.feature_names_]

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names_)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["text_length_target", "denoising_steps", "actual_cpu_hz", "actual_gpu_hz", "actual_emc_hz"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Cannot build features; missing columns: {missing}")

        out = pd.DataFrame(index=df.index)
        text_length = _numeric(df, "text_length_target")
        out["log1p_text_length_target"] = np.log1p(text_length.clip(lower=0.0))

        if "actual_text_words" in self.optional_features_:
            words = _numeric_or_default(df, "actual_text_words", text_length)
            out["log1p_actual_text_words"] = np.log1p(words.clip(lower=0.0))
        if "input_token_count" in self.optional_features_:
            tokens = _numeric_or_default(df, "input_token_count", text_length)
            out["log1p_input_token_count"] = np.log1p(tokens.clip(lower=0.0))

        denoising_steps = _numeric(df, "denoising_steps")
        out["denoising_steps"] = denoising_steps
        if self.has_num_views_:
            num_views = _numeric_or_default(df, "num_views", 2.0)
        else:
            num_views = pd.Series(2.0, index=df.index, dtype=float)
        out["num_views"] = num_views

        cpu = (_numeric(df, "actual_cpu_hz") / 1e9).replace(0, np.nan)
        gpu = (_numeric(df, "actual_gpu_hz") / 1e9).replace(0, np.nan)
        emc = (_numeric(df, "actual_emc_hz") / 1e9).replace(0, np.nan)
        out["cpu_freq_ghz"] = cpu
        out["gpu_freq_ghz"] = gpu
        out["emc_freq_ghz"] = emc
        out["inv_cpu_freq_ghz"] = 1.0 / cpu
        out["inv_gpu_freq_ghz"] = 1.0 / gpu
        out["inv_emc_freq_ghz"] = 1.0 / emc

        log_text = out["log1p_text_length_target"]
        out["denoising_over_gpu"] = denoising_steps / gpu
        out["denoising_over_emc"] = denoising_steps / emc
        out["text_over_cpu"] = log_text / cpu
        out["text_over_gpu"] = log_text / gpu
        out["views_over_emc"] = num_views / emc
        out["denoising_times_text"] = denoising_steps * log_text
        out["denoising_times_views"] = denoising_steps * num_views
        out["gpu_times_emc"] = gpu * emc
        out["cpu_times_gpu"] = cpu * gpu

        if self.include_deadline_feature:
            if "fixed_period_ms" not in df.columns:
                raise ValueError("include_deadline_feature=True requires fixed_period_ms")
            out["fixed_period_ms"] = _numeric(df, "fixed_period_ms")

        if out.isna().any().any() or np.isinf(out.to_numpy(dtype=float)).any():
            bad = out.columns[out.isna().any() | np.isinf(out.to_numpy(dtype=float)).any(axis=0)].tolist()
            raise ValueError(f"Feature matrix contains NaN or infinite values in columns: {bad}")
        return out.astype(float)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="raise").astype(float)


def _numeric_or_default(df: pd.DataFrame, column: str, default: float | pd.Series) -> pd.Series:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        return values.fillna(default)
    if isinstance(default, pd.Series):
        return default.astype(float)
    return pd.Series(float(default), index=df.index, dtype=float)
