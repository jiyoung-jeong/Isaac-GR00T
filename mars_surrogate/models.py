from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mars_surrogate.features import FeatureBuilder
from mars_surrogate.schema import validate_summary_df


MODEL_TYPES = ("ridge", "random_forest", "gbr", "auto")


class SurrogateModel:
    def __init__(
        self,
        model_type: str = "auto",
        *,
        include_deadline_feature: bool = False,
        log_target: bool = False,
        random_state: int = 0,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
        self.model_type = model_type
        self.include_deadline_feature = include_deadline_feature
        self.log_target = log_target
        self.random_state = random_state
        self.feature_builder = FeatureBuilder(include_deadline_feature=include_deadline_feature)
        self.latency_model: Any | None = None
        self.tail_latency_model: Any | None = None
        self.energy_model: Any | None = None
        self.latency_target = "e2e_median_ms"
        self.tail_target = "e2e_max_ms"
        self.energy_target = "vin_energy_j_per_timed_iteration"
        self.resolved_model_type_: str | None = None
        self.model_selection_: dict[str, Any] | None = None

    @property
    def feature_columns(self) -> list[str]:
        return self.feature_builder.get_feature_names()

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        *,
        latency_target: str = "e2e_median_ms",
        tail_target: str = "e2e_max_ms",
        energy_target: str = "vin_energy_j_per_timed_iteration",
    ) -> dict[str, dict[str, float]]:
        self.latency_target = latency_target
        self.tail_target = tail_target
        self.energy_target = energy_target
        for target in [latency_target, tail_target, energy_target]:
            if target not in train_df.columns:
                raise ValueError(f"Training DataFrame is missing target column {target!r}")

        train_df = validate_summary_df(train_df)
        if val_df is not None:
            val_df = validate_summary_df(val_df)

        if self.model_type == "auto":
            return self._fit_auto(train_df, val_df, latency_target, tail_target, energy_target)
        self._fit_single(train_df, self.model_type, latency_target, tail_target, energy_target)
        self.resolved_model_type_ = self.model_type
        if val_df is not None and not val_df.empty:
            return self.evaluate(
                val_df,
                latency_target=latency_target,
                tail_target=tail_target,
                energy_target=energy_target,
            )
        return self.evaluate(
            train_df,
            latency_target=latency_target,
            tail_target=tail_target,
            energy_target=energy_target,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.latency_model is None or self.tail_latency_model is None or self.energy_model is None:
            raise RuntimeError("SurrogateModel must be fitted before predict().")
        features = self.feature_builder.transform(df)
        latency = self.latency_model.predict(features)
        tail_latency = self.tail_latency_model.predict(features)
        energy = self.energy_model.predict(features)
        if self.log_target:
            latency = np.expm1(latency)
            tail_latency = np.expm1(tail_latency)
            energy = np.expm1(energy)
        pred_median = np.clip(latency, 0.0, None)
        pred_tail = np.clip(tail_latency, 0.0, None)
        return pd.DataFrame(
            {
                "pred_median_latency_ms": pred_median,
                "pred_tail_latency_ms": pred_tail,
                "pred_latency_ms": pred_median,
                "pred_energy_j": np.clip(energy, 0.0, None),
            },
            index=df.index,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        latency_target: str | None = None,
        tail_target: str | None = None,
        energy_target: str | None = None,
    ) -> dict[str, dict[str, float]]:
        latency_target = latency_target or self.latency_target
        tail_target = tail_target or self.tail_target
        energy_target = energy_target or self.energy_target
        pred = self.predict(df)
        return {
            "latency": regression_metrics(df[latency_target], pred["pred_median_latency_ms"]),
            "tail_latency": regression_metrics(df[tail_target], pred["pred_tail_latency_ms"]),
            "energy": regression_metrics(df[energy_target], pred["pred_energy_j"]),
        }

    def save(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path / "model.joblib")
        joblib.dump(self.feature_builder, out_path / "feature_builder.joblib")
        metadata = {
            "model_type": self.model_type,
            "resolved_model_type": self.resolved_model_type_,
            "latency_target": self.latency_target,
            "tail_target": self.tail_target,
            "energy_target": self.energy_target,
            "log_target": self.log_target,
            "include_deadline_feature": self.include_deadline_feature,
            "feature_columns": self.feature_columns,
        }
        (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.model_selection_ is not None:
            (out_path / "model_selection.json").write_text(
                json.dumps(_json_safe(self.model_selection_), indent=2),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, out_dir: str | Path) -> "SurrogateModel":
        path = Path(out_dir)
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model artifact: {model_path}")
        model = joblib.load(model_path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact {model_path} is not a SurrogateModel")
        return model

    def _fit_auto(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        latency_target: str,
        tail_target: str,
        energy_target: str,
    ) -> dict[str, dict[str, float]]:
        if val_df is None or val_df.empty:
            if len(train_df) < 5:
                val_df = train_df
                inner_train = train_df
            else:
                inner_train, val_df = train_test_split(
                    train_df,
                    test_size=0.2,
                    random_state=self.random_state,
                )
        else:
            inner_train = train_df

        candidates = []
        best_score = float("inf")
        best: SurrogateModel | None = None
        for model_type in ["ridge", "random_forest", "gbr"]:
            candidate = SurrogateModel(
                model_type=model_type,
                include_deadline_feature=self.include_deadline_feature,
                log_target=self.log_target,
                random_state=self.random_state,
            )
            candidate._fit_single(inner_train, model_type, latency_target, tail_target, energy_target)
            candidate.resolved_model_type_ = model_type
            metrics = candidate.evaluate(
                val_df,
                latency_target=latency_target,
                tail_target=tail_target,
                energy_target=energy_target,
            )
            score = metrics["latency"]["mape"] + metrics["tail_latency"]["mape"] + metrics["energy"]["mape"]
            candidates.append({"model_type": model_type, "score": score, "metrics": metrics})
            if score < best_score:
                best_score = score
                best = candidate

        if best is None:
            raise RuntimeError("Auto model selection did not train any candidates")
        self.latency_model = deepcopy(best.latency_model)
        self.tail_latency_model = deepcopy(best.tail_latency_model)
        self.energy_model = deepcopy(best.energy_model)
        self.feature_builder = deepcopy(best.feature_builder)
        self.resolved_model_type_ = best.resolved_model_type_
        self.model_selection_ = {
            "requested_model": "auto",
            "selected_model": self.resolved_model_type_,
            "selection_score": best_score,
            "candidates": candidates,
        }
        return self.evaluate(
            val_df,
            latency_target=latency_target,
            tail_target=tail_target,
            energy_target=energy_target,
        )

    def _fit_single(
        self,
        train_df: pd.DataFrame,
        model_type: str,
        latency_target: str,
        tail_target: str,
        energy_target: str,
    ) -> None:
        features = self.feature_builder.fit_transform(train_df)
        latency_y = _target(train_df[latency_target], self.log_target)
        tail_y = _target(train_df[tail_target], self.log_target)
        energy_y = _target(train_df[energy_target], self.log_target)
        self.latency_model = _make_regressor(model_type, self.random_state)
        self.tail_latency_model = _make_regressor(model_type, self.random_state)
        self.energy_model = _make_regressor(model_type, self.random_state)
        self.latency_model.fit(features, latency_y)
        self.tail_latency_model.fit(features, tail_y)
        self.energy_model.fit(features, energy_y)


def _make_regressor(model_type: str, random_state: int):
    if model_type == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV())])
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_type == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unknown model type: {model_type}")


def _target(series: pd.Series, log_target: bool) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").astype(float).to_numpy()
    if (values < 0).any():
        raise ValueError("Targets must be non-negative")
    return np.log1p(values) if log_target else values


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(actual, errors="raise").astype(float).to_numpy()
    pred = pd.to_numeric(predicted, errors="raise").astype(float).to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse,
        "mape": _mape(y, pred),
        "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
    }


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), 1e-9)
    return float(np.mean(np.abs((actual - predicted) / denom)) * 100.0)


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
