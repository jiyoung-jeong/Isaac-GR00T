from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mars_surrogate.features import FeatureBuilder
from mars_surrogate.schema import validate_summary_df


CLASSIFIER_TYPES = ("logistic_regression", "random_forest_classifier", "gbdt_classifier", "auto")


class StrictFeasibilityClassifier:
    def __init__(self, model_type: str = "auto", *, random_state: int = 0) -> None:
        if model_type not in CLASSIFIER_TYPES:
            raise ValueError(f"model_type must be one of {CLASSIFIER_TYPES}, got {model_type!r}")
        self.model_type = model_type
        self.random_state = random_state
        self.feature_builder = FeatureBuilder(include_deadline_feature=True)
        self.model: Any | None = None
        self.resolved_model_type_: str | None = None
        self.model_selection_: dict[str, Any] | None = None

    @property
    def feature_columns(self) -> list[str]:
        return self.feature_builder.get_feature_names()

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, Any]:
        train_df = validate_summary_df(train_df)
        if val_df is not None:
            val_df = validate_summary_df(val_df)
        if self.model_type == "auto":
            return self._fit_auto(train_df, val_df)
        self._fit_single(train_df, self.model_type)
        self.resolved_model_type_ = self.model_type
        return self.evaluate(val_df if val_df is not None and not val_df.empty else train_df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("StrictFeasibilityClassifier must be fitted before predict_proba().")
        rows = _ensure_deadline_column(df)
        features = self.feature_builder.transform(rows)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[:, 1]
        scores = self.model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        y = strict_feasible_labels(df)
        prob = self.predict_proba(df)
        pred = (prob >= 0.5).astype(int)
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }
        metrics["roc_auc"] = _safe_roc_auc(y, prob)
        return metrics

    def save(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path / "feasibility_model.joblib")
        metadata = {
            "model_type": self.model_type,
            "resolved_model_type": self.resolved_model_type_,
            "feature_columns": self.feature_columns,
            "target": "fixed_period_deadline_miss_pct == 0",
        }
        (out_path / "feasibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | Path) -> "StrictFeasibilityClassifier":
        path = Path(out_dir) / "feasibility_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing feasibility classifier artifact: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact {path} is not a StrictFeasibilityClassifier")
        return model

    def _fit_auto(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> dict[str, Any]:
        val = val_df if val_df is not None and not val_df.empty else train_df
        candidates = []
        best_score = -float("inf")
        best: StrictFeasibilityClassifier | None = None
        for model_type in ["logistic_regression", "random_forest_classifier", "gbdt_classifier"]:
            candidate = StrictFeasibilityClassifier(model_type=model_type, random_state=self.random_state)
            candidate._fit_single(train_df, model_type)
            candidate.resolved_model_type_ = model_type
            metrics = candidate.evaluate(val)
            score = metrics["f1"]
            candidates.append({"model_type": model_type, "score": score, "metrics": metrics})
            if score > best_score:
                best_score = score
                best = candidate
        if best is None:
            raise RuntimeError("Auto classifier selection did not train any candidates")
        self.model = best.model
        self.feature_builder = best.feature_builder
        self.resolved_model_type_ = best.resolved_model_type_
        self.model_selection_ = {
            "requested_model": "auto",
            "selected_model": self.resolved_model_type_,
            "selection_score": best_score,
            "candidates": candidates,
        }
        return self.evaluate(val)

    def _fit_single(self, train_df: pd.DataFrame, model_type: str) -> None:
        rows = _ensure_deadline_column(train_df)
        features = self.feature_builder.fit_transform(rows)
        labels = strict_feasible_labels(rows)
        if len(np.unique(labels)) < 2:
            raise ValueError("Strict feasibility classifier needs both feasible and infeasible rows")
        self.model = _make_classifier(model_type, self.random_state)
        self.model.fit(features, labels)


def strict_feasible_labels(df: pd.DataFrame) -> np.ndarray:
    if "fixed_period_deadline_miss_pct" not in df.columns:
        raise ValueError("strict feasibility labels require fixed_period_deadline_miss_pct")
    return (pd.to_numeric(df["fixed_period_deadline_miss_pct"], errors="raise").astype(float) == 0.0).astype(int).to_numpy()


def _ensure_deadline_column(df: pd.DataFrame) -> pd.DataFrame:
    if "fixed_period_ms" not in df.columns:
        raise ValueError("Feasibility classifier input requires fixed_period_ms as deadline feature")
    return df


def _make_classifier(model_type: str, random_state: int):
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_type == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type == "gbdt_classifier":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier type: {model_type}")


def _safe_roc_auc(y: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob))
