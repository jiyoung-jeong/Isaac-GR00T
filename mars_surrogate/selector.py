from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mars_surrogate.schema import OPP_COLUMNS


@dataclass(frozen=True)
class OPP:
    cpu_hz: int
    gpu_hz: int
    emc_hz: int


class ModeAwareSelector:
    def __init__(
        self,
        surrogate_model,
        candidate_opps: pd.DataFrame,
        feasibility_model=None,
        safety_margin_ms: float = 0.0,
        hysteresis_ms: float = 0.0,
    ) -> None:
        missing = [column for column in OPP_COLUMNS if column not in candidate_opps.columns]
        if missing:
            raise ValueError(f"candidate_opps missing required columns: {missing}")
        self.surrogate_model = surrogate_model
        self.feasibility_model = feasibility_model
        self.candidate_opps = candidate_opps[OPP_COLUMNS].drop_duplicates().reset_index(drop=True)
        if self.candidate_opps.empty:
            raise ValueError("candidate_opps must contain at least one OPP")
        self.safety_margin_ms = float(safety_margin_ms)
        self.hysteresis_ms = float(hysteresis_ms)

    def select_eco(
        self,
        metadata: dict[str, Any],
        deadline_ms: float,
        prev_opp: dict[str, int] | OPP | None = None,
    ) -> dict[str, Any]:
        return self.select_tail_aware(
            metadata,
            deadline_ms,
            mode="median_margin",
            prev_opp=prev_opp,
        )

    def select_tail_aware(
        self,
        metadata: dict[str, Any],
        deadline_ms: float,
        *,
        mode: str = "median_margin",
        feasible_prob_threshold: float = 0.5,
        prev_opp: dict[str, int] | OPP | None = None,
    ) -> dict[str, Any]:
        del prev_opp
        if mode not in {"median_margin", "tail", "risk"}:
            raise ValueError("mode must be one of {'median_margin', 'tail', 'risk'}")

        candidate_rows = self._candidate_rows(metadata)
        candidate_rows["fixed_period_ms"] = float(deadline_ms)
        predictions = self.surrogate_model.predict(candidate_rows)
        table = pd.concat(
            [candidate_rows[OPP_COLUMNS].reset_index(drop=True), predictions.reset_index(drop=True)],
            axis=1,
        )

        if mode == "risk":
            if self.feasibility_model is None:
                raise ValueError("risk selector mode requires a feasibility_model")
            table["pred_strict_feasible_prob"] = self.feasibility_model.predict_proba(candidate_rows)
        else:
            table["pred_strict_feasible_prob"] = float("nan")

        median_latency = table["pred_median_latency_ms"] if "pred_median_latency_ms" in table else table["pred_latency_ms"]
        tail_latency = table["pred_tail_latency_ms"] if "pred_tail_latency_ms" in table else median_latency
        if mode == "median_margin":
            table["selector_latency_ms"] = median_latency
        else:
            table["selector_latency_ms"] = tail_latency
        table["effective_latency_ms"] = table["selector_latency_ms"] + self.safety_margin_ms
        table["feasible_predicted"] = table["effective_latency_ms"] <= float(deadline_ms)
        if mode == "risk":
            table["feasible_predicted"] &= table["pred_strict_feasible_prob"] >= float(feasible_prob_threshold)

        feasible = table[table["feasible_predicted"]]
        if feasible.empty:
            selected = table.sort_values(["selector_latency_ms", "pred_energy_j"], ascending=True).iloc[0]
            feasible_predicted = False
        else:
            selected = feasible.sort_values(["pred_energy_j", "selector_latency_ms"], ascending=True).iloc[0]
            feasible_predicted = True

        selected_opp = {column: int(selected[column]) for column in OPP_COLUMNS}
        return {
            "selected_opp": selected_opp,
            "deadline_ms": float(deadline_ms),
            "mode": mode,
            "feasible_prob_threshold": float(feasible_prob_threshold),
            "pred_latency_ms": float(selected.get("pred_latency_ms", selected["selector_latency_ms"])),
            "pred_median_latency_ms": float(
                selected.get("pred_median_latency_ms", selected.get("pred_latency_ms", selected["selector_latency_ms"]))
            ),
            "pred_tail_latency_ms": float(selected.get("pred_tail_latency_ms", selected["selector_latency_ms"])),
            "pred_energy_j": float(selected["pred_energy_j"]),
            "pred_strict_feasible_prob": _safe_float(selected.get("pred_strict_feasible_prob")),
            "effective_latency_ms": float(selected["effective_latency_ms"]),
            "feasible_predicted": bool(feasible_predicted),
            "candidate_predictions": table,
        }

    def _candidate_rows(self, metadata: dict[str, Any]) -> pd.DataFrame:
        rows = self.candidate_opps.copy()
        for key, value in metadata.items():
            rows[key] = value
        return rows


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
