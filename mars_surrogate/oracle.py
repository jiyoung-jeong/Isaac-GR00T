from __future__ import annotations

import numpy as np
import pandas as pd


def oracle_eco(df_group: pd.DataFrame, feasibility: str = "strict") -> pd.Series:
    feasible_mask = _feasible_mask(df_group, feasibility)
    if feasible_mask.any():
        chosen = _sort_candidates(df_group[feasible_mask], ["vin_energy_j_per_timed_iteration", "e2e_median_ms"]).iloc[0]
        chosen = chosen.copy()
        chosen["oracle_infeasible"] = False
        return chosen
    chosen = _sort_candidates(df_group, ["e2e_median_ms", "vin_energy_j_per_timed_iteration"]).iloc[0].copy()
    chosen["oracle_infeasible"] = True
    return chosen


def oracle_sprint(df_group: pd.DataFrame) -> pd.Series:
    return _sort_candidates(df_group, ["e2e_median_ms", "vin_energy_j_per_timed_iteration"]).iloc[0].copy()


def oracle_balanced(df_group: pd.DataFrame) -> pd.Series:
    front = pareto_front(df_group, latency_col="e2e_median_ms", energy_col="vin_energy_j_per_timed_iteration")
    latency = _normalize(front["e2e_median_ms"])
    energy = _normalize(front["vin_energy_j_per_timed_iteration"])
    idx = int(np.argmin(np.sqrt(latency**2 + energy**2)))
    return front.iloc[idx].copy()


def pareto_front(df: pd.DataFrame, *, latency_col: str, energy_col: str) -> pd.DataFrame:
    ordered = df.sort_values([latency_col, energy_col], ascending=True).reset_index(drop=True)
    keep = []
    best_energy = float("inf")
    for idx, row in ordered.iterrows():
        energy = float(row[energy_col])
        if energy < best_energy:
            keep.append(idx)
            best_energy = energy
    return ordered.loc[keep].reset_index(drop=True)


def _feasible_mask(df: pd.DataFrame, feasibility: str) -> pd.Series:
    if feasibility == "strict":
        return df["fixed_period_deadline_miss_pct"].astype(float) == 0.0
    if feasibility == "max_latency":
        return df["e2e_max_ms"].astype(float) <= df["fixed_period_ms"].astype(float)
    if feasibility == "median_latency":
        return df["e2e_median_ms"].astype(float) <= df["fixed_period_ms"].astype(float)
    raise ValueError("feasibility must be one of {'strict', 'max_latency', 'median_latency'}")


def _sort_candidates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.sort_values(columns, ascending=[True] * len(columns)).reset_index(drop=True)


def _normalize(series: pd.Series) -> np.ndarray:
    values = series.astype(float).to_numpy()
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return np.zeros_like(values)
    return (values - low) / (high - low)
