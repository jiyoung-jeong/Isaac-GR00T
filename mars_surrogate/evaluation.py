from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mars_surrogate.oracle import oracle_eco
from mars_surrogate.schema import OPP_COLUMNS, validate_summary_df
from mars_surrogate.selector import ModeAwareSelector


def evaluate_eco_selector(
    df: pd.DataFrame,
    surrogate_model,
    candidate_opps: pd.DataFrame,
    train_mask: pd.Series | None = None,
    group_cols: tuple[str, ...] = ("text_length_target", "denoising_steps"),
    feasibility: str = "strict",
    safety_margin_ms: float = 0.0,
    selector_mode: str = "median_margin",
    feasibility_model=None,
    feasible_prob_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    del train_mask
    clean = validate_summary_df(df)
    ceiling, _ = oracle_feasibility_report(clean, group_cols=group_cols)
    possible_keys = _oracle_possible_key_set(ceiling, group_cols)
    selector = ModeAwareSelector(
        surrogate_model,
        candidate_opps,
        feasibility_model=feasibility_model,
        safety_margin_ms=safety_margin_ms,
    )
    rows = []
    for key, group in clean.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        first = group.iloc[0]
        metadata = _metadata_from_row(first)
        deadline_ms = float(first["fixed_period_ms"])
        selection = selector.select_tail_aware(
            metadata,
            deadline_ms=deadline_ms,
            mode=selector_mode,
            feasible_prob_threshold=feasible_prob_threshold,
        )
        selected_opp = selection["selected_opp"]
        selected_row = _lookup_selected(group, selected_opp)
        oracle_row = oracle_eco(group, feasibility=feasibility)
        oracle_possible = key_tuple in possible_keys
        rows.append(
            _decision_row(
                group_cols,
                key_tuple,
                deadline_ms,
                selection,
                selected_row,
                oracle_row,
                oracle_possible=oracle_possible,
            )
        )

    decisions = pd.DataFrame(rows)
    return decisions, summarize_eco_decisions(decisions)


def compare_selector_variants(
    df: pd.DataFrame,
    surrogate_model,
    candidate_opps: pd.DataFrame,
    *,
    feasibility_model=None,
    group_cols: tuple[str, ...] = ("text_length_target", "denoising_steps"),
    feasibility: str = "strict",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for margin in [0.0, 5.0, 10.0, 15.0]:
        variants.append(
            {
                "variant": f"median_margin_{margin:g}ms",
                "selector_mode": "median_margin",
                "safety_margin_ms": margin,
                "feasible_prob_threshold": np.nan,
            }
        )
    variants.append(
        {
            "variant": "tail",
            "selector_mode": "tail",
            "safety_margin_ms": 0.0,
            "feasible_prob_threshold": np.nan,
        }
    )
    for threshold in [0.5, 0.7, 0.9]:
        variants.append(
            {
                "variant": f"risk_p{threshold:g}",
                "selector_mode": "risk",
                "safety_margin_ms": 0.0,
                "feasible_prob_threshold": threshold,
            }
        )

    rows = []
    details: dict[str, Any] = {}
    for variant in variants:
        if variant["selector_mode"] == "risk" and feasibility_model is None:
            continue
        decisions, metrics = evaluate_eco_selector(
            df,
            surrogate_model,
            candidate_opps,
            group_cols=group_cols,
            feasibility=feasibility,
            safety_margin_ms=float(variant["safety_margin_ms"]),
            selector_mode=variant["selector_mode"],
            feasibility_model=feasibility_model,
            feasible_prob_threshold=float(variant["feasible_prob_threshold"])
            if not pd.isna(variant["feasible_prob_threshold"])
            else 0.5,
        )
        row = {**variant, **metrics}
        rows.append(row)
        details[variant["variant"]] = {"metrics": metrics, "n_decisions": int(len(decisions))}
    frame = pd.DataFrame(rows)
    return frame, details


def oracle_feasibility_report(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("text_length_target", "denoising_steps"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean = validate_summary_df(df)
    rows = []
    for key, group in clean.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        feasible = group[group["fixed_period_deadline_miss_pct"].astype(float) == 0.0]
        row: dict[str, Any] = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "deadline_ms": float(group["fixed_period_ms"].iloc[0]),
                "num_total_opps": int(group[OPP_COLUMNS].drop_duplicates().shape[0]),
                "num_strict_feasible_opps": int(feasible[OPP_COLUMNS].drop_duplicates().shape[0]),
                "oracle_strict_possible": bool(not feasible.empty),
            }
        )
        if feasible.empty:
            row.update(
                {
                    "oracle_best_energy_j": np.nan,
                    "oracle_best_latency_median_ms": np.nan,
                    "oracle_best_latency_max_ms": np.nan,
                }
            )
        else:
            best = feasible.sort_values(["vin_energy_j_per_timed_iteration", "e2e_median_ms"]).iloc[0]
            row.update(
                {
                    "oracle_best_energy_j": float(best["vin_energy_j_per_timed_iteration"]),
                    "oracle_best_latency_median_ms": float(best["e2e_median_ms"]),
                    "oracle_best_latency_max_ms": float(best["e2e_max_ms"]),
                }
            )
        rows.append(row)
    report = pd.DataFrame(rows)
    summary = {
        "n_groups": int(len(report)),
        "oracle_possible_groups": int(report["oracle_strict_possible"].sum()) if not report.empty else 0,
        "oracle_possible_rate": float(report["oracle_strict_possible"].mean()) if not report.empty else float("nan"),
        "mean_num_strict_feasible_opps": float(report["num_strict_feasible_opps"].mean()) if not report.empty else float("nan"),
    }
    return report, summary


def summarize_eco_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "strict_success_rate": float("nan"),
            "deadline_miss_rate": float("nan"),
            "mean_energy_regret_j": float("nan"),
            "mean_energy_regret_pct": float("nan"),
            "mean_latency_regret_ms": float("nan"),
            "top1_match_rate": float("nan"),
            "n_groups": 0,
            "oracle_possible_groups": 0,
            "selector_success_among_oracle_possible_groups": float("nan"),
        }
    possible = decisions[decisions["oracle_strict_possible"]]
    strict_success = decisions["selected_actual_feasible_strict"].astype(bool)
    metrics = {
        "strict_success_rate": float(strict_success.mean()),
        "deadline_miss_rate": float((~strict_success).mean()),
        "mean_energy_regret_j": float(decisions["energy_regret_j"].mean()),
        "mean_energy_regret_pct": float(decisions["energy_regret_pct"].mean()),
        "mean_latency_regret_ms": float(decisions["latency_regret_ms"].mean()),
        "top1_match_rate": float(decisions["top1_match"].mean()),
        "n_groups": int(len(decisions)),
        "oracle_possible_groups": int(decisions["oracle_strict_possible"].sum()),
        "selector_success_among_oracle_possible_groups": float(possible["selected_actual_feasible_strict"].mean())
        if not possible.empty
        else float("nan"),
    }
    metrics["strict_deadline_success_rate"] = metrics["strict_success_rate"]
    return metrics


def _metadata_from_row(row: pd.Series) -> dict[str, Any]:
    metadata = {
        "text_length_target": row["text_length_target"],
        "denoising_steps": row["denoising_steps"],
    }
    for column in ["actual_text_words", "input_token_count", "num_views"]:
        if column in row.index and pd.notna(row[column]):
            metadata[column] = row[column]
    return metadata


def _lookup_selected(group: pd.DataFrame, selected_opp: dict[str, int]) -> pd.Series:
    mask = pd.Series(True, index=group.index)
    for column in OPP_COLUMNS:
        mask &= group[column].astype(int) == int(selected_opp[column])
    matches = group[mask]
    if matches.empty:
        out = pd.Series(dtype=float)
        for column in [
            *OPP_COLUMNS,
            "e2e_median_ms",
            "e2e_max_ms",
            "vin_energy_j_per_timed_iteration",
            "fixed_period_deadline_miss_pct",
        ]:
            out[column] = np.nan
        return out
    return matches.sort_values(
        ["vin_energy_j_per_timed_iteration", "e2e_median_ms"],
        ascending=True,
    ).iloc[0]


def _decision_row(
    group_cols: tuple[str, ...],
    key_tuple: tuple[Any, ...],
    deadline_ms: float,
    selection: dict[str, Any],
    selected: pd.Series,
    oracle: pd.Series,
    *,
    oracle_possible: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(zip(group_cols, key_tuple))
    selected_opp = selection["selected_opp"]
    selected_energy = float(selected["vin_energy_j_per_timed_iteration"])
    oracle_energy = float(oracle["vin_energy_j_per_timed_iteration"])
    selected_latency = float(selected["e2e_median_ms"])
    oracle_latency = float(oracle["e2e_median_ms"])
    row.update(
        {
            "deadline_ms": deadline_ms,
            "selector_mode": selection.get("mode", "median_margin"),
            "selected_cpu_hz": selected_opp["actual_cpu_hz"],
            "selected_gpu_hz": selected_opp["actual_gpu_hz"],
            "selected_emc_hz": selected_opp["actual_emc_hz"],
            "selected_pred_latency_ms": selection["pred_median_latency_ms"],
            "selected_pred_median_latency_ms": selection["pred_median_latency_ms"],
            "selected_pred_tail_latency_ms": selection["pred_tail_latency_ms"],
            "selected_pred_energy_j": selection["pred_energy_j"],
            "selected_pred_strict_feasible_prob": selection.get("pred_strict_feasible_prob"),
            "selected_actual_e2e_median_ms": selected_latency,
            "selected_actual_e2e_max_ms": float(selected["e2e_max_ms"]),
            "selected_actual_energy_j": selected_energy,
            "selected_actual_deadline_miss_pct": float(selected["fixed_period_deadline_miss_pct"]),
            "selected_actual_feasible_strict": bool(float(selected["fixed_period_deadline_miss_pct"]) == 0.0),
            "oracle_cpu_hz": int(oracle["actual_cpu_hz"]),
            "oracle_gpu_hz": int(oracle["actual_gpu_hz"]),
            "oracle_emc_hz": int(oracle["actual_emc_hz"]),
            "oracle_actual_e2e_median_ms": oracle_latency,
            "oracle_actual_e2e_max_ms": float(oracle["e2e_max_ms"]),
            "oracle_actual_energy_j": oracle_energy,
            "oracle_actual_deadline_miss_pct": float(oracle["fixed_period_deadline_miss_pct"]),
            "oracle_infeasible": bool(oracle.get("oracle_infeasible", False)),
            "oracle_strict_possible": bool(oracle_possible),
            "energy_regret_j": selected_energy - oracle_energy,
            "energy_regret_pct": (selected_energy - oracle_energy) / oracle_energy * 100.0
            if oracle_energy != 0
            else np.nan,
            "latency_regret_ms": selected_latency - oracle_latency,
            "top1_match": all(int(selected_opp[c]) == int(oracle[c]) for c in OPP_COLUMNS),
            "deadline_miss_selected": float(selected["fixed_period_deadline_miss_pct"]) > 0.0,
        }
    )
    return row


def _oracle_possible_key_set(report: pd.DataFrame, group_cols: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if report.empty:
        return set()
    keys = set()
    for _, row in report[report["oracle_strict_possible"]].iterrows():
        keys.add(tuple(row[column] for column in group_cols))
    return keys
