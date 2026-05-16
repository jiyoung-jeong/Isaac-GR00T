from __future__ import annotations

from mars_surrogate.oracle import oracle_eco


def test_oracle_eco_strict(synthetic_summary_df):
    group = synthetic_summary_df[
        (synthetic_summary_df["text_length_target"] == 64)
        & (synthetic_summary_df["denoising_steps"] == 4)
    ].copy()
    group.loc[group.index, "fixed_period_deadline_miss_pct"] = 100.0
    feasible_idx = group.index[:2]
    group.loc[feasible_idx, "fixed_period_deadline_miss_pct"] = 0.0
    group.loc[feasible_idx[0], "vin_energy_j_per_timed_iteration"] = 10.0
    group.loc[feasible_idx[1], "vin_energy_j_per_timed_iteration"] = 1.0
    chosen = oracle_eco(group, feasibility="strict")
    assert chosen["vin_energy_j_per_timed_iteration"] == 1.0
    assert chosen["oracle_infeasible"] is False
