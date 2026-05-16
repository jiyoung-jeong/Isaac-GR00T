from __future__ import annotations

from mars_surrogate.evaluation import evaluate_eco_selector, oracle_feasibility_report
from mars_surrogate.models import SurrogateModel
from mars_surrogate.schema import OPP_COLUMNS


def test_evaluation_regret_columns_exist(synthetic_summary_df):
    model = SurrogateModel(model_type="ridge", log_target=True)
    model.fit(synthetic_summary_df)
    candidates = synthetic_summary_df[OPP_COLUMNS].drop_duplicates()
    decisions, metrics = evaluate_eco_selector(synthetic_summary_df, model, candidates)
    for column in ["energy_regret_j", "energy_regret_pct", "latency_regret_ms", "top1_match"]:
        assert column in decisions.columns
    assert metrics["n_groups"] == synthetic_summary_df.groupby(["text_length_target", "denoising_steps"]).ngroups


def test_oracle_ceiling_report(synthetic_summary_df):
    report, summary = oracle_feasibility_report(synthetic_summary_df)
    assert "num_strict_feasible_opps" in report.columns
    assert "oracle_strict_possible" in report.columns
    assert summary["n_groups"] == synthetic_summary_df.groupby(["text_length_target", "denoising_steps"]).ngroups
