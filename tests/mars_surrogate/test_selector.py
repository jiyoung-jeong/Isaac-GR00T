from __future__ import annotations

import pandas as pd

from mars_surrogate.selector import ModeAwareSelector


class FakeModel:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        gpu_ghz = df["actual_gpu_hz"] / 1e9
        return pd.DataFrame(
            {
                "pred_median_latency_ms": 100.0 / gpu_ghz,
                "pred_tail_latency_ms": 120.0 / gpu_ghz,
                "pred_latency_ms": 100.0 / gpu_ghz,
                "pred_energy_j": gpu_ghz,
            },
            index=df.index,
        )


def test_selector_eco_respects_deadline():
    candidates = pd.DataFrame(
        {
            "actual_cpu_hz": [1_000_000_000, 1_000_000_000],
            "actual_gpu_hz": [1_000_000_000, 2_000_000_000],
            "actual_emc_hz": [2_000_000_000, 2_000_000_000],
        }
    )
    selector = ModeAwareSelector(FakeModel(), candidates)
    result = selector.select_eco({"text_length_target": 64, "denoising_steps": 4}, deadline_ms=75)
    assert result["feasible_predicted"] is True
    assert result["selected_opp"]["actual_gpu_hz"] == 2_000_000_000
    assert result["effective_latency_ms"] <= 75


def test_selector_fallback_when_no_feasible_candidate():
    candidates = pd.DataFrame(
        {
            "actual_cpu_hz": [1_000_000_000, 1_000_000_000],
            "actual_gpu_hz": [1_000_000_000, 2_000_000_000],
            "actual_emc_hz": [2_000_000_000, 2_000_000_000],
        }
    )
    selector = ModeAwareSelector(FakeModel(), candidates)
    result = selector.select_eco({"text_length_target": 64, "denoising_steps": 4}, deadline_ms=10)
    assert result["feasible_predicted"] is False
    assert result["selected_opp"]["actual_gpu_hz"] == 2_000_000_000


def test_tail_aware_selector_uses_tail_latency():
    candidates = pd.DataFrame(
        {
            "actual_cpu_hz": [1_000_000_000, 1_000_000_000],
            "actual_gpu_hz": [1_000_000_000, 2_000_000_000],
            "actual_emc_hz": [2_000_000_000, 2_000_000_000],
        }
    )
    selector = ModeAwareSelector(FakeModel(), candidates)
    result = selector.select_tail_aware({"text_length_target": 64, "denoising_steps": 4}, deadline_ms=75, mode="tail")
    assert result["selected_opp"]["actual_gpu_hz"] == 2_000_000_000
    assert result["pred_tail_latency_ms"] <= 75
