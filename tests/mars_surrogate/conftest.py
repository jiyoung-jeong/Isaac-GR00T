from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_summary_df() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for text in [16, 64, 128]:
        for steps in [4, 8]:
            for cpu in [800_000_000, 1_600_000_000]:
                for gpu in [600_000_000, 1_200_000_000]:
                    for emc in [1_600_000_000, 3_200_000_000]:
                        cpu_ghz = cpu / 1e9
                        gpu_ghz = gpu / 1e9
                        emc_ghz = emc / 1e9
                        log_text = np.log1p(text)
                        latency = (
                            12.0
                            + 1.5 * log_text
                            + 5.0 * steps / gpu_ghz
                            + 1.2 * log_text / cpu_ghz
                            + 1.8 * steps / emc_ghz
                            + rng.normal(0.0, 0.1)
                        )
                        power = 2.4 + 1.2 * gpu_ghz + 0.5 * cpu_ghz + 0.25 * emc_ghz
                        energy = latency / 1000.0 * power + rng.normal(0.0, 0.001)
                        deadline = 75.0 if steps == 4 else 105.0
                        rows.append(
                            {
                                "repeat_id": len(rows),
                                "config_name": "synthetic",
                                "cpu_label": str(cpu),
                                "gpu_label": str(gpu),
                                "emc_label": str(emc),
                                "text_length_target": text,
                                "actual_text_words": text,
                                "input_token_count": text * 4,
                                "num_views": 2,
                                "denoising_steps": steps,
                                "actual_cpu_hz": cpu,
                                "actual_gpu_hz": gpu,
                                "actual_emc_hz": emc,
                                "fixed_period_ms": deadline,
                                "fixed_period_deadline_misses": int(latency * 1.1 > deadline),
                                "fixed_period_deadline_miss_pct": 100.0 if latency * 1.1 > deadline else 0.0,
                                "fixed_period_elapsed_median_ms": deadline,
                                "fixed_period_elapsed_mean_ms": deadline,
                                "e2e_median_ms": latency,
                                "e2e_mean_ms": latency * 1.01,
                                "e2e_max_ms": latency * 1.1,
                                "vin_energy_j": energy * 10.0,
                                "vin_energy_j_per_timed_iteration": energy,
                                "gpu_energy_j_per_timed_iteration": energy * 0.45,
                                "cpu_soc_mss_energy_j_per_timed_iteration": energy * 0.3,
                            }
                        )
    return pd.DataFrame(rows)
