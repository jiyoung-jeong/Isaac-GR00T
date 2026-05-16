#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_summary(random_state: int = 0, noise: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    text_lengths = [16, 32, 64, 128]
    denoising_steps = [4, 8, 12]
    cpu_hz = [729_600_000, 1_497_600_000, 2_304_000_000]
    gpu_hz = [612_000_000, 918_000_000, 1_224_000_000]
    emc_hz = [1_600_000_000, 2_133_000_000, 3_200_000_000]
    rows = []
    repeat_id = 0
    for text in text_lengths:
        for steps in denoising_steps:
            for cpu in cpu_hz:
                for gpu in gpu_hz:
                    for emc in emc_hz:
                        cpu_ghz = cpu / 1e9
                        gpu_ghz = gpu / 1e9
                        emc_ghz = emc / 1e9
                        log_text = np.log1p(text)
                        latency = (
                            14.0
                            + 2.3 * log_text
                            + 5.5 * steps / gpu_ghz
                            + 1.7 * log_text / cpu_ghz
                            + 2.2 * steps / emc_ghz
                        )
                        latency *= 1.0 + rng.normal(0.0, noise)
                        power = 2.8 + 1.6 * gpu_ghz + 0.55 * cpu_ghz + 0.35 * emc_ghz
                        energy = latency / 1000.0 * power
                        energy *= 1.0 + rng.normal(0.0, noise)
                        deadline = 120.0 if steps <= 8 else 170.0
                        rows.append(
                            {
                                "repeat_id": repeat_id,
                                "config_name": f"cpu{cpu}_gpu{gpu}_emc{emc}",
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
                                "fixed_period_deadline_misses": int(latency * 1.08 > deadline),
                                "fixed_period_deadline_miss_pct": 100.0 if latency * 1.08 > deadline else 0.0,
                                "fixed_period_elapsed_median_ms": deadline,
                                "fixed_period_elapsed_mean_ms": deadline,
                                "e2e_median_ms": latency,
                                "e2e_mean_ms": latency * 1.02,
                                "e2e_max_ms": latency * 1.08,
                                "vin_energy_j": energy * 10,
                                "vin_energy_j_per_timed_iteration": energy,
                                "gpu_energy_j_per_timed_iteration": energy * 0.46,
                                "cpu_soc_mss_energy_j_per_timed_iteration": energy * 0.32,
                            }
                        )
                        repeat_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic fixed-period summary.csv for tests.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.02)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_summary(args.random_state, args.noise).to_csv(args.out, index=False)
    print(f"Wrote synthetic summary to {args.out}")


if __name__ == "__main__":
    main()
