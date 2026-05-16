#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mars_surrogate.oracle import pareto_front
from mars_surrogate.schema import OPP_COLUMNS, validate_summary_df


def make_candidates(df: pd.DataFrame, mode: str = "all_unique") -> pd.DataFrame:
    clean = validate_summary_df(df)
    if mode == "all_unique":
        source = clean
    elif mode == "measured_pareto":
        aggregate = clean.groupby(OPP_COLUMNS, as_index=False).agg(
            e2e_median_ms=("e2e_median_ms", "mean"),
            vin_energy_j_per_timed_iteration=("vin_energy_j_per_timed_iteration", "mean"),
        )
        source = pareto_front(
            aggregate,
            latency_col="e2e_median_ms",
            energy_col="vin_energy_j_per_timed_iteration",
        )
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")
    return source[OPP_COLUMNS].drop_duplicates().sort_values(OPP_COLUMNS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a unique measured OPP candidate set.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["all_unique", "measured_pareto"], default="all_unique")
    args = parser.parse_args()

    candidates = make_candidates(pd.read_csv(args.csv), mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.out, index=False)
    print(f"Wrote {len(candidates)} candidate OPPs to {args.out}")


if __name__ == "__main__":
    main()
