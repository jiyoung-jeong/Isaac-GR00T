#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mars_surrogate.schema import validate_summary_df


def find_summary_csvs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            paths.extend(sorted(item.rglob("summary.csv")))
        else:
            raise FileNotFoundError(f"Input path does not exist: {item}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("No summary.csv files found")
    return unique


def merge_summary_csvs(inputs: list[Path]) -> pd.DataFrame:
    frames = []
    for path in find_summary_csvs(inputs):
        df = validate_summary_df(pd.read_csv(path))
        df.insert(0, "source_summary_csv", str(path))
        df.insert(1, "source_case", _source_case(path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _source_case(path: Path) -> str:
    for parent in path.parents:
        name = parent.name
        if name.startswith("d") and "_p" in name:
            return name
    return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge one or more measured summary.csv files.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="summary.csv files or directories to search recursively",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    merged = merge_summary_csvs(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {len(merged)} rows from {merged['source_summary_csv'].nunique()} summaries to {args.out}")


if __name__ == "__main__":
    main()
