#!/usr/bin/env python3
"""
Plot per-inference average power grouped by backend.

For each backend (pytorch / torchcompile / tensorRT), scan result/thor_gr00t_server_*/inference_energy.csv,
compute per-inference average power excluding 1st inference:
  P_gpu_W = E_gpu_J / duration_s
  P_vin_W = E_vin_J / duration_s
and overlay series for each condition suffix (everything after _<backend> in run dir name).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BACKENDS = ["pytorch", "torchcompile", "tensorRT"]


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    backend: str
    suffix: str
    csv_path: Path


def parse_run_name(name: str) -> Optional[Tuple[str, str]]:
    for b in BACKENDS:
        token = f"_{b}"
        if token in name:
            _, rest = name.split(token, 1)
            suffix = rest[1:] if rest.startswith("_") else rest
            return b, (suffix if suffix else "(no_suffix)")
    return None


def load_power_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"inference_id", "duration_ms", "E_gpu_J", "E_vin_J"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")
    df = df[df["inference_id"] >= 2].copy()
    if len(df) == 0:
        raise ValueError(f"No inferences after excluding 1st in {csv_path}")
    t_s = df["duration_ms"].astype(float) / 1e3
    df["P_gpu_W"] = df["E_gpu_J"].astype(float) / t_s
    df["P_vin_W"] = df["E_vin_J"].astype(float) / t_s
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-dir",
        default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result",
        help="Directory containing thor_gr00t_server_* run dirs",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/Thor/compare_plots_by_backend",
        help="Output directory for generated PNGs (must be writable)",
    )
    ap.add_argument(
        "--only-suffix-regex",
        default="",
        help="If set, only include runs whose suffix matches this regex.",
    )
    ap.add_argument(
        "--max-series",
        type=int,
        default=20,
        help="Max number of series per backend plot (default 20). Long legends get unreadable.",
    )
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_re = re.compile(args.only_suffix_regex) if args.only_suffix_regex else None

    runs: List[RunInfo] = []
    for csv_path in sorted(result_dir.glob("thor_gr00t_server_*/inference_energy.csv")):
        run_dir = csv_path.parent
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        backend, suffix = parsed
        if suffix_re and not suffix_re.search(suffix):
            continue
        runs.append(RunInfo(run_dir=run_dir, backend=backend, suffix=suffix, csv_path=csv_path))

    # backend -> suffix -> RunInfo (keep latest if duplicates)
    by_backend: Dict[str, Dict[str, RunInfo]] = {b: {} for b in BACKENDS}
    for r in runs:
        by_backend[r.backend][r.suffix] = r

    written = 0
    for backend in BACKENDS:
        suffix_map = by_backend.get(backend, {})
        if not suffix_map:
            continue

        # stable order: shorter suffix first, then lexicographic
        suffixes = sorted(suffix_map.keys(), key=lambda s: (len(s), s))
        if len(suffixes) > args.max_series:
            suffixes = suffixes[: args.max_series]

        fig, axes = plt.subplots(2, 1, figsize=(12, max(7, 0.35 * len(suffixes) + 5)), sharex=True)

        for suffix in suffixes:
            r = suffix_map[suffix]
            df = load_power_df(r.csv_path)
            label = suffix
            axes[0].plot(df["inference_id"], df["P_gpu_W"], marker="o", linewidth=1.4, label=label)
            axes[1].plot(df["inference_id"], df["P_vin_W"], marker="o", linewidth=1.4, label=label)

        axes[0].set_ylabel("GPU avg power (W) = E_gpu_J / duration_s")
        axes[1].set_ylabel("VIN avg power (W) = E_vin_J / duration_s")
        axes[1].set_xlabel("inference_id")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8, ncol=1)
        axes[1].legend(loc="upper right", fontsize=8, ncol=1)
        fig.suptitle(f"{backend} — per-inference avg power (exclude 1st)")
        plt.tight_layout()

        out_path = out_dir / f"{backend}__compare_inference_power.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        written += 1

    print(f"[OK] result_dir={result_dir}")
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] written={written}")


if __name__ == "__main__":
    main()

