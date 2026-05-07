#!/usr/bin/env python3
"""
Scan result/thor_gr00t_server_*/inference_energy.csv and generate comparison plots
grouped by "condition suffix" (everything after the backend token).

Example run names:
  thor_gr00t_server_YYYYMMDD_HHMMSS_tensorRT_gpufreq_801MHz
  thor_gr00t_server_YYYYMMDD_HHMMSS_torchcompile_gpufreq_801MHz
  thor_gr00t_server_YYYYMMDD_HHMMSS_pytorch_gpufreq_801MHz

For each group (same suffix), plot per-inference average power (exclude 1st inference):
  P_gpu_W = E_gpu_J / duration_s
  P_vin_W = E_vin_J / duration_s
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


BACKENDS = ["tensorRT", "torchcompile", "pytorch"]


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    backend: str
    suffix: str
    csv_path: Path


def parse_run_name(name: str) -> Optional[Tuple[str, str]]:
    """
    Return (backend, suffix) from run directory name.
    suffix includes leading "_" if present, e.g. "_gpufreq_801MHz".
    """
    for b in BACKENDS:
        token = f"_{b}"
        if token in name:
            _, rest = name.split(token, 1)
            return b, rest  # rest may be "" or like "_gpufreq_801MHz"
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


def nice_suffix(suffix: str) -> str:
    if not suffix:
        return "(no_suffix)"
    if suffix.startswith("_"):
        return suffix[1:]
    return suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-dir",
        default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result",
        help="Directory containing thor_gr00t_server_* run dirs",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/Thor/compare_plots",
        help="Output directory for generated PNGs (must be writable)",
    )
    ap.add_argument(
        "--only-suffix-regex",
        default="",
        help="If set, only plot groups whose suffix matches this regex (applied to suffix without leading underscore).",
    )
    ap.add_argument(
        "--min-backends",
        type=int,
        default=2,
        help="Only plot groups that have at least this many backends present (default 2).",
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
        sfx = nice_suffix(suffix)
        if suffix_re and not suffix_re.search(sfx):
            continue
        runs.append(RunInfo(run_dir=run_dir, backend=backend, suffix=sfx, csv_path=csv_path))

    # group by suffix
    groups: Dict[str, Dict[str, RunInfo]] = {}
    for r in runs:
        groups.setdefault(r.suffix, {})[r.backend] = r  # latest wins if duplicate

    written = 0
    skipped = 0
    for suffix, by_backend in sorted(groups.items()):
        if len(by_backend) < args.min_backends:
            skipped += 1
            continue

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for backend in BACKENDS:
            r = by_backend.get(backend)
            if not r:
                continue
            df = load_power_df(r.csv_path)
            axes[0].plot(df["inference_id"], df["P_gpu_W"], marker="o", linewidth=1.6, label=backend)
            axes[1].plot(df["inference_id"], df["P_vin_W"], marker="o", linewidth=1.6, label=backend)

        axes[0].set_ylabel("GPU avg power (W) = E_gpu_J / duration_s")
        axes[1].set_ylabel("VIN avg power (W) = E_vin_J / duration_s")
        axes[1].set_xlabel("inference_id")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        fig.suptitle(f"Per-inference avg power (exclude 1st) — {suffix}")
        plt.tight_layout()

        out_path = out_dir / f"compare_inference_power__{suffix}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        written += 1

    print(f"[OK] result_dir={result_dir}")
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] groups_total={len(groups)} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()

