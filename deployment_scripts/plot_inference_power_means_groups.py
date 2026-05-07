#!/usr/bin/env python3
"""
Create bar plots of mean per-inference average power (exclude 1st inference),
grouped by condition suffix, with one bar per backend.

Input: result/thor_gr00t_server_*/inference_energy.csv
Per inference i (i>=2):
  P_gpu_W = E_gpu_J / (duration_ms/1000)
  P_vin_W = E_vin_J / (duration_ms/1000)
We then summarize each run to mean (and optional error bars).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def per_inference_power(df: pd.DataFrame) -> pd.DataFrame:
    required = {"inference_id", "duration_ms", "E_gpu_J", "E_vin_J"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df[df["inference_id"] >= 2].copy()
    if len(df) == 0:
        raise ValueError("no inferences after excluding 1st")
    t_s = df["duration_ms"].astype(float) / 1e3
    df["P_gpu_W"] = df["E_gpu_J"].astype(float) / t_s
    df["P_vin_W"] = df["E_vin_J"].astype(float) / t_s
    return df


def summarize(vals: np.ndarray, mode: str) -> Tuple[float, float, float]:
    """
    Return (center, err_low, err_high).
    mode:
      - none: err_low=err_high=0
      - std: +/- 1 std
      - iqr: p25..p75 around median (center=median)
      - minmax: min..max around mean (center=mean)
    """
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0.0, 0.0
    if mode == "none":
        return float(np.mean(vals)), 0.0, 0.0
    if mode == "std":
        c = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
        return c, s, s
    if mode == "iqr":
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        return float(q50), float(q50 - q25), float(q75 - q50)
    if mode == "minmax":
        c = float(np.mean(vals))
        return c, float(c - np.min(vals)), float(np.max(vals) - c)
    raise ValueError(f"unknown errorbar mode: {mode}")


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main():
    ap = argparse.ArgumentParser(description="Plot mean power bars per backend grouped by suffix (exclude 1st inference).")
    ap.add_argument(
        "--result-dir",
        default="/home/Thor/Workspace/jyjeong/Isaac-GR00T/result",
        help="Directory containing thor_gr00t_server_* run dirs",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/Thor/compare_plots_means",
        help="Output directory (must be writable)",
    )
    ap.add_argument(
        "--only-suffix-regex",
        default="",
        help="If set, only plot suffixes matching this regex.",
    )
    ap.add_argument(
        "--min-backends",
        type=int,
        default=2,
        help="Only plot groups that have at least this many backends present (default 2).",
    )
    ap.add_argument(
        "--errorbars",
        choices=["none", "std", "iqr", "minmax"],
        default="iqr",
        help="Error bars for per-inference power distribution within a run (default iqr).",
    )
    ap.add_argument(
        "--write-summary-csv",
        action="store_true",
        help="Also write a summary CSV with per-group mean power per backend.",
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

    # group by suffix
    groups: Dict[str, Dict[str, RunInfo]] = {}
    for r in runs:
        groups.setdefault(r.suffix, {})[r.backend] = r  # last wins if duplicate

    summary_rows: List[dict] = []

    written = 0
    skipped = 0
    for suffix, by_backend in sorted(groups.items()):
        present = [b for b in BACKENDS if b in by_backend]
        if len(present) < args.min_backends:
            skipped += 1
            continue

        # compute per-backend stats
        stats: Dict[str, Dict[str, float]] = {}
        for backend in BACKENDS:
            r = by_backend.get(backend)
            if not r:
                stats[backend] = {"gpu_c": float("nan"), "gpu_lo": 0.0, "gpu_hi": 0.0, "vin_c": float("nan"), "vin_lo": 0.0, "vin_hi": 0.0}
                continue
            df = pd.read_csv(r.csv_path)
            p = per_inference_power(df)
            gpu_c, gpu_lo, gpu_hi = summarize(p["P_gpu_W"].to_numpy(dtype=float), args.errorbars)
            vin_c, vin_lo, vin_hi = summarize(p["P_vin_W"].to_numpy(dtype=float), args.errorbars)
            stats[backend] = {"gpu_c": gpu_c, "gpu_lo": gpu_lo, "gpu_hi": gpu_hi, "vin_c": vin_c, "vin_lo": vin_lo, "vin_hi": vin_hi}

        # plot (two panels, grouped bars)
        x = np.arange(len(BACKENDS))
        gpu_vals = [stats[b]["gpu_c"] for b in BACKENDS]
        vin_vals = [stats[b]["vin_c"] for b in BACKENDS]
        gpu_yerr = np.array([[stats[b]["gpu_lo"] for b in BACKENDS], [stats[b]["gpu_hi"] for b in BACKENDS]])
        vin_yerr = np.array([[stats[b]["vin_lo"] for b in BACKENDS], [stats[b]["vin_hi"] for b in BACKENDS]])

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].bar(x, np.nan_to_num(gpu_vals, nan=0.0), color=["C0", "C1", "C2"], alpha=0.9)
        axes[1].bar(x, np.nan_to_num(vin_vals, nan=0.0), color=["C0", "C1", "C2"], alpha=0.9)

        if args.errorbars != "none":
            # Only draw error bars where center is finite
            gpu_mask = np.isfinite(gpu_vals)
            vin_mask = np.isfinite(vin_vals)
            axes[0].errorbar(x[gpu_mask], np.array(gpu_vals)[gpu_mask], yerr=gpu_yerr[:, gpu_mask], fmt="none", ecolor="k", elinewidth=1.1, capsize=4, alpha=0.8)
            axes[1].errorbar(x[vin_mask], np.array(vin_vals)[vin_mask], yerr=vin_yerr[:, vin_mask], fmt="none", ecolor="k", elinewidth=1.1, capsize=4, alpha=0.8)

        axes[0].set_ylabel("GPU power (W)\nmean of P_gpu (i>=2)")
        axes[1].set_ylabel("VIN power (W)\nmean of P_vin (i>=2)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(BACKENDS, rotation=0)
        axes[0].grid(True, alpha=0.25, axis="y")
        axes[1].grid(True, alpha=0.25, axis="y")
        fig.suptitle(f"Mean per-inference avg power (exclude 1st) — {suffix}\nerrorbars={args.errorbars}")
        plt.tight_layout()

        out_path = out_dir / f"mean_power__{safe_filename(suffix)}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        written += 1

        summary_rows.append(
            {
                "suffix": suffix,
                **{f"{b}_P_gpu_W": stats[b]["gpu_c"] for b in BACKENDS},
                **{f"{b}_P_vin_W": stats[b]["vin_c"] for b in BACKENDS},
            }
        )

    if args.write_summary_csv and summary_rows:
        out_csv = out_dir / "mean_power_summary.csv"
        pd.DataFrame(summary_rows).sort_values("suffix").to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv}")

    print(f"[OK] result_dir={result_dir}")
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] groups_total={len(groups)} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()

