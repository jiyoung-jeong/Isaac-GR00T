#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/Thor/regenerated_artifacts_20260508")
OUT = ROOT / "gr00t_n1d6_ncu_roofline_from_full128_reps"

REPORTS = [
    (
        "Action head TensorRT",
        ROOT / "gr00t_n1d6_ncu_governor_profiler_range/action_head_tensorrt_profiler_range_governor_raw.csv",
        ROOT / "gr00t_n1d6_ncu_governor_profiler_range/action_head_tensorrt_profiler_range_governor_details.txt",
    ),
    (
        "ViT full128",
        ROOT / "gr00t_n1d6_ncu_governor_backbone_full/vit_full128_profiler_range_governor_raw.csv",
        ROOT / "gr00t_n1d6_ncu_governor_backbone_full/vit_full128_profiler_range_governor_details.txt",
    ),
    (
        "LLM full128",
        ROOT / "gr00t_n1d6_ncu_governor_backbone_full/llm_full128_profiler_range_governor_raw.csv",
        ROOT / "gr00t_n1d6_ncu_governor_backbone_full/llm_full128_profiler_range_governor_details.txt",
    ),
]

TENSOR_PER_SECOND = [
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum.per_second",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum.per_second",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum.per_second",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum.per_second",
    "sm__ops_path_tensor_src_int8.sum.per_second",
    "sm__ops_path_tensor_src_fp4_dst_fp32.sum.per_second",
    "sm__ops_path_tensor_src_fp4_fp6_fp8_dst_fp16.sum.per_second",
    "sm__ops_path_tensor_src_fp4_fp6_fp8_dst_fp32.sum.per_second",
]

TENSOR_PEAK_PER_SECOND = [
    key.replace(".per_second", ".peak_sustained_elapsed.per_second")
    for key in TENSOR_PER_SECOND
]


def fnum(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except ValueError:
        return 0.0


def short_kernel(name):
    if name.startswith("sm80_xmma_gemm"):
        if "_tn_" in name:
            return "xmma_gemm_tn"
        if "_nn_" in name:
            return "xmma_gemm_nn"
        return "xmma_gemm"
    if name.startswith("__myl_Fc_"):
        return "trt_fc_tensorop"
    if name.startswith("__myl_Silu"):
        return "silu"
    if name.startswith("__myl_Tran"):
        return "transpose"
    if name.startswith("__myl_CastMean"):
        return "norm/cast"
    if name.startswith("__myl_MoveMove"):
        return "softmax-ish"
    if name.startswith("__myl_MoveRepl"):
        return "reshape/concat"
    if name.startswith("elementwise_kernel"):
        return "elementwise"
    if name.startswith("im2col_kernel"):
        return "im2col"
    if "upsample" in name:
        return "upsample"
    if name.startswith("reduce_kernel"):
        return "reduce"
    if name.startswith("nvjet"):
        return "nvjet_attention"
    return name[:32]


def read_ncu_csv(path):
    with path.open(newline="") as f:
        reader = csv.reader(f)
        cols = next(reader)
        units = next(reader)
        return [dict(zip(cols, row)) for row in reader], dict(zip(cols, units))


def build_points():
    points = []
    peak_compute = []
    l2_roofs = []
    l1_roofs = []

    for phase, raw_csv, _details in REPORTS:
        rows, _units = read_ncu_csv(raw_csv)
        for idx, row in enumerate(rows, 1):
            kernel = row.get("Kernel Name", "")
            if not kernel:
                continue

            tensor_ops_s = sum(fnum(row.get(k)) for k in TENSOR_PER_SECOND)
            active_tensor_peaks = [
                fnum(row.get(peak_key))
                for op_key, peak_key in zip(TENSOR_PER_SECOND, TENSOR_PEAK_PER_SECOND)
                if fnum(row.get(op_key)) > 0
            ]

            l2_bw_gbs = fnum(row.get("derived__lts__lts2xbar_bytes.sum.per_second"))
            l2_peak_kbyte = fnum(row.get("derived__lts__lts2xbar_bytes.sum.peak_sustained"))
            l2_freq_ghz = fnum(row.get("lts__cycles_elapsed.avg.per_second"))
            l1_peak_bytes = fnum(row.get("derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.peak_sustained"))
            l1_freq_ghz = fnum(row.get("l1tex__cycles_elapsed.avg.per_second"))

            if l2_peak_kbyte > 0 and l2_freq_ghz > 0:
                # NCU reports this roof as Kbyte/cycle and GHz.
                l2_roofs.append(l2_peak_kbyte * l2_freq_ghz * 1024.0)
            if l1_peak_bytes > 0 and l1_freq_ghz > 0:
                # byte/cycle * GHz == GB/s.
                l1_roofs.append(l1_peak_bytes * l1_freq_ghz)

            scalar_ops_per_cycle = (
                2.0 * fnum(row.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"))
                + fnum(row.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"))
                + fnum(row.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"))
            )
            smsp_freq_sum_ghz = fnum(row.get("smsp__cycles_elapsed.sum.per_second"))
            scalar_ops_s = scalar_ops_per_cycle * smsp_freq_sum_ghz * 1e9
            ops_s = tensor_ops_s + scalar_ops_s
            op_class = "tensor" if tensor_ops_s >= scalar_ops_s and tensor_ops_s > 0 else "scalar"
            scalar_peak_ops_s = (
                2.0
                * fnum(row.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"))
                * fnum(row.get("sm__cycles_elapsed.avg.per_second"))
                * 1e9
            )
            active_peak_ops_s = max(active_tensor_peaks) if active_tensor_peaks else scalar_peak_ops_s
            if active_peak_ops_s > 0:
                peak_compute.append(active_peak_ops_s / 1e12)

            if ops_s <= 0 or l2_bw_gbs <= 0:
                continue

            tops = ops_s / 1e12
            ai = ops_s / (l2_bw_gbs * 1e9)
            duration_ms = fnum(row.get("gpu__time_duration.sum"))

            points.append(
                {
                    "phase": phase,
                    "kernel_index": idx,
                    "kernel": kernel,
                    "label": short_kernel(kernel),
                    "duration_ms": duration_ms,
                    "l2_bandwidth_GBps": l2_bw_gbs,
                    "math_TOPS": tops,
                    "tensor_TOPS": tensor_ops_s / 1e12,
                    "scalar_TOPS": scalar_ops_s / 1e12,
                    "op_class": op_class,
                    "arithmetic_intensity_ops_per_byte_L2": ai,
                    "active_peak_TOPS": active_peak_ops_s / 1e12 if active_peak_ops_s else 0.0,
                }
            )

    def median(vals):
        vals = sorted(v for v in vals if v > 0)
        return vals[len(vals) // 2] if vals else 0.0

    return points, {
        "tensor_peak_TOPS": max(peak_compute) if peak_compute else 0.0,
        "l2_peak_GBps": median(l2_roofs),
        "l1_peak_GBps": median(l1_roofs),
    }


def write_points(points):
    path = OUT / "ncu_rep_math_l2_roofline_points.csv"
    cols = [
        "phase",
        "kernel_index",
        "kernel",
        "label",
        "duration_ms",
        "l2_bandwidth_GBps",
        "math_TOPS",
        "tensor_TOPS",
        "scalar_TOPS",
        "op_class",
        "arithmetic_intensity_ops_per_byte_L2",
        "active_peak_TOPS",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(points)
    return path


def plot(points, roofs):
    colors = {
        "Action head TensorRT": "#d55e00",
        "ViT full128": "#0072b2",
        "LLM full128": "#009e73",
    }

    xs = [p["arithmetic_intensity_ops_per_byte_L2"] for p in points]
    ys = [p["math_TOPS"] for p in points]
    x_min = max(1e-2, min(xs) / 3)
    x_max = max(xs) * 3
    y_min = max(1e-3, min(ys) / 5)
    y_max = max(max(ys) * 4, roofs["tensor_peak_TOPS"] * 1.35)

    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=180)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", color="#d9d9d9", linewidth=0.65, alpha=0.75)

    x_line = [10 ** (math.log10(x_min) + i * (math.log10(x_max / x_min) / 300)) for i in range(301)]
    tensor_peak = roofs["tensor_peak_TOPS"]
    l2_peak = roofs["l2_peak_GBps"]
    l1_peak = roofs["l1_peak_GBps"]

    if l2_peak > 0:
        y_l2 = [min(tensor_peak, x * l2_peak / 1000.0) for x in x_line]
        ax.plot(x_line, y_l2, color="#444444", linewidth=2.2, label=f"L2 roof ~{l2_peak:,.0f} GB/s")
    if l1_peak > 0:
        y_l1 = [min(tensor_peak, x * l1_peak / 1000.0) for x in x_line]
        ax.plot(x_line, y_l1, color="#777777", linewidth=1.8, linestyle="--", label=f"L1/TEX roof ~{l1_peak:,.0f} GB/s")
    if tensor_peak > 0:
        ax.axhline(tensor_peak, color="#222222", linewidth=2.4, linestyle="-.", label=f"Tensor peak ~{tensor_peak:,.0f} TOPS")

    top = sorted(points, key=lambda p: p["duration_ms"], reverse=True)[:8]
    top_index = {id(p): i + 1 for i, p in enumerate(top)}

    for phase in colors:
        group = [p for p in points if p["phase"] == phase]
        if not group:
            continue
        size = [max(28, min(340, 45 + p["duration_ms"] * 95)) for p in group]
        ax.scatter(
            [p["arithmetic_intensity_ops_per_byte_L2"] for p in group],
            [p["math_TOPS"] for p in group],
            s=size,
            color=colors[phase],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.75,
            label=f"{phase} ({len(group)} math kernels)",
        )
        for p in group:
            n = top_index.get(id(p))
            if n is None:
                continue
            ax.text(
                p["arithmetic_intensity_ops_per_byte_L2"],
                p["math_TOPS"],
                str(n),
                color="white",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

    top_lines = ["Top kernels by NCU duration"]
    for i, p in enumerate(top, 1):
        top_lines.append(
            f"{i}. {p['phase'].split()[0]} {p['label']}  {p['duration_ms']:.3f} ms, {p['math_TOPS']:.1f} TOPS"
        )
    ax.text(
        0.985,
        0.31,
        "\n".join(top_lines),
        transform=ax.transAxes,
        fontsize=8.0,
        color="#202020",
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.92},
    )

    ax.set_xlabel("Arithmetic intensity from NCU L2 traffic (math ops / byte)")
    ax.set_ylabel("Achieved math throughput (TOPS, Tensor Core + FP32 scalar)")
    ax.set_title("GR00T N1.6 NCU-Rep Roofline, Governor Clocks, Unified Memory Thor")
    ax.text(
        0.01,
        0.02,
        "Dots are kernels captured in the .ncu-rep files. ViT/LLM use 128-kernel profiler ranges.\n"
        "Unified memory SoC: L2/L1 roofs are shown; DRAM/EMC should be interpreted with tegrastats/microbench.",
        transform=ax.transAxes,
        fontsize=8.6,
        color="#4a4a4a",
        va="bottom",
    )
    ax.legend(loc="upper left", fontsize=8.3, frameon=True, framealpha=0.92)
    fig.tight_layout()

    png = OUT / "gr00t_n1d6_ncu_rep_math_l2_roofline.png"
    pdf = OUT / "gr00t_n1d6_ncu_rep_math_l2_roofline.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points, roofs = build_points()
    csv_path = write_points(points)
    png, pdf = plot(points, roofs)
    print(f"points={len(points)}")
    print(f"tensor_peak_TOPS={roofs['tensor_peak_TOPS']:.3f}")
    print(f"l2_peak_GBps={roofs['l2_peak_GBps']:.3f}")
    print(f"l1_peak_GBps={roofs['l1_peak_GBps']:.3f}")
    print(csv_path)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
