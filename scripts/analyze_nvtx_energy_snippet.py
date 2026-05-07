# ========== analyze_and_plot() heredoc 안에 삽입할 코드 ==========
# 위치: txt.append(stats("BENCH (BENCH_START~BENCH_END)", tmin=bs, tmax=be)) 다음,
#       (outdir/"summary.txt").write_text(...) 직전

# NVTX 구간이 있으면 inference별 에너지 적분 (E = sum(P*dt))
nvtx_path = outdir / "nvtx_ranges.csv"
if nvtx_path.exists():
    nv = pd.read_csv(nvtx_path, header=None, names=["ts_ns", "event"])
    nv = nv[nv["event"].isin(["POLICY_INFER_START", "POLICY_INFER_END"])].sort_values("ts_ns")
    starts = nv[nv["event"] == "POLICY_INFER_START"]["ts_ns"].values
    ends = nv[nv["event"] == "POLICY_INFER_END"]["ts_ns"].values
    if len(starts) == len(ends) and len(starts) > 0:
        energies = []
        for i, (s_ns, e_ns) in enumerate(zip(starts, ends)):
            seg = df[(df["ts_ns"] >= s_ns) & (df["ts_ns"] <= e_ns)].sort_values("ts_ns")
            if len(seg) < 2:
                continue
            ts = seg["ts_ns"].values.astype(np.float64)
            dt = np.diff(ts) * 1e-9  # seconds
            E_gpu = np.sum(seg["vdd_gpu_W"].values[:-1] * dt)
            E_cpu = np.sum(seg["vdd_cpu_soc_mss_W"].values[:-1] * dt)
            E_vin = np.sum(seg["vin_W"].values[:-1] * dt)
            dur_ms = (e_ns - s_ns) * 1e-6
            energies.append({"inference_id": i + 1, "duration_ms": dur_ms, "E_gpu_J": E_gpu, "E_cpu_J": E_cpu, "E_vin_J": E_vin})
        if energies:
            en_df = pd.DataFrame(energies)
            en_df.to_csv(outdir / "inference_energy.csv", index=False)
            txt.append("INFERENCE (per-call energy from NVTX ranges):")
            txt.append(f"  N={len(energies)} calls, E_gpu_J sum={en_df['E_gpu_J'].sum():.3f}, mean={en_df['E_gpu_J'].mean():.3f}")
            txt.append(f"  E_vin_J sum={en_df['E_vin_J'].sum():.3f}, mean={en_df['E_vin_J'].mean():.3f}")
