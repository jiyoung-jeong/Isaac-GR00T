#!/usr/bin/env python3
"""
Thor 전력/주파수 telemetry 로거. bash 루프보다 가벼워 5~10ms 실제 주기로 샘플링 가능.
환경변수로 출력 경로·주기·sysfs 경로 지정. SIGTERM 받을 때까지 무한 루프.
"""
import os
import signal
import time

# 환경변수 (thor_server_measure_and_analyze.sh와 동일 기본값)
RAW_CSV = os.environ.get("THOR_RAW_CSV", "")
INTERVAL_MS = float(os.environ.get("THOR_INTERVAL_MS", "5"))
GPU_V = os.environ.get("THOR_GPU_V", "/sys/bus/i2c/devices/2-0040/hwmon/hwmon6/in1_input")
GPU_I = os.environ.get("THOR_GPU_I", "/sys/bus/i2c/devices/2-0040/hwmon/hwmon6/curr1_input")
CPU_V = os.environ.get("THOR_CPU_V", "/sys/bus/i2c/devices/2-0040/hwmon/hwmon6/in2_input")
CPU_I = os.environ.get("THOR_CPU_I", "/sys/bus/i2c/devices/2-0040/hwmon/hwmon6/curr2_input")
VIN_PWR_UW = os.environ.get("THOR_VIN_PWR_UW", "/sys/bus/i2c/devices/2-0044/hwmon/hwmon5/power1_input")
GPUF = os.environ.get("THOR_GPUF", "/sys/class/devfreq/gpu-gpc-0/cur_freq")
CPU_POLICY_IDS = [0, 2, 4, 6, 8, 10, 12]
CPU_POLICY_PATHS = {
    policy_id: os.environ.get(
        f"THOR_CPU_POLICY{policy_id}",
        f"/sys/devices/system/cpu/cpufreq/policy{policy_id}/cpuinfo_cur_freq",
    )
    for policy_id in CPU_POLICY_IDS
}
GPU_GPC0_RATE = os.environ.get("THOR_GPU_GPC0_RATE", "/sys/kernel/debug/bpmp/debug/clk/gpu_gpc0/rate")
GPU_GPC1_RATE = os.environ.get("THOR_GPU_GPC1_RATE", "/sys/kernel/debug/bpmp/debug/clk/gpu_gpc1/rate")
GPU_GPC2_RATE = os.environ.get("THOR_GPU_GPC2_RATE", "/sys/kernel/debug/bpmp/debug/clk/gpu_gpc2/rate")
GPU_SYS_RATE = os.environ.get("THOR_GPU_SYS_RATE", "/sys/kernel/debug/bpmp/debug/clk/gpu_sys/rate")
GPU_NVD_RATE = os.environ.get("THOR_GPU_NVD_RATE", "/sys/kernel/debug/bpmp/debug/clk/gpu_nvd/rate")
EMC_RATE = os.environ.get("THOR_EMC_RATE", "/sys/kernel/debug/bpmp/debug/clk/emc/rate")

RUN = True


def _read(path: str, default: str = "0") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, IOError):
        return default


def _sample():
    ts_ns = time.monotonic_ns()
    vg = float(_read(GPU_V, "0"))
    ig = float(_read(GPU_I, "0"))
    vc = float(_read(CPU_V, "0"))
    ic = float(_read(CPU_I, "0"))
    pg = (vg * ig) / 1e6
    pc = (vc * ic) / 1e6
    vin_uw = float(_read(VIN_PWR_UW, "0"))
    vin_w = vin_uw / 1e6
    def _int(s: str, default: int = -1) -> int:
        try:
            return int(s)
        except (ValueError, TypeError):
            return default
    gf = _int(_read(GPUF, "-1"))
    cpu_policy_freqs = [_int(_read(CPU_POLICY_PATHS[policy_id], "-1")) for policy_id in CPU_POLICY_IDS]
    gpc0 = _int(_read(GPU_GPC0_RATE, "-1"))
    gpc1 = _int(_read(GPU_GPC1_RATE, "-1"))
    gpc2 = _int(_read(GPU_GPC2_RATE, "-1"))
    gsys = _int(_read(GPU_SYS_RATE, "-1"))
    gnvd = _int(_read(GPU_NVD_RATE, "-1"))
    emc = _int(_read(EMC_RATE, "-1"))
    return (ts_ns, pg, pc, vin_w, gf, *cpu_policy_freqs, gpc0, gpc1, gpc2, gsys, gnvd, emc)


def _stop(*_):
    global RUN
    RUN = False


def main():
    global RUN
    if not RAW_CSV:
        raise SystemExit("THOR_RAW_CSV not set")
    interval_sec = INTERVAL_MS / 1000.0
    signal.signal(signal.SIGTERM, _stop)
    # 셸이 이미 헤더를 썼으므로 append
    with open(RAW_CSV, "a") as f:
        while RUN:
            row = _sample()
            head = [str(row[0]), f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"]
            tail = [str(v) for v in row[4:]]
            f.write(",".join(head + tail) + "\n")
            f.flush()
            time.sleep(interval_sec)


if __name__ == "__main__":
    main()
