# LIBERO-Spatial Pareto (All-default removed)

- source root: `/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/combo_sweep_vla_auto_20260427_091937`
- removed run: `mode == default` only
- remaining rows: `314`
- pareto rows: `7`
- figure outputs: `pareto_latency_energy.png`, `pareto_latency_energy.pdf`
- figure script: `make_pareto_latency_energy.py`
- measurement scripts: `source_scripts/thor_combofreq_remote_client_sweep.sh`, `source_scripts/thor_combofreq_power_sweep.py`

## Best latency
- combo: `cpufreq_2.601GHz_gpufreq_default_emcfreq_default`
- label:

```text
Best latency
CPU 2.601GHz
GPU 1.575GHz
EMC 3.2GHz
```

- latency: `65.76 ms`
- energy: `3984.34 J`

## Best energy
- combo: `cpufreq_2.430GHz_gpufreq_1.107GHz_emcfreq_3.200GHz`
- label:

```text
Best energy
CPU 2.43GHz
GPU 1.107GHz
EMC 3.2GHz
```

- latency: `79.78 ms`
- energy: `2558.14 J`

## Best trade-off
- combo: `cpufreq_2.430GHz_gpufreq_1.503GHz_emcfreq_default`
- label:

```text
Best trade-off
CPU 2.43GHz
GPU 1.503GHz
EMC 3.2GHz
```

- latency: `66.91 ms`
- energy: `2649.60 J`
