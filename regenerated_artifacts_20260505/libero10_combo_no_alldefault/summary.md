# LIBERO-10 Pareto (All-default removed)

- source root: `/home/Thor/Workspace/jyjeong/Isaac-GR00T/thor_measurements/libero10_combo_trt_20260430_090849`
- removed run: `mode == default` only
- remaining rows: `314`
- pareto rows: `6`
- figure outputs: `pareto_latency_energy.png`, `pareto_latency_energy.pdf`
- figure script: `make_pareto_latency_energy.py`
- measurement scripts: `source_scripts/thor_combofreq_remote_client_sweep.sh`, `source_scripts/thor_combofreq_power_sweep.py`

## Best latency
- combo: `cpufreq_2.601GHz_gpufreq_default_emcfreq_4.266GHz`
- label:

```text
Best latency
CPU 2.601GHz
GPU 1.575GHz
EMC 4.266GHz
```

- latency: `62.31 ms`
- energy: `2774.25 J`

## Best energy
- combo: `cpufreq_2.430GHz_gpufreq_1.305GHz_emcfreq_2.750GHz`
- label:

```text
Best energy
CPU 2.43GHz
GPU 1.305GHz
EMC 2.75GHz
```

- latency: `84.80 ms`
- energy: `2577.07 J`

## Best trade-off
- combo: `cpufreq_2.430GHz_gpufreq_default_emcfreq_4.266GHz`
- label:

```text
Best trade-off
CPU 2.43GHz
GPU 1.575GHz
EMC 4.266GHz
```

- latency: `62.96 ms`
- energy: `2605.83 J`
