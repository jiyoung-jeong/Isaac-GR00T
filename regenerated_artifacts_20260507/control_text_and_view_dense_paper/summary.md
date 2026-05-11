# Dense Control Scenario Analysis

- figure script: `make_annotated_dense_figures.py`
- annotated figure outputs:
  - `text_only_main_figure_annotated_dense.png`
  - `text_only_main_figure_annotated_dense.pdf`
  - `viewcount_main_figure_annotated_dense.png`
  - `viewcount_main_figure_annotated_dense.pdf`
- measurement scripts:
  - `source_scripts/run_control_reduced_combo_sweep.sh`
  - `source_scripts/run_control_text_and_view_sweeps.sh`
  - `source_scripts/benchmark_input_sweep.py`

## Text-only (both_views)

### best_latency
- mean latency: 93.51 +- 0.68 ms
- mean VIN energy: 42.84 +- 0.32 J
- mean latency gain vs all-default: 16.04%
- mean energy gain vs all-default: 12.26%
- most common winner: CPU 2.601GHz, GPU 1.575GHz, EMC 4.266GHz

### best_energy
- mean latency: 93.68 +- 0.58 ms
- mean VIN energy: 42.83 +- 0.33 J
- mean latency gain vs all-default: 15.84%
- mean energy gain vs all-default: 12.28%
- most common winner: CPU 2.601GHz, GPU 1.575GHz, EMC 4.266GHz

### best_tradeoff
- mean latency: 93.51 +- 0.68 ms
- mean VIN energy: 42.84 +- 0.32 J
- mean latency gain vs all-default: 16.04%
- mean energy gain vs all-default: 12.26%
- most common winner: CPU 2.601GHz, GPU 1.575GHz, EMC 4.266GHz

## View-count-only (64 words)

### best_latency
- mean latency: 86.89 +- 0.46 ms
- mean VIN energy: 40.16 +- 0.39 J
- mean latency gain vs all-default: 23.12%
- mean energy gain vs all-default: 15.50%
- most common winner: CPU 2.601GHz, GPU 1.575GHz, EMC 4.266GHz

### best_energy
- mean latency: 87.46 +- 0.66 ms
- mean VIN energy: 40.04 +- 0.36 J
- mean latency gain vs all-default: 22.62%
- mean energy gain vs all-default: 15.75%
- most common winner: CPU 2.376GHz, GPU 1.575GHz, EMC 4.266GHz

### best_tradeoff
- mean latency: 86.89 +- 0.48 ms
- mean VIN energy: 40.12 +- 0.41 J
- mean latency gain vs all-default: 23.12%
- mean energy gain vs all-default: 15.59%
- most common winner: CPU 2.601GHz, GPU 1.575GHz, EMC 4.266GHz
