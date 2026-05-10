# Final Switching Overhead Result

## Main Result

기존 heatmap 기반 `stable_ms`보다 새 breakdown re-measurement의 `usable_stable_ms`가 더 길게 나오지 않았다. 동일하게 fixed-frequency transition에서 diagonal/default를 제외하고 비교했다.

| Device | Previous heatmap p50 | Breakdown p50 | Previous p90 | Breakdown p90 |
|---|---:|---:|---:|---:|
| CPU | 19.5 ms | 8.1 ms | 24.3 ms | 12.5 ms |
| GPU | 145.1 ms | 106.7 ms | 165.5 ms | 109.6 ms |
| EMC | 65.7 ms | 25.8 ms | 141.2 ms | 28.7 ms |

## Breakdown Interpretation

새 측정은 `usable_stable_ms = write_total_ms + post_write_to_first_match_ms + stable_extra_ms`로 해석할 수 있다. p50 기준 구성은 다음과 같다.

| Device | Write path | Until first observed | Stability confirmation | Total usable stable |
|---|---:|---:|---:|---:|
| CPU | 1.1 ms | 4.5 ms | 2.8 ms | 8.1 ms |
| GPU | 89.6 ms | 8.3 ms | 8.6 ms | 106.7 ms |
| EMC | 17.4 ms | 3.9 ms | 4.5 ms | 25.8 ms |

## Paper Text Draft

We re-measured the userspace manual DVFS switching overhead with a finer breakdown of the control path. Unlike the previous heatmap-only measurement, the new experiment separates the time spent in sysfs/debugfs writes, the time until the requested frequency is first observed, and the additional time required for stable confirmation. Under the same fixed-frequency, non-diagonal transition setting, the re-measured overhead was not larger than the previous heatmap estimate: CPU decreased from 19.5 ms to 8.1 ms p50, GPU from 145.1 ms to 106.7 ms p50, and EMC from 65.7 ms to 25.8 ms p50. This indicates that the previous heatmap provided a conservative estimate of practical userspace switching overhead.

For phase-level scheduling, the breakdown is more informative than the aggregate heatmap. CPU transitions became usable within about 8.1 ms p50, and EMC within about 25.8 ms p50. GPU switching remained much larger at about 106.7 ms p50, dominated by the userspace BPMP/debugfs rate-write path. Therefore, CPU and EMC can be considered for coarse phase- or inference-level DVFS decisions, while GPU frequency changes through the userspace debugfs path should be treated as inference-level or long-phase decisions rather than short per-phase actions.

## Figures

- `final_old_heatmap_vs_breakdown.png`: main comparison figure.
- `final_breakdown_components.png`: explains where the re-measured overhead comes from.
