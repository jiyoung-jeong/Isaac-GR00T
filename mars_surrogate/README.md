# MARS Surrogate OPP Selector

This package trains measured-data surrogate models for deadline-aware VLA OPP selection:

- `(workload metadata, candidate OPP) -> e2e_median_ms`
- `(workload metadata, candidate OPP) -> e2e_max_ms`
- `(workload metadata, candidate OPP) -> vin_energy_j_per_timed_iteration`
- `(workload metadata, candidate OPP, deadline_ms) -> P(strict_feasible)`

At runtime, the Eco selector evaluates every candidate OPP with the surrogates, applies a deadline constraint, and chooses the lowest predicted-energy feasible candidate. It does not run VLA inference and it does not learn `metadata -> best OPP` labels.

`fixed_period_ms` is a deadline constraint, not a latency label. It is never used as a regression target. The default median latency target is `e2e_median_ms`; the tail latency target is `e2e_max_ms`; `fixed_period_elapsed_median_ms` is not used because it includes fixed-period loop behavior. The default energy target is per-period VIN energy, `vin_energy_j_per_timed_iteration`, not total `vin_energy_j` across timed iterations.

Median latency is useful for energy-efficient average-case selection, but it is not sufficient for strict fixed-period deadline guarantees. A row can have `e2e_median_ms <= fixed_period_ms` and still miss deadlines because tail latency or loop jitter exceeds the period. The tail model predicts `e2e_max_ms`, and the strict-feasibility classifier predicts deadline-miss-free execution, `fixed_period_deadline_miss_pct == 0`.

`fixed_period_ms` is only used as:

- a selector constraint: `predicted_latency + safety_margin <= fixed_period_ms`
- an input to the strict-feasibility classifier, because feasibility is defined relative to a deadline

`fixed_period_deadline_miss_pct` is used only to construct classifier labels and evaluate decisions. It is not an input feature.

## Train

```bash
python scripts/make_candidate_set.py \
  --csv data/summary.csv \
  --out artifacts/candidates.csv

python scripts/train_surrogate.py \
  --csv data/summary.csv \
  --out artifacts/surrogate_auto \
  --model auto \
  --split random_rows \
  --train-fraction 0.25 \
  --latency-target e2e_median_ms \
  --tail-target e2e_max_ms \
  --energy-target vin_energy_j_per_timed_iteration
```

Training saves `model.joblib`, `feasibility_model.joblib`, `metrics.json`, `feasibility_metrics.json`, `predictions_test.csv`, and `calibrated_margin.json`. Test predictions include `pred_median_latency_ms`, `pred_tail_latency_ms`, `pred_energy_j`, and `pred_strict_feasible_prob`.

## Evaluate

```bash
python scripts/evaluate_selector.py \
  --csv data/summary.csv \
  --model-dir artifacts/surrogate_auto \
  --candidate-csv artifacts/candidates.csv \
  --out-dir artifacts/eval \
  --feasibility strict \
  --safety-margin-ms 0
```

Evaluation writes an oracle ceiling report, selector variant comparison, and plots:

- `oracle_feasibility_by_group.csv`
- `oracle_feasibility_summary.json`
- `selector_variant_comparison.csv`
- `selector_variant_comparison.json`
- `margin_sweep_success_energy.png`
- `oracle_vs_selector_energy.png`
- `tail_pred_vs_actual.png`
- `feasibility_confusion_matrix.png`

## Runtime Selection

```bash
python scripts/select_opp.py \
  --model-dir artifacts/surrogate_auto \
  --candidate-csv artifacts/candidates.csv \
  --text-length-target 64 \
  --num-views 2 \
  --denoising-steps 4 \
  --deadline-ms 110
```

Selector modes:

- `median_margin`: feasible if `pred_median_latency_ms + safety_margin_ms <= deadline_ms`
- `tail`: feasible if `pred_tail_latency_ms + safety_margin_ms <= deadline_ms`
- `risk`: feasible if tail mode is feasible and `pred_strict_feasible_prob >= threshold`
