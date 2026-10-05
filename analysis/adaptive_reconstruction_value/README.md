# The value of adaptive reconstruction

Compares the adaptive framework against a static (frozen, non-adaptive) baseline
over the Phase 2 rolling period, to show the value of adaptive reconstruction.

## Scripts (run in order)

1. `compute_static_metrics.py`    (env: **dwf-ann**)
   Applies the frozen V3 model to every cycle's forecast horizon and records the
   static baseline's entropy / KLD / switching. → `results/static_history.csv`
2. `compute_matching_distance.py` (env: **dwf-hmm**)
   Measures the archive-to-data matching distance for the
   static frozen archive vs the adaptive archive in force each cycle.
   → `results/archive_data_distance.csv`
3. `plot_adaptive_reconstruction_value.py` (env: any)
   Draws the static-vs-adaptive comparison (distance, entropy, KLD).
   → `results/adaptive_reconstruction_value.{png,pdf}`

## Inputs

- Frozen baseline model: the first fully-trained cycle (V3 in this study; set via
  `STATIC_VERSION`), under `results/HMM_LSTM_Model_<STATIC_VERSION>/`
  (`lstm_model.h5`, `transmat.npy`, `scaler.joblib`).
- Static archive: `results/merged_state_features_physical_and_time.csv`.
- Adaptive run history: `results/performance_history.csv` (provides the adaptive
  entropy/KLD and the reconstruction-triggered cycles).
- Cycle metadata (dates, archive per cycle, `n_states`) from the phase-2 config
  (`CYCLES`). Fill `CYCLES` with all cycles (V3-V24) before running.

## Notes

- Requires the full Phase 2 pipeline to have been run (per-cycle models +
  `performance_history.csv`, enriched with `Reconstruction_Triggered`).
- Reconstruction cycles are read automatically from `performance_history.csv`.
