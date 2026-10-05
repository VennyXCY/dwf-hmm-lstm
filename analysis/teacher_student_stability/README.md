# Teacher–student stability

Evidence for the stability advantage of the teacher-student paradigm: compares
the state-switching stability of the standalone online HMM, the standalone LSTM, 
and the proposed HMM-LSTM across the Phase 2 update cycles.

## Scripts (run in order)

1. `compute_switching_hmm.py`   (env: **dwf-hmm**)
   Re-fits the local HMM per cycle, runs single-step online filtering over the
   forecast horizon, and reads the HMM-LSTM sequence from the saved phase-2
   inference results. → `results/switching_hmm_vs_hmmlstm.csv`
2. `compute_switching_lstm.py`  (env: **dwf-ann**)
   Trains the standalone LSTM per cycle and computes its switching
   rate. Merges everything → `results/switching_standalone_lstm.csv`,
   `results/switching_all.csv`
3. `plot_teacher_student_stability.py`  (env: any)
   Reads `switching_all.csv` and draws the switching-rate comparison (shaded
   columns = drift cycles). → `results/teacher_student_stability.{png,pdf}`

## Inputs

- Per-cycle phase-2 outputs under `results/HMM_LSTM_Model_<V>/`
  (`inference_results_*.csv`, `scaler.joblib`, `ann_training_data/`).
- Cycle metadata (dates, `n_states`, data file) read from the phase-2 config
  (`script/phase2_adaptive/config.py`, the `CYCLES` table). Fill `CYCLES` with
  all cycles before running.
- The cleaned sensor file(s) referenced by `CYCLES` must be present in `data/`.

## Notes

- Requires the full Phase 2 pipeline to have been run for the cycles analysed.
- Drift cycles (default `V10, V14, V17`) are the cycles with a distributional
  shift, highlighted in the figure; set them via `DRIFT` in
  `plot_teacher_student_stability.py`.

