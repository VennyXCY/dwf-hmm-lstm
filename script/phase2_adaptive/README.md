# Phase 2 — Adaptive rolling recognition and prediction

Each update cycle runs as **three stages** (matching the paper), because the HMM
stage uses `hmmlearn` (env `dwf-hmm`) and the LSTM stages use TensorFlow
(env `dwf-ann`); the two cannot share one environment. Stages hand off through
files written to `results/HMM_LSTM_Model_<version>/`.

## Files

**Function libraries** (imported, not run directly):

| File | Role |
|------|------|
| `config.py` | Shared constants (features, `WINDOW_SIZE`, matching `alpha`, diagnostic thresholds), repo-relative paths, and the per-cycle `CYCLES` table. Set `ACTIVE_CYCLE` here. |
| `online_hmm.py` | HMM stage: load the training window, fit the local Gaussian HMM, extract hour-of-day distributions, the single-step online prior (`hmm_prior_step`), and build the LSTM training targets. |
| `state_matching.py` | Load the baseline archive; match local states to it with a water-quality/flow + time cost matrix (weight `alpha`) solved by the Hungarian algorithm; flag low-confidence matches as Evolved States; re-index the HMM to the archive order. |
| `lstm_model.py` | Build and train the teacher-student LSTM (two inputs: sensor window + HMM prior). |
| `diagnostics.py` | Predictive entropy, KL divergence to the prior, switching frequency, short-dwell ratio, state-frequency (vanishing) analysis, performance-history update, and the reconstruction-trigger decision. |
| `reconstruction.py` | The "architect": over a long window, report BIC, state occupancy and similarity, and save a new archive. The number of states is chosen with a manual override (`N_OVERRIDE`), not by BIC alone. |

**Stage entry scripts** (run directly):

| File | Environment | Stage |
|------|-------------|-------|
| `part1_hmm_matching.py` | `dwf-hmm` | Fit local HMM, match to archive, standardize, generate LSTM training data. |
| `part2_lstm_training.py` | `dwf-ann` | Train the LSTM. |
| `part3_rolling_inference.py` | `dwf-ann` | Autoregressive rolling inference + diagnostics + reconstruction decision. |

## Running one update cycle

Prerequisite: run phase 1 first (`build_baseline_archive.py`), which
**generates** the initial baseline archive at
`results/merged_state_features_physical_and_time.csv` (you do not supply this
file). Also place the cleaned sensor file for this cycle in `data/` (same
cleaning as phase 1; see `data/README.md`).

1. **Edit `config.py`**: set `ACTIVE_CYCLE`, and fill/adjust its `CYCLES` entry
   (`n_states`, `archive_csv`, `data_csv`, the training window `train_*`, the
   forecast horizon `infer_*`, `reconstruction_enabled`). The window lengths are
   whatever date ranges you set here and are fully configurable; the paper uses a
   9-day training window (W), a 3-day step (S) and a 3-day horizon (H), but any
   lengths work (the paper also reports a 3-day vs 5-day sensitivity analysis).
2. **`dwf-hmm`**: `python script/phase2_adaptive/part1_hmm_matching.py`
   → writes `ann_training_data/*.npy`, `transmat.npy`, `scaler.joblib`,
   `evolved_map.joblib`.
3. **`dwf-ann`**: `python script/phase2_adaptive/part2_lstm_training.py`
   → writes `lstm_model.h5`.
4. **`dwf-ann`**: `python script/phase2_adaptive/part3_rolling_inference.py`
   → writes `inference_results_<version>.csv`, updates
   `results/performance_history.csv`, prints the diagnostics and whether
   reconstruction is triggered.
5. **If reconstruction is triggered** (and enabled for this cycle): in `dwf-hmm`,
   set the analysis window in `reconstruction.py` and run it once to inspect the
   BIC suggestion, state occupancy and similarity. Then choose the state count by
   judgement, set `N_OVERRIDE` to it, and run it again to write the new archive.
   Set `RECON_TAG` (e.g. `"V10_N5"`) to label this reconstruction; the archive is
   written to `results/merged_state_features_<RECON_TAG>.csv`, so multiple
   reconstructions do not overwrite each other. Point the next cycle's
   `archive_csv` at the new archive and set `n_states` accordingly. Cycles that do
   **not** trigger reconstruction skip this step entirely and keep the current
   archive.

Advance to the next cycle by sliding the windows forward by one step (the paper
uses S = 3 days; set the next cycle's dates accordingly) and repeating.
Reconstruction is human-in-the-loop by design (the state count is
chosen with judgement), so cycles are run one at a time rather than in an
unattended loop.

## Applying the framework to other data

1. Put your cleaned sensor record in `data/` with the schema
   `Datetime, Temp, Cond, pH, NH3, Flow, Turb` (consistent units; Flow in m3/h).
   Clean it with the same method as phase 1 if needed.
2. Build an initial archive on your data by running phase 1
   (`build_baseline_archive.py`), which produces
   `results/merged_state_features_physical_and_time.csv`.
3. In `config.py`, adjust `FEATURE_COLS` (if your sensors differ), `WINDOW_SIZE`,
   the diagnostic thresholds for your site, and add `CYCLES` entries with your
   own dates and `n_states`.
4. Run the three stages per cycle as above; reconstruct when triggered.

## Notes

- Diagnostic thresholds in `config.py` are **reference** values; the final values
  used in the paper take precedence.
- `switching_frequency` and `short_dwell_ratio` are provided as reference implementations 
  of the stability indicators reported in the paper; they are printed for information and 
  do not drive the reconstruction trigger. Their exact definitions follow the paper.
- The LSTM is saved as `lstm_model.h5` (HDF5), compatible with the pinned
  TensorFlow 2.10. On newer TensorFlow you may switch to the `.keras` format.
