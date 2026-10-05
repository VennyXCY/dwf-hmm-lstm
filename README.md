# Adaptive HMM–LSTM framework for dry weather flow

Source code for an adaptive HMM–LSTM framework that recognizes and predicts
**dry weather flow (DWF) states** in a municipal separate storm sewer system
(MS4), using continuous, routinely measured water-quality and flow sensor data.

The framework has two phases:

- **Phase 1 – baseline archive.** A Gaussian HMM is fitted per data segment and
  the states are merged (agglomerative clustering) into a small set of physically
  interpretable **baseline states** (the archive).
- **Phase 2 – adaptive rolling recognition and prediction.** On a rolling window,
  a local HMM is matched to the archive and distilled into a teacher–student
  **HMM–LSTM** that predicts the DWF state. Online diagnostics (predictive
  entropy, KL divergence, state switching/vanishing) drive **archive
  reconstruction** when conditions drift.

A related article is in preparation; its citation will be added here on
publication (see `CITATION.cff`).

## Repository layout

```
dwf-hmm-lstm/
├── README.md                     ← this file
├── LICENSE                       ← MIT
├── CITATION.cff                  ← how to cite this software
├── requirements-hmm.txt          ← dependencies for the dwf-hmm-pip environment
├── requirements-ann.txt          ← dependencies for the dwf-ann environment
├── data/                         ← data provenance + a synthetic sample generator
│   ├── README.md
│   └── make_synthetic_sample.py
├── script/
│   ├── phase1_baseline/
│   │   └── build_baseline_archive.py        ← Phase 1 (env: dwf-hmm)
│   └── phase2_adaptive/                      ← Phase 2 (see its README)
│       ├── README.md
│       ├── config.py                         ← shared config + per-cycle CYCLES table
│       ├── online_hmm.py, state_matching.py, lstm_model.py,
│       ├── diagnostics.py, reconstruction.py
│       └── part1_hmm_matching.py, part2_lstm_training.py, part3_rolling_inference.py
├── figures/                      ← scripts that render the paper figures
│   ├── plot_posterior_segments.py, plot_elbow_dendrogram.py        (Phase 1)
│   ├── plot_baseline_radar_temporal.py, plot_baseline_violin_tpm.py (baseline states)
│   └── plot_state_matching.py, plot_state_recognition.py, plot_state_prediction.py   (Phase 2)
├── analysis/                     ← comparative studies (each with its own README)
│   ├── teacher_student_stability/     ← HMM vs LSTM vs HMM–LSTM stability
│   ├── adaptive_reconstruction_value/ ← static vs adaptive system
│   └── window_length_effect/          ← 3-day vs 5-day window sensitivity
└── results/                      ← all generated outputs (data tables + figures)
```

## Environments

The HMM stage (`hmmlearn`) and the LSTM stage (TensorFlow) use incompatible
`scikit-learn` versions, so they run in **two separate conda environments**:

| Environment | Install | Used by |
|-------------|---------|---------|
| **dwf-hmm-pip** | `pip install -r requirements-hmm.txt` | Phase 1, Phase 2 part 1, reconstruction, HMM-based figures/analyses |
| **dwf-ann**     | `pip install -r requirements-ann.txt` | Phase 2 part 2 / part 3, LSTM-based analyses |

Each script's header states which environment it needs.

## Data

This repository releases **code only**. The monitoring data are governed through
the project's **HydroShare** resource and are not redistributed here:

- **Dataset (DOI):** https://doi.org/10.4211/hs.c9be88f8d04e40df8aeea88dd6d7274f

To run the code without the real data, generate a synthetic sample
(`python data/make_synthetic_sample.py`). See `data/README.md` for full data
provenance, what each phase reads, and how to reuse the cleaning method on other
data.

## How to run

Run all commands **from the repository root**. Outputs go to `results/`.

**0. Set up both environments** from the two requirements files.

**1. Phase 1 — build the baseline archive** (env: dwf-hmm-pip). Place the input
at `data/outlet_phase1.csv` (or use the synthetic sample).
```bash
python script/phase1_baseline/build_baseline_archive.py
```
This writes the archive (`results/merged_state_features_physical_and_time.csv`)
and intermediate tables.

**2. Phase 1 figures** (env: dwf-hmm-pip).
```bash
python figures/plot_elbow_dendrogram.py
python figures/plot_posterior_segments.py
python figures/plot_baseline_radar_temporal.py
python figures/plot_baseline_violin_tpm.py
```

**3. Phase 2 — adaptive rolling recognition & prediction.** Each update cycle
runs as three stages across the two environments. Set the cycle in
`script/phase2_adaptive/config.py` (`ACTIVE_CYCLE` + its `CYCLES` entry). Phase 2
works directly on **already-cleaned** data: point `data_csv` at a cleaned sensor
file covering that cycle's time windows (phase 2 applies only forward/backward
fill). If your data still needs cleaning, clean it first with the same method as
phase 1 (`build_baseline_archive.py`). See `data/README.md`. Then:
```bash
# dwf-hmm-pip
python script/phase2_adaptive/part1_hmm_matching.py
# dwf-ann
python script/phase2_adaptive/part2_lstm_training.py
python script/phase2_adaptive/part3_rolling_inference.py
```
When part 3 reports a reconstruction trigger, run
`script/phase2_adaptive/reconstruction.py` (dwf-hmm-pip) with a manually chosen
state count, then continue with the new archive. See
`script/phase2_adaptive/README.md` for the full per-cycle workflow.

**4. Phase 2 figures** (env depends on the script header).
```bash
python figures/plot_state_matching.py      # dwf-hmm-pip
python figures/plot_state_recognition.py   # dwf-hmm-pip
python figures/plot_state_prediction.py    # any env
```

**5. Analyses.** After the full Phase 2 run, see each folder's README:
`analysis/teacher_student_stability/`, `analysis/adaptive_reconstruction_value/`,
`analysis/window_length_effect/`.

## Notes

- `results/` holds all generated outputs (data tables and figures) and is safe
  to delete and regenerate.
- The LSTM is saved as `lstm_model.h5` for compatibility with the pinned
  TensorFlow 2.10.
- Diagnostic thresholds in `config.py` are reference values; the final values
  used in the paper take precedence.
- Reproducibility: all random seeds are fixed (42). Results obtained from the
  HydroShare dataset are consistent with those reported in the paper, with minor
  numerical differences from the cleaning difference (see `data/README.md`).

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this software, please cite it using the metadata in `CITATION.cff`
(GitHub shows a "Cite this repository" button). The associated article citation
will be added on publication.
