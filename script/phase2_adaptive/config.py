"""
config.py
=========
Shared configuration for the phase-2 adaptive pipeline.

Because the HMM stage (hmmlearn) and the LSTM stage (TensorFlow) run in two
different conda environments, each update cycle is executed as three separate
stages, exactly as in the paper:

    part1_hmm_matching.py   (env: dwf-hmm)   -> HMM fit, matching, LSTM targets
    part2_lstm_training.py  (env: dwf-ann)   -> train the teacher-student LSTM
    part3_rolling_inference.py (env: dwf-ann)-> autoregressive inference + diagnostics

Set the cycle you want to run in ACTIVE_CYCLE and run the three stages in order.
Paths are resolved relative to the repository root, so the scripts work from any
working directory.
"""

import os
import pandas as pd

# --- Paths (relative to the repository root) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
HISTORY_CSV = os.path.join(RESULTS_DIR, "performance_history.csv")

# --- Model / matching constants ---
FEATURE_COLS = ["Temp", "Cond", "pH", "NH3", "Flow", "Turb"]
N_FEATURES = 6
N_HOURS = 24
WINDOW_SIZE = 30                # LSTM input window (samples)
MATCHING_ALPHA = 0.667          # cost-matrix weight (water-quality/flow vs time)
CONFIDENCE_THRESHOLD = 0.65     # matching confidence below this -> Evolved State
BREAK_THRESHOLD = pd.Timedelta(minutes=30)
SEED = 42

# --- Diagnostic thresholds ---
# REFERENCE values only; the final values used in the paper take precedence.
H_ALERT = 0.30                  # mean predictive-entropy alert
KLD_ALERT = 0.10                # 95th-percentile KLD-vs-prior alert
STATE_VANISH = 0.01             # baseline state below this frequency = vanishing
STATE_ANALYSIS_WINDOW = 3       # cycles averaged for the vanishing check
RECONSTRUCTION_VERSION_THRESHOLD = 10   # scheduled long-term reconstruction (~30 days)
SHORT_DWELL_MIN = 10            # minutes; dwell shorter than this counts as short


def cycle_dir(version):
    """Per-cycle output directory under results/ (created if needed)."""
    d = os.path.join(RESULTS_DIR, f"HMM_LSTM_Model_{version}")
    os.makedirs(os.path.join(d, "ann_training_data"), exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Per-cycle configuration (one entry per update cycle).
#   - archive_csv : the baseline archive in force for this cycle
#                   (results/merged_state_features_physical_and_time.csv for the
#                    initial N=4 archive; a reconstructed archive after V10/V20)
#   - data_csv    : the cleaned sensor file (same cleaning as phase 1; see
#                   data/README.md). 
#   - train window (9 days) / inference horizon (3 days): W=9, S=3, H=3
#   - reconstruction_enabled : False during a post-reconstruction protection period
# Edit / extend this table for V1..V24.
# ---------------------------------------------------------------------------
CYCLES = {
    "V4": dict(
        version="V4",
        n_states=4,
        archive_csv=os.path.join(RESULTS_DIR, "merged_state_features_physical_and_time.csv"),
        data_csv=os.path.join(DATA_DIR, "OUTLET4-11.1-11.21-CLEANED-m3h.csv"),
        train_start="2025-11-04 00:00:00", train_end="2025-11-12 23:59:00",
        infer_start="2025-11-13 00:00:00", infer_end="2025-11-15 23:59:00",
        reconstruction_enabled=False,
    ),
    # "V13": dict(version="V13", n_states=5,
    #     archive_csv=os.path.join(RESULTS_DIR, "merged_state_features_V7_N5.csv"),
    #     data_csv=..., train_start=..., train_end=..., infer_start=..., infer_end=...,
    #     reconstruction_enabled=True),
}

# The cycle the stage scripts act on. Change this before running each stage.
ACTIVE_CYCLE = "V4"
