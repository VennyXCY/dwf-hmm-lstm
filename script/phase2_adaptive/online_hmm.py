"""
online_hmm.py   (environment: dwf-hmm)
=====================================
HMM-stage helpers for one update cycle: load the training window, fit a local
Gaussian HMM, extract its state structure, and build the soft targets used to
train the teacher-student LSTM. The HMM transition matrix A also drives the
single-step online prior used during rolling inference (hmm_prior_step).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

import config


def load_and_prep_window(data_csv, feature_cols, start, end):
    """Load a cleaned sensor file, slice the training window, fill residual NaNs
    (the input is already cleaned with the same method as phase 1; see
    data/README.md), and fit a LOCAL standardizer on this window."""
    df = pd.read_csv(data_csv, parse_dates=["Datetime"], encoding="utf-8")
    df = df[["Datetime"] + feature_cols].set_index("Datetime").sort_index()
    try:
        win = df[pd.Timestamp(start):pd.Timestamp(end)].copy()
    except Exception:
        df.index = df.index.tz_localize(None)
        win = df[pd.Timestamp(start):pd.Timestamp(end)].copy()
    if win.empty:
        raise ValueError(f"No data in window {start}..{end} of {data_csv}")
    win = win.ffill().bfill()
    scaler = StandardScaler()
    std = win.copy()
    std[feature_cols] = scaler.fit_transform(win[feature_cols])
    return std, win, scaler


def get_data_chunks(df, threshold=config.BREAK_THRESHOLD):
    """Split into continuous chunks at time gaps (hmmlearn 'lengths')."""
    df = df.sort_index()
    diffs = df.reset_index()["Datetime"].diff()
    bps = diffs[diffs > threshold].index
    lengths, s = [], 0
    for bp in bps:
        if bp - s > 0:
            lengths.append(bp - s)
        s = bp
    if len(df) - s > 0:
        lengths.append(len(df) - s)
    return df.values, lengths


def fit_local_hmm(X_concat, lengths, n_states):
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=100, tol=1e-3, random_state=config.SEED, verbose=False)
    m.fit(X_concat, lengths)
    if not m.monitor_.converged:
        print("Warning: local HMM did not converge.")
    return m


def extract_local_time_dists(df_with_datetime, state_sequence, n_states):
    """Hour-of-day distribution (24-vector) for each local state."""
    t = df_with_datetime.copy()
    t["s"] = state_sequence
    t["hour"] = t.index.hour
    out = []
    for i in range(n_states):
        sd = t[t["s"] == i]
        if sd.empty:
            out.append(np.zeros(config.N_HOURS))
        else:
            out.append(sd["hour"].value_counts(normalize=True)
                       .reindex(range(config.N_HOURS), fill_value=0.0).values)
    return np.array(out)


def hmm_prior_step(posterior, transmat):
    """Single-step online HMM prior: propagate the previous posterior one step
    through the transition matrix. Used recursively during rolling inference and
    to build the LSTM's R_prior training input."""
    return posterior @ transmat


def generate_ann_training_data(X_concat, lengths, standardized_hmm, window_size):
    """Build (X_seq, R_prior) -> Y_soft samples. Y_soft is the standardized HMM
    posterior at time t; R_prior is the one-step HMM prior from t-1."""
    A = standardized_hmm.transmat_
    X_seq_list, R_prior_list, Y_list = [], [], []
    s = 0
    for L in lengths:
        chunk = X_concat[s:s + L]
        if L >= window_size + 1:
            Y_soft = standardized_hmm.predict_proba(chunk)
            for t in range(window_size, L):
                X_seq_list.append(chunk[t - window_size:t])
                R_prior_list.append(hmm_prior_step(Y_soft[t - 1], A))
                Y_list.append(Y_soft[t])
        s += L
    return np.array(X_seq_list), np.array(R_prior_list), np.array(Y_list)
