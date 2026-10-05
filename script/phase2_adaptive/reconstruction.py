"""
reconstruction.py   (environment: dwf-hmm)
=========================================
Scheduled/triggered reconstruction of the baseline archive (the "architect").
Fits Gaussian HMMs over a long window, reports BIC, state occupancy and state
similarity, and saves a new archive.

The number of states is NOT chosen by BIC alone: a manual override is applied
using occupancy/similarity judgement (pass n_override). Example from the paper
(V6 reconstruction): BIC suggested N=8, but two states occupied only ~5% each
(at risk of dropping below the ~1% vanishing line and causing frequent alarms).
N=5 was chosen instead, where the smallest state still held ~12% and no two
states were highly similar, giving a robust, well-separated archive.

Run this as a script when part3 reports a reconstruction trigger; inspect the
printed occupancy/similarity, set N, then start a new cycle against the new
archive.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from hmmlearn import hmm

import config

# --- reconstruction configuration (edit for the actual reconstruction window) ---
DATA_CSV = os.path.join(config.DATA_DIR, "OUTLET4-11.1-12.8-CLEANED-m3h.csv")
ANALYSIS_START = "2025-11-02 00:00:00"
ANALYSIS_END = "2025-11-21 23:59:00"    # ~ most recent 20-30 days
MIN_K, MAX_K = 3, 8
N_OVERRIDE = None                       # e.g. 5 to force N=5 regardless of BIC
RECON_TAG = "V10_N5"                    # label THIS reconstruction (e.g. cycle + N)
                                        # so it does not overwrite earlier ones
OUT_CSV = os.path.join(config.RESULTS_DIR, f"merged_state_features_{RECON_TAG}.csv")

RARE_FREQUENCY = 0.05                   # occupancy below this is flagged "rare"
HIGH_SIMILARITY = 0.85                  # similarity above this suggests merging


def load_long_window(data_csv, start, end):
    df = pd.read_csv(data_csv, parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    df = df[start:end].ffill().bfill()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[config.FEATURE_COLS])
    return df, X, [len(X)], scaler


def find_optimal_k_bic(X, lengths, min_k, max_k):
    best_bic, best_n = np.inf, -1
    print(f"BIC analysis (N={min_k}..{max_k}):")
    for n in range(min_k, max_k + 1):
        try:
            m = hmm.GaussianHMM(n_components=n, covariance_type="full",
                                n_iter=100, random_state=config.SEED, verbose=False)
            m.fit(X, lengths)
            bic = m.bic(X)
            print(f"  N={n} | BIC={bic:.0f}")
            if bic < best_bic:
                best_bic, best_n = bic, n
        except Exception:
            print(f"  N={n} | failed")
    return best_n


def fit_hmm(X, lengths, n_states):
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=100, random_state=config.SEED, verbose=False)
    m.fit(X, lengths)
    return m


def report_occupancy(model, X, lengths):
    states = model.predict(X, lengths)
    props = pd.Series(states).value_counts(normalize=True).sort_index()
    print("State occupancy:")
    for s in range(model.n_components):
        p = props.get(s, 0.0)
        flag = "   <-- rare" if p < RARE_FREQUENCY else ""
        print(f"  State {s}: {p * 100:5.2f}%{flag}")
    return props


def report_similarity(model):
    mn = MinMaxScaler().fit_transform(model.means_)
    sim = 1.0 / (1.0 + cdist(mn, mn, "euclidean"))
    n = model.n_components
    found = False
    print("State similarity:")
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > HIGH_SIMILARITY:
                print(f"  States {i} & {j} highly similar ({sim[i, j]:.2f}) -> consider merging")
                found = True
    if not found:
        print("  no highly similar states")
    return sim


def save_new_archive(df, model, X, lengths, scaler, out_csv):
    means_phys = scaler.inverse_transform(model.means_)
    states = model.predict(X, lengths)
    tmp = df.copy()
    tmp["State"] = states
    tmp["Hour"] = tmp.index.hour
    time_dists = []
    for i in range(model.n_components):
        sd = tmp[tmp["State"] == i]
        if sd.empty:
            time_dists.append(np.zeros(config.N_HOURS))
        else:
            time_dists.append(sd["Hour"].value_counts(normalize=True).sort_index()
                              .reindex(range(config.N_HOURS), fill_value=0).values)
    cols = config.FEATURE_COLS + [f"Hour_{i}" for i in range(config.N_HOURS)]
    data = np.hstack([means_phys, np.array(time_dists)])
    pd.DataFrame(data, columns=cols).to_csv(out_csv, index=False)
    print(f"New archive saved: {out_csv}")


def reconstruct(data_csv, start, end, out_csv, min_k=MIN_K, max_k=MAX_K, n_override=None):
    df, X, lengths, scaler = load_long_window(data_csv, start, end)
    bic_n = find_optimal_k_bic(X, lengths, min_k, max_k)
    print(f"BIC-suggested N = {bic_n}")
    n = bic_n if n_override is None else n_override
    if n_override is not None:
        print(f"MANUAL OVERRIDE: using N = {n} (inspect occupancy/similarity below)")
    model = fit_hmm(X, lengths, n)
    report_occupancy(model, X, lengths)
    report_similarity(model)
    save_new_archive(df, model, X, lengths, scaler, out_csv)
    return model, n


if __name__ == "__main__":
    reconstruct(DATA_CSV, ANALYSIS_START, ANALYSIS_END, OUT_CSV,
                min_k=MIN_K, max_k=MAX_K, n_override=N_OVERRIDE)
