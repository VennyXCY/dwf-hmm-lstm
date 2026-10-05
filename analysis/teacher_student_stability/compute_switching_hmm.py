"""
compute_switching_hmm.py   (environment: dwf-hmm)
================================================
Prediction-stability analysis, part 1: for each update cycle, compute the state
switching rate (switches/hour) and short-dwell ratio of (i) the standalone
online HMM and (ii) the proposed HMM-LSTM.

The standalone HMM is re-fitted on the cycle's training window and run with
single-step online filtering over the forecast horizon; the HMM-LSTM sequence is
read from the saved phase-2 inference results.

Cycle metadata (dates, n_states, data file) is read from the phase-2 config.
Output: results/switching_hmm_vs_hmmlstm.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from hmmlearn import hmm as hmmlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, os.path.join(REPO_ROOT, "script", "phase2_adaptive"))
import config as p2  # noqa: E402

FEATURE_COLS = p2.FEATURE_COLS
SHORT_DWELL_MIN = p2.SHORT_DWELL_MIN
BREAK = p2.BREAK_THRESHOLD
OUT_CSV = os.path.join(RESULTS_DIR, "switching_hmm_vs_hmmlstm.csv")


def emission_probs(h, X):
    T, n = len(X), h.n_components
    em = np.zeros((T, n))
    for s in range(n):
        em[:, s] = multivariate_normal.pdf(X, mean=h.means_[s], cov=h.covars_[s],
                                            allow_singular=True)
    rs = em.sum(1, keepdims=True); rs[rs == 0] = 1e-300
    return em / rs


def hmm_online_predict(h, X):
    """Single-step online HMM filtering: argmax of the filtered posterior."""
    em = emission_probs(h, X); A = h.transmat_
    last = h.startprob_.copy(); out = []
    for t in range(len(X)):
        prior = last @ A; post = prior * em[t]
        s = post.sum(); post = (prior.copy() if s == 0 else post / s)
        out.append(int(np.argmax(post))); last = post
    return np.array(out)


def get_chunks(df_feat):
    diffs = pd.Series(df_feat.index).diff()
    bk = diffs[diffs > BREAK].index
    lengths, start = [], 0
    for bp in bk:
        if bp - start > 0: lengths.append(bp - start)
        start = bp
    if len(df_feat) - start > 0: lengths.append(len(df_feat) - start)
    return df_feat.values, lengths


def switch_metrics(seq):
    seq = np.asarray(seq); n = len(seq)
    if n < 2:
        return np.nan, np.nan
    sw = int(np.sum(seq[1:] != seq[:-1])); hrs = n / 60.0
    runs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]: j += 1
        runs.append(j - i + 1); i = j + 1
    short = sum(L for L in runs if L < SHORT_DWELL_MIN)
    return sw / hrs, short / n


def main():
    rows = []
    for cfg in p2.CYCLES.values():
        v, n = cfg['version'], cfg['n_states']
        md = os.path.join(RESULTS_DIR, f"HMM_LSTM_Model_{v}")
        inf = sorted(glob.glob(os.path.join(md, "inference_results_*.csv")))
        if not inf:
            print(f"  {v}: no inference_results, skip"); continue

        df = pd.read_csv(cfg['data_csv'], parse_dates=['Datetime']).set_index('Datetime').sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep='first')]

        # standalone HMM: refit on training window, online-filter over horizon
        train = df.loc[cfg['train_start']:cfg['train_end'], FEATURE_COLS].ffill().bfill().dropna()
        if len(train) < n * 50:
            print(f"  {v}: training window too short, skip"); continue
        sc = StandardScaler()
        trainz = pd.DataFrame(sc.fit_transform(train), index=train.index, columns=FEATURE_COLS)
        X, lengths = get_chunks(trainz)
        m = hmmlib.GaussianHMM(n_components=n, covariance_type="full",
                               n_iter=100, random_state=42, tol=1e-3)
        m.fit(X, lengths)

        inf_seg = df.loc[cfg['infer_start']:cfg['infer_end'], FEATURE_COLS].ffill().bfill().dropna()
        hmm_seq = pd.Series(hmm_online_predict(m, sc.transform(inf_seg)),
                            index=inf_seg.index, name='HMM')
        hmm_seq = hmm_seq[~hmm_seq.index.duplicated(keep='first')]

        # HMM-LSTM: read saved inference
        ann = pd.read_csv(inf[0], parse_dates=['Datetime']).set_index('Datetime')['Dominant_State']
        ann = ann[~ann.index.duplicated(keep='first')]
        comp = pd.concat([hmm_seq, ann.rename('HMMLSTM')], axis=1, join='inner').dropna()
        if len(comp) < 60:
            print(f"  {v}: aligned series too short, skip"); continue

        h_sw, h_sr = switch_metrics(comp['HMM'].values)
        a_sw, a_sr = switch_metrics(comp['HMMLSTM'].values)
        rows.append({'Version': v, 'N': n,
                     'HMM_switch_ph': h_sw, 'HMMLSTM_switch_ph': a_sw,
                     'HMM_short_ratio': h_sr, 'HMMLSTM_short_ratio': a_sr})
        print(f"  {v}: HMM_sw={h_sw:.2f}  HMM-LSTM_sw={a_sw:.2f}  (N={n})")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
