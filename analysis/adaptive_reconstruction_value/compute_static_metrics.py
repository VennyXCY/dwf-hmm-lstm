"""
compute_static_metrics.py   (environment: dwf-ann)
=================================================
Value-of-adaptive-reconstruction analysis, part 1: the STATIC (non-adaptive)
baseline. The first fully-trained model is frozen and applied to every cycle's
forecast horizon, and its mean predictive entropy, 95th-pct KLD and switching
rate are recorded for comparison with the adaptive framework.

The frozen model is V3 in this study (the first cycle with a complete training
window, after the initial protection period). For other data, set STATIC_VERSION
to your first fully-trained cycle.

Reads the frozen model from results/HMM_LSTM_Model_<STATIC_VERSION>/ and cycle
metadata from the phase-2 config. Output: results/static_history.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, os.path.join(REPO_ROOT, "script", "phase2_adaptive"))
import config as p2  # noqa: E402
import diagnostics as dg  # noqa: E402

STATIC_VERSION = "V3"              # frozen baseline = first fully-trained cycle
                                   # (V3 in this study; change for your own data)
STATIC_DIR = os.path.join(RESULTS_DIR, f"HMM_LSTM_Model_{STATIC_VERSION}")
FEATURE_COLS = p2.FEATURE_COLS
WINDOW = p2.WINDOW_SIZE
N_FEAT = p2.N_FEATURES
OUT_CSV = os.path.join(RESULTS_DIR, "static_history.csv")


def run_static_inference(predict_fn, transmat, scaler, df, start, end, n_states):
    seg = df.loc[start:end].copy()
    if len(seg) <= WINDOW:
        return None
    seg[FEATURE_COLS] = seg[FEATURE_COLS].ffill().bfill()
    X = scaler.transform(seg[FEATURE_COLS]).astype('float32')
    last_Y = np.full(n_states, 1.0 / n_states, dtype='float32')
    rows = []
    for t in range(WINDOW, len(X)):
        X_seq = X[t - WINDOW:t].reshape(1, WINDOW, N_FEAT)
        R_prior = (last_Y @ transmat).reshape(1, n_states)
        Y = predict_fn(tf.constant(X_seq), tf.constant(R_prior)).numpy()[0]
        rows.append({'Datetime': seg.index[t], 'Dominant_State': int(np.argmax(Y)),
                     'Entropy': dg.entropy(Y), 'KLD_vs_Prior': dg.kld(Y, R_prior.ravel())})
        last_Y = Y
    return pd.DataFrame(rows).set_index('Datetime')


def main():
    model = load_model(os.path.join(STATIC_DIR, "lstm_model.h5"), compile=False)
    transmat = np.load(os.path.join(STATIC_DIR, "transmat.npy")).astype('float32')
    scaler = joblib.load(os.path.join(STATIC_DIR, "scaler.joblib"))
    n_states = p2.CYCLES.get(STATIC_VERSION, {}).get("n_states", transmat.shape[0])

    @tf.function(reduce_retracing=True)
    def predict_fn(x_seq, r_prior):
        return model([x_seq, r_prior], training=False)

    rows = []
    for cfg in p2.CYCLES.values():
        v = cfg['version']
        df = pd.read_csv(cfg['data_csv'], parse_dates=['Datetime']).set_index('Datetime').sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        res = run_static_inference(predict_fn, transmat, scaler, df,
                                   cfg['infer_start'], cfg['infer_end'], n_states)
        if res is None:
            print(f"  [static] {v}: horizon too short, skip"); continue
        sw = dg.switching_frequency(res['Dominant_State'].values, res.index)
        rows.append({'Model_Version': v,
                     'H_Mean': float(res['Entropy'].mean()),
                     'KLD_95': float(res['KLD_vs_Prior'].quantile(0.95)),
                     'Switches_Per_Hour': sw})
        print(f"  [static] {v}: H={rows[-1]['H_Mean']:.3f} KLD95={rows[-1]['KLD_95']:.3f}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
