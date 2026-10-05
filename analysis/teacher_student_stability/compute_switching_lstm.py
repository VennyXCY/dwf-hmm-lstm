"""
compute_switching_lstm.py   (environment: dwf-ann)
=================================================
Prediction-stability analysis, part 2: the standalone (ablation) LSTM. For each
cycle, train an LSTM with the SAME architecture and training configuration as
the HMM-LSTM but WITHOUT the HMM-prior input branch, then compute its switching
rate over the forecast horizon.

Reads the phase-2 training data (ann_training_data/) and scaler per cycle, plus
cycle metadata from the phase-2 config. Merges the result with
switching_hmm_vs_hmmlstm.csv into switching_all.csv.

Output: results/switching_standalone_lstm.csv
        results/switching_all.csv
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, os.path.join(REPO_ROOT, "script", "phase2_adaptive"))
import config as p2  # noqa: E402

SEED = p2.SEED
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

FEATURE_COLS = p2.FEATURE_COLS
WINDOW = p2.WINDOW_SIZE
N_FEAT = p2.N_FEATURES
SHORT_DWELL_MIN = p2.SHORT_DWELL_MIN
OUT_CSV = os.path.join(RESULTS_DIR, "switching_standalone_lstm.csv")
HMM_CSV = os.path.join(RESULTS_DIR, "switching_hmm_vs_hmmlstm.csv")
ALL_CSV = os.path.join(RESULTS_DIR, "switching_all.csv")


def build_standalone_lstm(n):
    inp = Input((WINDOW, N_FEAT))
    x = LSTM(32, activation='relu')(inp)
    x = Dense(16, activation='relu')(x)
    out = Dense(n, activation='softmax')(x)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def switch_rate(seq):
    seq = np.asarray(seq); n = len(seq)
    if n < 2:
        return np.nan, np.nan
    sw = int(np.sum(seq[1:] != seq[:-1])); hrs = n / 60.0
    runs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]: j += 1
        runs.append(j - i + 1); i = j + 1
    return sw / hrs, sum(L for L in runs if L < SHORT_DWELL_MIN) / n


def main():
    rows = []
    for cfg in p2.CYCLES.values():
        v = cfg['version']
        md = os.path.join(RESULTS_DIR, f"HMM_LSTM_Model_{v}")
        adir = os.path.join(md, "ann_training_data")
        xf = os.path.join(adir, "X_seq_data.npy")
        yf = os.path.join(adir, "Y_target_data.npy")
        sp = os.path.join(md, "scaler.joblib")
        if not all(os.path.exists(p) for p in (xf, yf, sp)):
            print(f"  {v}: missing phase-2 artifacts, skip"); continue

        X = np.load(xf); Y = np.load(yf); n = Y.shape[1]
        scaler = joblib.load(sp)
        Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=0.2, random_state=SEED, shuffle=True)
        m = build_standalone_lstm(n)
        m.fit(Xtr, Ytr, validation_data=(Xv, Yv), epochs=100, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)])

        df = pd.read_csv(cfg['data_csv'], parse_dates=['Datetime']).set_index('Datetime').sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        seg = df.loc[cfg['infer_start']:cfg['infer_end'], FEATURE_COLS].ffill().bfill().dropna()
        Xs = scaler.transform(seg).astype('float32')
        if len(Xs) <= WINDOW:
            print(f"  {v}: horizon too short, skip"); continue
        wins = np.stack([Xs[t - WINDOW:t] for t in range(WINDOW, len(Xs))])
        pred = np.argmax(m.predict(wins, verbose=0), axis=1)
        sw, sr = switch_rate(pred)
        rows.append({'Version': v, 'StandaloneLSTM_switch_ph': sw, 'StandaloneLSTM_short_ratio': sr})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"  {v}: standalone-LSTM_sw={sw:.2f}  ({len(rows)} saved)")

    # merge with the HMM / HMM-LSTM table
    if os.path.exists(HMM_CSV):
        (pd.read_csv(HMM_CSV)
           .merge(pd.DataFrame(rows), on='Version', how='left')
           .to_csv(ALL_CSV, index=False))
        print(f"Saved: {ALL_CSV}")


if __name__ == "__main__":
    main()
