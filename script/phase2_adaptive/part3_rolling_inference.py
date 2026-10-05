"""
part3_rolling_inference.py   (environment: dwf-ann)
==================================================
Stage 3 of an update cycle: autoregressive rolling inference over the forecast
horizon, sentinel diagnostics, performance-history update, and the
reconstruction-trigger decision.

At each step the previous LSTM posterior is propagated one step through the HMM
transition matrix to form the prior (R_prior), which is fed back into the LSTM
together with the current sensor window.

Output: results/HMM_LSTM_Model_<version>/inference_results_<version>.csv
Run:  python script/phase2_adaptive/part3_rolling_inference.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

import config
import diagnostics as dg


def load_inference_window(data_csv, feature_cols, scaler, start, end):
    df = pd.read_csv(data_csv, parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    try:
        win = df[start:end].copy()
    except Exception:
        df.index = df.index.tz_localize(None)
        win = df[start:end].copy()
    if win.empty:
        raise ValueError(f"No data in inference window {start}..{end}")
    win[feature_cols] = win[feature_cols].ffill().bfill()
    return win, scaler.transform(win[feature_cols])


def run(cfg):
    outdir = config.cycle_dir(cfg["version"])
    ns, ws = cfg["n_states"], config.WINDOW_SIZE

    model = load_model(os.path.join(outdir, "lstm_model.h5"), compile=False)
    transmat = np.load(os.path.join(outdir, "transmat.npy"))
    scaler = joblib.load(os.path.join(outdir, "scaler.joblib"))

    df_phys, data_scaled = load_inference_window(
        cfg["data_csv"], config.FEATURE_COLS, scaler, cfg["infer_start"], cfg["infer_end"])
    if len(data_scaled) <= ws:
        raise ValueError(f"Need more than {ws} samples; got {len(data_scaled)}.")

    # autoregressive inference
    last_Y = np.full(ns, 1.0 / ns)
    rows = []
    for t in range(ws, len(data_scaled)):
        X_seq = data_scaled[t - ws:t].reshape(1, ws, config.N_FEATURES)
        R_prior = last_Y @ transmat
        Y_soft = model.predict([X_seq, R_prior.reshape(1, ns)], verbose=0)[0]
        rows.append({
            "Datetime": df_phys.index[t],
            "Dominant_State": int(np.argmax(Y_soft)),
            "Entropy": dg.entropy(Y_soft),
            "KLD_vs_Prior": dg.kld(Y_soft, R_prior),
            **{f"State_{i}_Prob": float(Y_soft[i]) for i in range(ns)},
        })
        last_Y = Y_soft

    df_res = pd.DataFrame(rows).set_index("Datetime")
    out_csv = os.path.join(outdir, f"inference_results_{cfg['version'].lower()}.csv")
    df_res.to_csv(out_csv)
    print(f"Inference done ({len(df_res)} steps) -> {out_csv}")

    # reference stability metrics
    sw = dg.switching_frequency(df_res["Dominant_State"].values, df_res.index)
    sd = dg.short_dwell_ratio(df_res["Dominant_State"].values, df_res.index)
    print(f"switching/hour = {sw:.3f} | short-dwell ratio = {sd:.3f}")

    # sentinel diagnostics + reconstruction decision (+ metadata from part1)
    try:
        meta = joblib.load(os.path.join(outdir, "cycle_meta.joblib"))
    except FileNotFoundError:
        meta = {}
    diag = dg.diagnose_cycle(df_res, cfg["version"], ns, switches_per_hour=sw,
                             conf_mean_raw=meta.get("conf_mean_raw"),
                             baseline_used=meta.get("baseline_used"))
    print(f"H_mean={diag['record']['H_Mean']:.4f} | KLD_95={diag['record']['KLD_95']:.4f}")
    print(f"perf_healthy={diag['perf_healthy']} | state_healthy={diag['state_healthy']} "
          f"(vanishing={diag['vanishing']}) | scheduled={diag['scheduled']} "
          f"-> reconstruction_triggered={diag['reconstruction_triggered']}")

    if cfg.get("reconstruction_enabled", True) and diag["reconstruction_triggered"]:
        print("\n>>> Reconstruction triggered. Run reconstruction.py in the dwf-hmm "
              "environment, choose N manually from the printed occupancy/similarity, "
              "then start a new cycle against the new archive.")
    print(f"Stage 3 done -> {outdir}")


if __name__ == "__main__":
    run(config.CYCLES[config.ACTIVE_CYCLE])
