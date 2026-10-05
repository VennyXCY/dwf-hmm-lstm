"""
diagnostics.py   (numpy/pandas; runs in either environment)
==========================================================
Online sentinel diagnostics for each update cycle: predictive entropy and KL
divergence to the HMM prior, state switching and short-dwell statistics,
state-frequency (vanishing) analysis, the per-cycle performance-history record,
and the reconstruction-trigger decision.

Thresholds live in config.py as REFERENCE values; the final values reported in
the paper take precedence.
"""

import os
import numpy as np
import pandas as pd

import config


# --- per-timestep sentinel metrics ---
def entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def kld(p, q):
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return float(np.sum(p * np.log(p / q)))


# --- window-level stability metrics (reference) ---
def switching_frequency(states, index):
    """State switches per hour over the inference window."""
    states = np.asarray(states)
    index = pd.DatetimeIndex(index)
    n_switch = int((np.diff(states) != 0).sum())
    hours = (index[-1] - index[0]).total_seconds() / 3600.0
    return n_switch / hours if hours > 0 else np.nan


def short_dwell_ratio(states, index, min_minutes=config.SHORT_DWELL_MIN):
    """Fraction of time spent in state visits shorter than min_minutes."""
    states = np.asarray(states)
    index = pd.DatetimeIndex(index)
    change = np.where(np.diff(states) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(states)]])
    total = (index[-1] - index[0]).total_seconds()
    short = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        end = index[b - 1] if b - 1 < len(index) else index[-1]
        dur = (end - index[a]).total_seconds()
        if dur < min_minutes * 60:
            short += dur
    return short / total if total > 0 else np.nan


def state_frequencies(df_results, n_states):
    total = len(df_results)
    return {f"State_{i}_Freq": (df_results["Dominant_State"] == i).sum() / total
            for i in range(n_states)}


# --- performance history + reconstruction trigger ---
def _expected_cols(n_states):
    freq = [f"State_{i}_Freq" for i in range(n_states)]
    return (["Model_Version", "N_States", "Baseline_Used", "Start_Date", "End_Date",
             "H_Mean", "KLD_95", "Conf_Mean_Raw", "Switches_Per_Hour",
             "H_Triggered", "KLD_Triggered", "Stability_Alert",
             "Reconstruction_Triggered"] + freq)


def load_history(n_states):
    cols = _expected_cols(n_states)
    if os.path.exists(config.HISTORY_CSV):
        h = pd.read_csv(config.HISTORY_CSV)
        for c in cols:
            if c not in h.columns:
                h[c] = False if c.endswith(("_Triggered", "_Alert")) else 0.0
        return h.reindex(columns=cols, fill_value=0.0)
    return pd.DataFrame(columns=cols)


def diagnose_cycle(df_results, version, n_states, switches_per_hour=None,
                   conf_mean_raw=None, baseline_used=None):
    """Build this cycle's record and alerts, run the vanishing-state analysis
    over the last N cycles, decide whether reconstruction is triggered, and
    append the record to results/performance_history.csv.

    Reconstruction is triggered when performance is unhealthy (entropy/KLD alert),
    a baseline state is vanishing, or the scheduled long-term interval is reached.
    """
    h = load_history(n_states)
    h = h[h["Model_Version"] != version].copy()   # replace any previous same-version row

    rec = {
        "Model_Version": version,
        "N_States": n_states,
        "Baseline_Used": baseline_used,
        "Start_Date": df_results.index.min().strftime("%Y-%m-%d %H:%M:%S"),
        "End_Date": df_results.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        "H_Mean": float(df_results["Entropy"].mean()),
        "KLD_95": float(df_results["KLD_vs_Prior"].quantile(0.95)),
        "Conf_Mean_Raw": conf_mean_raw,
        "Switches_Per_Hour": switches_per_hour,
    }
    rec.update(state_frequencies(df_results, n_states))

    freq_cols = [f"State_{i}_Freq" for i in range(n_states)]
    window = pd.concat([h, pd.DataFrame([rec])], ignore_index=True).tail(config.STATE_ANALYSIS_WINDOW)
    avg_freq = window[freq_cols].mean(axis=0)
    vanishing = [i for i in range(n_states) if avg_freq[f"State_{i}_Freq"] < config.STATE_VANISH]
    state_healthy = len(vanishing) == 0

    rec["H_Triggered"] = bool(rec["H_Mean"] >= config.H_ALERT)
    rec["KLD_Triggered"] = bool(rec["KLD_95"] >= config.KLD_ALERT)
    rec["Stability_Alert"] = bool(not state_healthy)

    perf_healthy = not (rec["H_Triggered"] or rec["KLD_Triggered"])
    version_count = len(h) + 1
    scheduled = version_count >= config.RECONSTRUCTION_VERSION_THRESHOLD
    reconstruction_triggered = (not perf_healthy) or (not state_healthy) or scheduled
    rec["Reconstruction_Triggered"] = bool(reconstruction_triggered)

    out = pd.concat([h, pd.DataFrame([rec])], ignore_index=True)
    out = out.reindex(columns=_expected_cols(n_states))
    out.to_csv(config.HISTORY_CSV, index=False)

    return dict(record=rec, vanishing=vanishing, state_healthy=state_healthy,
                perf_healthy=perf_healthy, scheduled=scheduled,
                version_count=version_count,
                reconstruction_triggered=reconstruction_triggered)
