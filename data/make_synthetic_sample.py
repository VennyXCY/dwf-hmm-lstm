"""
make_synthetic_sample.py
========================
Generate a SYNTHETIC sensor file with the same schema as the real input, so the
pipeline can be run end-to-end for a functionality check without the real data.

The output is entirely artificial (a hidden 3-regime Markov chain plus diurnal
patterns and noise). It does NOT reproduce the paper's results; it only lets you
verify that the code runs. The real monitoring data are available in the
HydroShare resource (see README.md).

Output: data/synthetic_sensordata.csv  (columns: Datetime, Temp, NH3, Turb, Cond, pH, Flow)

Usage:
    python data/make_synthetic_sample.py
Then either set INPUT_CSV in build_baseline_archive.py to this file, or copy it
to data/outlet_phase1.csv.
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT_CSV = os.path.join("data", "synthetic_sensordata.csv")

FEATURES = ["Temp", "NH3", "Turb", "Cond", "pH", "Flow"]
FREQ_MIN = 10          # sampling interval (minutes)
N_REGIMES = 3

# Per-regime feature means (arbitrary but plausible ranges)
REGIME_MEANS = {
    "Temp": [19.0, 22.0, 24.0],
    "NH3":  [40.0, 120.0, 220.0],
    "Turb": [15.0, 40.0, 70.0],
    "Cond": [600.0, 1100.0, 1600.0],
    "pH":   [7.2, 7.0, 6.8],
    "Flow": [8000.0, 20000.0, 35000.0],   # L/h (matches the raw unit convention)
}
REGIME_STD = {"Temp": 0.6, "NH3": 15, "Turb": 6, "Cond": 80, "pH": 0.08, "Flow": 2500}

# Sticky regime transition matrix
TRANS = np.array([[0.94, 0.04, 0.02],
                  [0.04, 0.92, 0.04],
                  [0.02, 0.05, 0.93]])


def simulate_block(start, days):
    """Simulate one continuous block of `days` days starting at `start`."""
    n = int(days * 24 * 60 / FREQ_MIN)
    times = pd.date_range(start=start, periods=n, freq=f"{FREQ_MIN}min")
    hour = times.hour + times.minute / 60.0
    diurnal = np.sin((hour - 6) / 24 * 2 * np.pi)  # peak in the afternoon

    # Hidden regime sequence
    regimes = np.empty(n, dtype=int)
    regimes[0] = RNG.integers(N_REGIMES)
    for t in range(1, n):
        regimes[t] = RNG.choice(N_REGIMES, p=TRANS[regimes[t - 1]])

    data = {"Datetime": times}
    for f in FEATURES:
        base = np.array([REGIME_MEANS[f][r] for r in regimes], dtype=float)
        series = base + RNG.normal(0, REGIME_STD[f], n)
        if f in ("Temp", "Flow"):
            series += diurnal * (1.5 if f == "Temp" else 4000.0)
        data[f] = series
    return pd.DataFrame(data)


def main():
    os.makedirs("data", exist_ok=True)
    # Two blocks separated by a multi-day gap -> two segments in the pipeline
    b1 = simulate_block("2025-01-01 00:00:00", days=8)
    b2 = simulate_block("2025-01-14 00:00:00", days=8)
    df = pd.concat([b1, b2], ignore_index=True)
    df["Flow"] = df["Flow"].clip(lower=0)
    df.to_csv(OUT_CSV, index=False)
    print(f"[SYNTHETIC] wrote {len(df)} rows to {OUT_CSV}")
    print("This is artificial data for a functionality check only, "
          "not the real monitoring data.")


if __name__ == "__main__":
    main()
