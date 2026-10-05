"""
compute_matching_distance.py   (environment: dwf-hmm)
====================================================
Value-of-adaptive-reconstruction analysis, part 2: how well the archive still
fits the incoming data, for the STATIC (frozen) archive versus the ADAPTIVE
(refreshed) archive. For each cycle, a local HMM is re-fitted on the training
window, matched to the archive, and the mean absolute matching distance is
measured in a FIXED global z-space.

Static archive  = results/merged_state_features_physical_and_time.csv (frozen).
Adaptive archive = the archive in force for the cycle (phase-2 config archive_csv).
Output: results/archive_data_distance.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from hmmlearn import hmm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, os.path.join(REPO_ROOT, "script", "phase2_adaptive"))
import config as p2  # noqa: E402

FEATURE_COLS = p2.FEATURE_COLS
N_HOURS = p2.N_HOURS
MATCHING_ALPHA = p2.MATCHING_ALPHA
BREAK = p2.BREAK_THRESHOLD
STATIC_ARCHIVE = os.path.join(RESULTS_DIR, "merged_state_features_physical_and_time.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "archive_data_distance.csv")


def load_archive(fp):
    df = pd.read_csv(fp)
    wq = df[FEATURE_COLS].values
    tm = df[[f'Hour_{i}' for i in range(N_HOURS)]].values
    return wq, tm, wq.shape[0]


def get_chunks(df_feat):
    diffs = pd.Series(df_feat.index).diff()
    bk = diffs[diffs > BREAK].index
    lengths, start = [], 0
    for bp in bk:
        if bp - start > 0: lengths.append(bp - start)
        start = bp
    if len(df_feat) - start > 0: lengths.append(len(df_feat) - start)
    return df_feat.values, lengths


def time_dists(df_phys, seq, n):
    tmp = df_phys.copy(); tmp['st'] = seq; tmp['hr'] = tmp.index.hour
    out = []
    for i in range(n):
        sub = tmp[tmp['st'] == i]
        out.append(np.zeros(N_HOURS) if sub.empty else
                   sub['hr'].value_counts(normalize=True).reindex(range(N_HOURS), fill_value=0.0).values)
    return np.array(out)


def mean_abs_distance(df, train_start, train_end, arch_wq, arch_tm, n, gscaler):
    seg = df.loc[train_start:train_end, FEATURE_COLS].ffill().bfill().dropna()
    if len(seg) < n * 60:
        return np.nan
    lsc = StandardScaler()
    segz = pd.DataFrame(lsc.fit_transform(seg), index=seg.index, columns=FEATURE_COLS)
    X, lengths = get_chunks(segz)
    m = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=100,
                        random_state=42, tol=1e-3)
    m.fit(X, lengths)
    local_phys = lsc.inverse_transform(m.means_)
    local_tm = time_dists(seg, m.predict(X, lengths), n)
    # assignment with the same normalized combined distance used in the pipeline
    sc = MinMaxScaler(); lm = sc.fit_transform(local_phys); sm = sc.transform(arch_wq)
    d_wq = MinMaxScaler().fit_transform(cdist(lm, sm, 'euclidean'))
    d_tm = MinMaxScaler().fit_transform(cdist(local_tm, arch_tm, 'euclidean'))
    d_tot = MATCHING_ALPHA * d_wq + (1 - MATCHING_ALPHA) * d_tm
    r, c = linear_sum_assignment(d_tot)
    # report: absolute Euclidean distance of matched pairs in the FIXED global z-space
    local_z = gscaler.transform(local_phys)
    arch_z = gscaler.transform(arch_wq)
    return float(np.mean([np.linalg.norm(local_z[i] - arch_z[j]) for i, j in zip(r, c)]))


def main():
    # global z-space from all cycle data files (fixed reference)
    frames = []
    for path in {cfg['data_csv'] for cfg in p2.CYCLES.values()}:
        d = pd.read_csv(path, parse_dates=['Datetime']).set_index('Datetime').sort_index()
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        frames.append(d[FEATURE_COLS])
    gscaler = StandardScaler().fit(pd.concat(frames).dropna())

    s_wq, s_tm, s_n = load_archive(STATIC_ARCHIVE)
    arch_cache = {}
    rows = []
    for cfg in p2.CYCLES.values():
        v = cfg['version']
        df = pd.read_csv(cfg['data_csv'], parse_dates=['Datetime']).set_index('Datetime').sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        d_static = mean_abs_distance(df, cfg['train_start'], cfg['train_end'], s_wq, s_tm, s_n, gscaler)
        af = cfg['archive_csv']
        if af not in arch_cache:
            arch_cache[af] = load_archive(af)
        a_wq, a_tm, a_n = arch_cache[af]
        d_adap = mean_abs_distance(df, cfg['train_start'], cfg['train_end'], a_wq, a_tm, a_n, gscaler)
        rows.append({'Model_Version': v, 'Static_Dist': d_static, 'Adaptive_Dist': d_adap})
        print(f"  {v}: static={d_static:.3f}  adaptive={d_adap:.3f}  (archive={os.path.basename(af)}, N={a_n})")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
