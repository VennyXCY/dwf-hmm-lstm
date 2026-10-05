"""
plot_state_recognition.py   (environment: dwf-hmm)
=================================================
State-recognition figure for a single update cycle: (a) the standardized HMM
dominant baseline state over time and (b) the posterior probabilities of the
baseline states, with Evolved States shaded/dashed. Recomputed from the archive
and the cleaned window (same procedure as phase 2).

Select the cycle with CONFIG below. Reads the archive from results/ and the
cleaned sensor file from data/; writes to results/.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_SAVE_DIR = RESULTS_DIR

# ---------- per-cycle configuration ----------
V1_CONFIG = dict(
    version="V1", n_states=4,
    archive_path=os.path.join(RESULTS_DIR, "merged_state_features_physical_and_time.csv"),
    data_path=os.path.join(DATA_DIR, "OUTLET4-11.1-11.10-CLEANED-m3h.csv"),
    start="2025-11-01 00:00:00", end="2025-11-03 23:59:00")
V10_CONFIG = dict(
    version="V10", n_states=5,
    archive_path=os.path.join(RESULTS_DIR, "merged_state_features_V7_N5.csv"),
    data_path=os.path.join(DATA_DIR, "OUTLET4-11.1-1.2-CLEANED-m3h.csv"),
    start="2025-11-22 00:00:00", end="2025-11-30 23:59:00")

CONFIG = V1_CONFIG   # <<< set the cycle to plot

MATCHING_ALPHA = 0.667
N_HOURS = 24
FEATURE_COLS = ['Temp', 'Cond', 'pH', 'NH3', 'Flow', 'Turb']
BREAK_THRESHOLD = pd.Timedelta(minutes=30)
CONFIDENCE_THRESHOLD = 0.65
PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A', '#8A8CBD', '#FECEA0', '#E6BEC7']
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 22,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.weight': 'bold',
    'axes.labelsize': 26, 'axes.titlesize': 26, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
})
FS_YLABEL, FS_XLABEL, FS_TICK, FS_LEG, FS_PANEL, FS_ANNO, FS_TITLE = 28, 28, 24, 22, 32, 20, 34


# ---------- pipeline ----------
def load_baseline_archive(fp):
    df = pd.read_csv(fp); tc = [f'Hour_{i}' for i in range(N_HOURS)]
    return df[FEATURE_COLS].values, df[tc].values


def load_and_prep_new_data(fp, start, end):
    df = pd.read_csv(fp, parse_dates=['Datetime'], encoding='utf-8')
    df = df[['Datetime'] + FEATURE_COLS].set_index('Datetime').sort_index()
    try:
        dc = df[pd.Timestamp(start):pd.Timestamp(end)].copy()
    except Exception:
        df.index = df.index.tz_localize(None)
        dc = df[pd.Timestamp(start):pd.Timestamp(end)].copy()
    dc = dc.ffill().bfill()
    sc = StandardScaler(); ds = dc.copy()
    ds[FEATURE_COLS] = sc.fit_transform(dc[FEATURE_COLS])
    return ds, dc, sc


def get_data_chunks(df, thr):
    df = df.sort_index(); diffs = df.reset_index()['Datetime'].diff()
    bps = diffs[diffs > thr].index; lengths, s = [], 0
    for bp in bps:
        if bp - s > 0: lengths.append(bp - s)
        s = bp
    if len(df) - s > 0: lengths.append(len(df) - s)
    return df.values, lengths


def fit_local_hmm(X, lengths, ns):
    m = hmm.GaussianHMM(n_components=ns, covariance_type="full", n_iter=100, random_state=42)
    m.fit(X, lengths); return m


def extract_local_time_dists(df_dt, seq, ns):
    t = df_dt.copy(); t['s'] = seq; t['hour'] = t.index.hour; out = []
    for i in range(ns):
        sd = t[t['s'] == i]
        out.append(np.zeros(N_HOURS) if sd.empty
                   else sd['hour'].value_counts(normalize=True).reindex(range(N_HOURS), fill_value=0.0).values)
    return np.array(out)


def get_state_mapping_combined(lm, bm, lt, bt, alpha, ns):
    s = MinMaxScaler(); ln = s.fit_transform(lm); bn = s.transform(bm)
    dwq = cdist(ln, bn, 'euclidean'); dt = cdist(lt, bt, 'euclidean')
    dtot = alpha * MinMaxScaler().fit_transform(dwq) + (1 - alpha) * MinMaxScaler().fit_transform(dt)
    r, c = linear_sum_assignment(dtot); mp = np.zeros(ns, dtype=int); mp[r] = c
    return mp, dtot


def check_matching_confidence(dm, mp, thr):
    ev, cf = {}, {}
    for ls, bs in enumerate(mp):
        c = 1.0 / (1.0 + dm[ls, bs]); cf[ls] = c
        if c < thr: ev[bs] = c
    return ev, cf


def standardize_hmm(m, mp, ns):
    std = hmm.GaussianHMM(n_components=ns, covariance_type="full", random_state=42)
    std.startprob_ = m.startprob_[mp]
    std.transmat_ = m.transmat_[mp][:, mp]
    std.means_ = m.means_[mp]
    std.covars_ = m.covars_[mp]
    return std


def compute_recognition(cfg):
    bwq, bt = load_baseline_archive(cfg['archive_path'])
    ds, dp, sc = load_and_prep_new_data(cfg['data_path'], cfg['start'], cfg['end'])
    X, lengths = get_data_chunks(ds[FEATURE_COLS], BREAK_THRESHOLD)
    m = fit_local_hmm(X, lengths, cfg['n_states'])
    lm = sc.inverse_transform(m.means_)
    seq = m.predict(X, lengths)
    lt = extract_local_time_dists(dp, seq, cfg['n_states'])
    mp, dm = get_state_mapping_combined(lm, bwq, lt, bt, MATCHING_ALPHA, cfg['n_states'])
    ev, cf = check_matching_confidence(dm, mp, CONFIDENCE_THRESHOLD)
    std = standardize_hmm(m, mp, cfg['n_states'])
    post = std.predict_proba(X, lengths)
    dom = std.predict(X, lengths)
    dfp = ds.copy(); dfp['Dominant_State'] = dom
    for i in range(cfg['n_states']):
        dfp[f'State_{i}_Prob'] = post[:, i]
    return dict(dfp=dfp, ns=cfg['n_states'], version=cfg['version'], ev=ev, cf=cf)


# ---------- panels ----------
def draw_dominant(ax, res, panel):
    dfp, ns, ev, cf = res['dfp'], res['ns'], res['ev'], res['cf']
    line, = ax.plot(dfp.index, dfp['Dominant_State'], color=PALETTE[0],
                    linewidth=2.5, label='Dominant State')
    handles = [line]
    eids = list(ev.keys())
    if eids:
        mask = dfp['Dominant_State'].isin(eids); in_e = False; seg = None
        for t, v in mask.items():
            if v and not in_e: seg = t; in_e = True
            elif not v and in_e: ax.axvspan(seg, t, color='#EFA485', alpha=0.18, linewidth=0); in_e = False
        if in_e: ax.axvspan(seg, dfp.index[-1], color='#EFA485', alpha=0.18, linewidth=0)
        handles.append(Patch(facecolor='#EFA485', alpha=0.4, label=f'Evolved State Period (S{eids})'))
    ax.legend(handles=handles, loc='upper right', fontsize=FS_LEG, frameon=True,
              framealpha=0.85, edgecolor='#cccccc')
    ax.set_ylabel('Baseline State ID', fontsize=FS_YLABEL, fontweight='bold')
    ax.set_yticks(range(ns)); ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7); ax.set_axisbelow(True)
    for sid in range(ns):
        c = cf.get(sid)
        if c is None: continue
        mask = dfp['Dominant_State'] == sid
        if mask.any():
            idxs = dfp.index[mask]; mid = idxs[len(idxs) // 2]
            cc = '#B6766A' if c < CONFIDENCE_THRESHOLD else '#4E6597'
            ax.annotate(f'S{sid}\nConf: {c:.2f}', xy=(mid, sid), fontsize=FS_ANNO,
                        fontweight='bold', color='white', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.45', facecolor=cc, alpha=0.88, edgecolor='none'))
    ax.set_title(f"State recognition at {res['version']}", fontsize=FS_TITLE,
                 fontweight='bold', pad=24, color='#333333')
    ax.text(-0.035, 1.10, panel, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='top', ha='left', color='#333333')


def draw_posterior(ax, res, panel):
    dfp, ns, ev, cf = res['dfp'], res['ns'], res['ev'], res['cf']
    for i in range(ns):
        is_e = i in ev
        cv = ev[i] if is_e else cf[i]
        lab = f'State {i} (Evolved, Conf: {cv:.2f})' if is_e else f'State {i} (Conf: {cv:.2f})'
        ax.plot(dfp.index, dfp[f'State_{i}_Prob'], label=lab, linewidth=2.0,
                color=PALETTE[i % len(PALETTE)], linestyle='--' if is_e else 'solid')
    ax.set_ylabel('Probability', fontsize=FS_YLABEL, fontweight='bold')
    ax.set_xlabel('Datetime', fontsize=FS_XLABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='upper right', fontsize=FS_LEG, frameon=True, framealpha=0.85, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7); ax.set_axisbelow(True)
    ax.text(-0.035, 1.10, panel, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='top', ha='left', color='#333333')


def run(cfg):
    res = compute_recognition(cfg)
    fig = plt.figure(figsize=(26, 12))
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.07, right=0.97,
                           top=0.92, bottom=0.10, hspace=0.16)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    draw_dominant(ax_a, res, '(a)')
    draw_posterior(ax_b, res, '(b)')
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax_b.tick_params(axis='x', labelsize=FS_TICK)
    idx = res['dfp'].index
    ax_b.set_xlim(idx.min(), idx.max().ceil('D'))

    base = f"state_recognition_{cfg['version']}"
    png = os.path.join(FIG_SAVE_DIR, base + ".png")
    pdf = os.path.join(FIG_SAVE_DIR, base + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    run(CONFIG)
