"""
plot_state_matching.py   (environment: dwf-hmm)
==============================================
State-matching comparison figure for a single update cycle: for each shown
state, a radar of the normalized water-quality/flow fingerprint (local vs
matched baseline) and a bar chart of the hour-of-day distribution, with the
matching confidence annotated. The matching is recomputed from the archive and
the cleaned window (same procedure as phase 2).

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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from math import pi
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_SAVE_DIR = RESULTS_DIR

# ---------- per-cycle configuration (edit paths/dates to your data) ----------
V1_CONFIG = dict(
    version="V1", n_states=4, show_pairs=None,
    archive_path=os.path.join(RESULTS_DIR, "merged_state_features_physical_and_time.csv"),
    data_path=os.path.join(DATA_DIR, "OUTLET4-11.1-11.10-CLEANED-m3h.csv"),
    start="2025-11-01 00:00:00", end="2025-11-03 23:59:00",
    basename="state_matching_V1",
)
V10_CONFIG = dict(
    version="V10", n_states=5, show_pairs=[0, 1],
    archive_path=os.path.join(RESULTS_DIR, "merged_state_features_V7_N5.csv"),
    data_path=os.path.join(DATA_DIR, "OUTLET4-11.1-1.2-CLEANED-m3h.csv"),
    start="2025-11-22 00:00:00", end="2025-11-30 23:59:00",
    basename="state_matching_V10",
)
V20_CONFIG = dict(
    version="V20", n_states=5, show_pairs=None,
    archive_path=os.path.join(RESULTS_DIR, "merged_state_features_V10_Reconstructed_N5.csv"),
    data_path=os.path.join(DATA_DIR, "OUTLET4-11.1-1.2-CLEANED-m3h.csv"),
    start="2025-12-22 00:00:00", end="2025-12-30 23:59:00",
    basename="state_matching_V20",
)

CONFIG = V1_CONFIG   # <<< set the cycle to plot

# ---------- constants ----------
MATCHING_ALPHA = 0.667
N_HOURS = 24
FEATURE_COLS = ['Temp', 'Cond', 'pH', 'NH3', 'Flow', 'Turb']
BREAK_THRESHOLD = pd.Timedelta(minutes=30)
CONFIDENCE_THRESHOLD = 0.65
PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A', '#8A8CBD', '#FECEA0', '#E6BEC7']
FEATURE_LABELS = {
    'Temp': 'Temperature \n(°C)',
    'Cond': 'EC\n(µS/cm)',
    'pH':   'pH',
    'NH3':  'NH₃-N (mg/L)',
    'Flow': 'Flow Rate\n(m³/h)',
    'Turb': 'Turbidity\n(NTU)',
}
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 20,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.weight': 'bold',
    'axes.labelsize': 24, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
})


# ---------- pipeline ----------
def load_baseline_archive(fp):
    df = pd.read_csv(fp)
    tc = [f'Hour_{i}' for i in range(N_HOURS)]
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
    df = df.sort_index()
    diffs = df.reset_index()['Datetime'].diff()
    bps = diffs[diffs > thr].index
    lengths, s = [], 0
    for bp in bps:
        if bp - s > 0: lengths.append(bp - s)
        s = bp
    if len(df) - s > 0: lengths.append(len(df) - s)
    return df.values, lengths


def fit_local_hmm(X, lengths, ns):
    m = hmm.GaussianHMM(n_components=ns, covariance_type="full", n_iter=100, random_state=42)
    m.fit(X, lengths); return m


def extract_local_time_dists(df_dt, seq, ns):
    t = df_dt.copy(); t['s'] = seq; t['hour'] = t.index.hour
    out = []
    for i in range(ns):
        sd = t[t['s'] == i]
        out.append(np.zeros(N_HOURS) if sd.empty
                   else sd['hour'].value_counts(normalize=True).reindex(range(N_HOURS), fill_value=0.0).values)
    return np.array(out)


def get_state_mapping_combined(lm, bm, lt, bt, alpha, ns):
    s = MinMaxScaler(); ln = s.fit_transform(lm); bn = s.transform(bm)
    dwq = cdist(ln, bn, 'euclidean'); dt = cdist(lt, bt, 'euclidean')
    dtot = alpha * MinMaxScaler().fit_transform(dwq) + (1 - alpha) * MinMaxScaler().fit_transform(dt)
    r, c = linear_sum_assignment(dtot)
    mp = np.zeros(ns, dtype=int); mp[r] = c
    return mp, dtot


def check_matching_confidence(dm, mp, thr):
    ev, cf = {}, {}
    for ls, bs in enumerate(mp):
        c = 1.0 / (1.0 + dm[ls, bs]); cf[ls] = c
        if c < thr: ev[bs] = c
    return ev, cf


# ---------- plotting ----------
def _add_panel_separators(fig, left_axes, right_axes):
    boxes = [ax.get_position() for ax in left_axes + right_axes]
    x0 = min(b.x0 for b in boxes) - 0.12
    x1 = max(b.x1 for b in boxes) + 0.01
    y0 = min(b.y0 for b in boxes) - 0.02
    y1 = max(b.y1 for b in boxes) + 0.07
    kw = dict(color='#B0B0B0', linewidth=1.15, linestyle='-',
              transform=fig.transFigure, figure=fig, zorder=0, clip_on=False)
    lbox = left_axes[0].get_position(); rbox = right_axes[0].get_position()
    xmid = 0.5 * (lbox.x1 + rbox.x0)
    fig.add_artist(Line2D([xmid, xmid], [y0, y1], **kw))
    for i in range(len(left_axes) - 1):
        y_top = min(left_axes[i].get_position().y0, right_axes[i].get_position().y0)
        y_bot = max(left_axes[i + 1].get_position().y1, right_axes[i + 1].get_position().y1)
        ymid = 0.5 * (y_top + y_bot)
        fig.add_artist(Line2D([x0, x1], [ymid, ymid], **kw))


def visualize_state_matching(lm_phys, ltimes, bm_phys, btimes, mapping, cf,
                             ns, version, show_pairs, basename):
    if show_pairs is None:
        show_pairs = list(range(ns))
    n_show = len(show_pairs)
    sc = MinMaxScaler(); sc.fit(np.vstack([lm_phys, bm_phys]))
    ln = sc.transform(lm_phys); bn = sc.transform(bm_phys)
    L = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']

    fig = plt.figure(figsize=(22, n_show * 6.0))
    gs = GridSpec(n_show, 2, figure=fig, width_ratios=[1.25, 0.90],
                  left=0.06, right=0.97, top=0.93, bottom=0.06, wspace=0, hspace=0.50)
    N = len(FEATURE_COLS)
    angles = [k / float(N) * 2 * pi for k in range(N)]; angles += angles[:1]
    left_axes, right_axes = [], []

    for row, ls in enumerate(show_pairs):
        bs = mapping[ls]; conf = cf[ls]
        # radar
        axr = fig.add_subplot(gs[row, 0], projection='polar')
        left_axes.append(axr)
        axr.set_theta_offset(pi / 2); axr.set_theta_direction(-1)
        axr.set_xticks(angles[:-1])
        axr.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_COLS], fontsize=22, fontweight='bold')
        axr.tick_params(pad=24); axr.set_yticklabels([])
        for i, vals in enumerate([bn[bs], ln[ls]]):
            vc = np.append(vals, vals[:1])
            axr.plot(angles, vc, linewidth=2.5, color=PALETTE[i],
                     label=[f'Baseline {bs} ', f'Local {ls} ({version})'][i])
            axr.fill(angles, vc, alpha=0.20, color=PALETTE[i])
        axr.legend(loc='upper right', bbox_to_anchor=(0.12, 0.12), fontsize=22)
        axr.text(-0.14, 1.22, L[row * 2], transform=axr.transAxes, fontsize=28,
                 fontweight='bold', va='top', ha='left', color='#333333')
        # bars
        axb = fig.add_subplot(gs[row, 1])
        right_axes.append(axb)
        x = np.arange(N_HOURS); w = 0.32
        for i, data in enumerate([btimes[bs], ltimes[ls]]):
            axb.bar(x + (i - 0.5) * w, data, w, alpha=0.82, color=PALETTE[i],
                    edgecolor='white', linewidth=0.3,
                    label=[f'Baseline {bs} ', f'Local {ls} ({version})'][i])
        axb.set_ylabel('Frequency', fontsize=28, fontweight='bold')
        axb.set_xlabel('Hour of day', fontsize=28, fontweight='bold')
        tl = np.arange(0, 24, 4); axb.set_xticks(tl); axb.set_xticklabels(tl, fontsize=22)
        axb.tick_params(axis='y', labelsize=22)
        axb.spines['top'].set_visible(False); axb.spines['right'].set_visible(False)
        axb.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
        axb.set_axisbelow(True)
        axb.legend(loc='upper left', fontsize=22, framealpha=0.35)
        ccolor = '#B6766A' if conf < CONFIDENCE_THRESHOLD else '#4E6597'
        axb.text(0.97, 0.95, f'Confidence: {conf:.2f}', transform=axb.transAxes,
                 fontsize=24, fontweight='bold', color=ccolor, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=ccolor, linewidth=1.5, alpha=0.90))
        axb.text(-0.12, 1.18, L[row * 2 + 1], transform=axb.transAxes, fontsize=28,
                 fontweight='bold', va='top', ha='left', color='#333333')

    fig.canvas.draw()
    _add_panel_separators(fig, left_axes, right_axes)
    png = os.path.join(FIG_SAVE_DIR, basename + ".png")
    pdf = os.path.join(FIG_SAVE_DIR, basename + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


def run(cfg):
    bwq, bt = load_baseline_archive(cfg['archive_path'])
    ds, dp, sc = load_and_prep_new_data(cfg['data_path'], cfg['start'], cfg['end'])
    X, lengths = get_data_chunks(ds[FEATURE_COLS], BREAK_THRESHOLD)
    m = fit_local_hmm(X, lengths, cfg['n_states'])
    lm = sc.inverse_transform(m.means_)
    seq = m.predict(X, lengths)
    lt = extract_local_time_dists(dp, seq, cfg['n_states'])
    mp, dm = get_state_mapping_combined(lm, bwq, lt, bt, MATCHING_ALPHA, cfg['n_states'])
    ev, cf = check_matching_confidence(dm, mp, CONFIDENCE_THRESHOLD)
    visualize_state_matching(lm, lt, bwq, bt, mp, cf,
                             cfg['n_states'], cfg['version'], cfg['show_pairs'], cfg['basename'])


if __name__ == "__main__":
    run(CONFIG)
