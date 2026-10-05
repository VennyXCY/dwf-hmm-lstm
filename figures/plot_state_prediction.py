"""
plot_state_prediction.py   (environment: any)
===========================================================================
State-prediction figure for a single update cycle: (a) the predicted dominant
baseline state over time and (b) the predicted posterior probabilities, with
Evolved States shaded/dashed. Reads the saved LSTM inference results produced by
phase 2 (part3_rolling_inference.py).

Select the cycle with VERSION below. Reads from
results/HMM_LSTM_Model_<VERSION>/ and writes to results/.
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_SAVE_DIR = RESULTS_DIR

VERSION = "V4"   # <<< set the cycle to plot (must have been run through part3)

PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A', '#8A8CBD', '#FECEA0', '#E6BEC7']
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 22,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.weight': 'bold',
    'axes.labelsize': 26, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
    'legend.fontsize': 22, 'lines.linewidth': 2.0,
})
FS_YLABEL, FS_XLABEL, FS_TICK, FS_LEG, FS_PANEL, FS_TITLE = 28, 28, 24, 22, 32, 36


def load_cycle(version):
    """Load one cycle's inference results and evolved-state map from
    results/HMM_LSTM_Model_<version>/ (written by phase-2 part3/part1)."""
    md = os.path.join(RESULTS_DIR, f"HMM_LSTM_Model_{version}")
    cands = sorted(glob.glob(os.path.join(md, "inference_results_*.csv")))
    if not cands:
        raise FileNotFoundError(f"No inference_results_*.csv in {md}. Run part3 first.")
    df = pd.read_csv(cands[0], parse_dates=["Datetime"], index_col="Datetime")
    idxs = [int(re.match(r"State_(\d+)_Prob", c).group(1))
            for c in df.columns if re.match(r"State_(\d+)_Prob", c)]
    ns = max(idxs) + 1
    try:
        ev = joblib.load(os.path.join(md, "evolved_map.joblib"))
    except FileNotFoundError:
        ev = {}
    print(f"[{version}] {os.path.basename(cands[0])} | n_states={ns} | evolved={list(ev.keys())}")
    return dict(dfp=df, ns=ns, version=version, ev=ev)


def draw_dominant(ax, res, panel):
    dfp, ns, ev, ver = res['dfp'], res['ns'], res['ev'], res['version']
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
    ax.legend(handles=handles, loc='upper left', fontsize=FS_LEG,
              frameon=True, framealpha=0.55, edgecolor='#cccccc')
    ax.set_ylabel('Baseline State ID', fontsize=FS_YLABEL, fontweight='bold')
    ax.set_yticks(range(ns)); ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7); ax.set_axisbelow(True)
    ax.set_title(f"Prediction at {ver}", fontsize=FS_TITLE, fontweight='bold', pad=18, color='#333333')
    ax.text(-0.045, 1.11, panel, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='top', ha='left', color='#333333')


def draw_posterior(ax, res, panel):
    dfp, ns, ev, ver = res['dfp'], res['ns'], res['ev'], res['version']
    for i in range(ns):
        is_e = i in ev
        lab = f'Evolved State {i}' if is_e else f'Baseline State {i}'
        ax.plot(dfp.index, dfp[f'State_{i}_Prob'], label=lab, linewidth=2.0,
                color=PALETTE[i % len(PALETTE)], linestyle='--' if is_e else 'solid')
    ax.set_ylabel('Probability', fontsize=FS_YLABEL, fontweight='bold')
    ax.set_xlabel('Datetime', fontsize=FS_XLABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='upper right', fontsize=FS_LEG, frameon=True, framealpha=0.55, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7); ax.set_axisbelow(True)
    ax.text(-0.045, 1.11, panel, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='top', ha='left', color='#333333')


def run(version):
    res = load_cycle(version)
    fig = plt.figure(figsize=(26, 12))
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.10, hspace=0.16)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    draw_dominant(ax_a, res, '(a)')
    draw_posterior(ax_b, res, '(b)')
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax_b.tick_params(axis='x', labelsize=FS_TICK)
    idx = res['dfp'].index
    ax_b.set_xlim(idx.min(), idx.max().ceil('D'))

    base = f"state_prediction_{version}"
    png = os.path.join(FIG_SAVE_DIR, base + ".png")
    pdf = os.path.join(FIG_SAVE_DIR, base + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    run(VERSION)
