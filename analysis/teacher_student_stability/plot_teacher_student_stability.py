"""
plot_prediction_stability.py   (environment: any)
================================================
Fig. - Prediction stability: state-switching rate (switches/hour) across update
cycles for the standalone online HMM, the standalone LSTM, and the proposed
HMM-LSTM. Shaded columns mark the drift cycles.

Input : results/switching_all.csv  (from compute_switching_hmm.py + compute_switching_lstm.py)
Output: results/prediction_stability.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
IN_CSV = os.path.join(RESULTS_DIR, "switching_all.csv")
FIG_BASENAME = "teacher_student_stability"

DRIFT = ['V10', 'V14', 'V17']            # drift cycles
C_HMM, C_LSTM, C_PURE, C_HL = '#E16882', '#4E6597', '#8A8CBD', '#EFA485'
FS_AXIS, FS_TICK, FS_LEG = 18, 15, 13

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'mathtext.fontset': 'stix', 'axes.linewidth': 1.0,
                     'font.weight': 'bold'})


def main():
    gb = pd.read_csv(IN_CSV)
    gb['is_drift'] = gb['Version'].isin(DRIFT)
    x = np.arange(len(gb))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, isd in enumerate(gb['is_drift']):
        if isd:
            ax.axvspan(i - 0.5, i + 0.5, color=C_HL, alpha=0.28, zorder=0)
    ax.plot(x, gb['HMM_switch_ph'], 'o-', color=C_HMM, lw=2, ms=6, label='Standalone HMM')
    ax.plot(x, gb['StandaloneLSTM_switch_ph'], '^-', color=C_PURE, lw=2, ms=6, label='Standalone LSTM')
    ax.plot(x, gb['HMMLSTM_switch_ph'], 's-', color=C_LSTM, lw=2.4, ms=6, label='HMM-LSTM')
    ax.set_yscale('symlog', linthresh=0.5)
    ax.set_ylabel('Switches / hour', fontsize=FS_AXIS, fontweight='bold')
    ax.set_xlabel('Update cycle', fontsize=FS_AXIS, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gb['Version'], rotation=45, fontsize=FS_TICK, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, ls='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=C_HL, alpha=0.28, label='Drift cycle'))
    ax.legend(handles=handles, prop={'weight': 'bold', 'size': FS_LEG},
              frameon=False, loc='upper left', ncol=2)

    fig.tight_layout()
    fig.canvas.draw()
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold'); lbl.set_fontsize(FS_TICK)

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
