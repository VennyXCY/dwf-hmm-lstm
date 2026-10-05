"""
plot_window_length_effect.py   (environment: any)
================================================
Fig. - Effect of rolling window length on system behaviour: 3-day vs 5-day
window configurations, across update cycles. (a) mean matching confidence,
(b) mean predictive entropy, (c) 95th-percentile KLD, (d) state switching rate.
Dotted vertical lines mark cycles where reconstruction was triggered.

This reads the performance history from TWO full pipeline runs (one with a 3-day
window configuration, one with a 5-day configuration). Run the Phase 2 pipeline
twice and save each run's results/performance_history.csv as the two files below.

Inputs (results/):
    performance_history_3day.csv
    performance_history_5day.csv
Output: results/window_length_effect.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
HIST_3DAY = os.path.join(RESULTS_DIR, "performance_history_3day.csv")
HIST_5DAY = os.path.join(RESULTS_DIR, "performance_history_5day.csv")
FIG_BASENAME = "window_length_effect"

C3, C3_LIGHT, C5, C5_LIGHT = '#4E6597', '#8A8CBD', '#B6766A', '#EFA485'
FS_LABEL, FS_TICK, FS_LEGEND, FS_CAPTION = 26, 22, 17, 30

PANELS = [
    ('Conf_Mean_Raw',     'Mean matching confidence',      '(a)'),
    ('H_Mean',            'Mean predictive entropy',       '(b)'),
    ('KLD_95',            'KLD (95th pct) vs HMM prior',   '(c)'),
    ('Switches_Per_Hour', 'State switches per hour',       '(d)'),
]

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.weight': 'bold', 'font.size': 16,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.alpha': 0.32, 'grid.linestyle': '--',
    'axes.facecolor': '#FAFAFA', 'figure.facecolor': 'white',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
})


def load(path):
    df = pd.read_csv(path)
    for c in ['Conf_Mean_Raw', 'H_Mean', 'KLD_95', 'Switches_Per_Hour']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['VI'] = range(1, len(df) + 1)
    return df


def mark_recon(ax, df, color, shift):
    for _, row in df.iterrows():
        if str(row.get('Reconstruction_Triggered', 'FALSE')).upper() == 'TRUE':
            ax.axvline(row['VI'] + shift, color=color, lw=2.0, ls=':', alpha=0.9, zorder=5)


def main():
    df3, df5 = load(HIST_3DAY), load(HIST_5DAY)
    fig, axes = plt.subplots(4, 1, figsize=(20, 23), sharex=False)

    for ax, (col, title, cap) in zip(axes, PANELS):
        for df, c, cl, label, shift in [(df3, C3, C3_LIGHT, '3-Day Window', -0.12),
                                        (df5, C5, C5_LIGHT, '5-Day Window', 0.12)]:
            if col not in df.columns:
                continue
            y = df[col].values; x = df['VI'].values
            mask = np.isfinite(y); x, y = x[mask], y[mask]
            s = pd.Series(y)
            rm = s.rolling(3, center=True, min_periods=1).mean().values
            rstd = s.rolling(3, center=True, min_periods=1).std().fillna(0).values
            ax.fill_between(x, rm - 0.5 * rstd, rm + 0.5 * rstd, color=cl, alpha=0.28, zorder=1)
            ax.plot(x, y, 'o-', color=c, lw=2.4, ms=7, label=label, alpha=0.92, zorder=3)
            ax.plot(x, rm, '--', color=c, lw=1.4, alpha=0.5, zorder=2, label=f'{label} (rolling mean)')
            mark_recon(ax, df, c, shift)

        ax.set_ylabel(title, fontsize=FS_LABEL, fontweight='bold')
        ax.set_xlabel('Update cycle index', fontsize=FS_LABEL, fontweight='bold')
        ax.tick_params(labelsize=FS_TICK)
        ax.text(-0.01, 1.02, cap, transform=ax.transAxes, fontsize=FS_CAPTION,
                fontweight='bold', va='bottom', ha='left', color='#222')
        handles, _ = ax.get_legend_handles_labels()
        if cap == '(a)':
            handles += [Line2D([0], [0], color=C3, ls=':', lw=2.2, label='Reconstruction triggered (3-Day)'),
                        Line2D([0], [0], color=C5, ls=':', lw=2.2, label='Reconstruction triggered (5-Day)')]
        ax.legend(handles=handles, fontsize=FS_LEGEND, frameon=True, framealpha=0.35,
                  edgecolor='#ccc', loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0)

    plt.tight_layout(h_pad=0.6)
    plt.subplots_adjust(hspace=0.20, top=0.96)
    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
