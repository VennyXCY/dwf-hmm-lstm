"""
plot_adaptive_reconstruction_value.py   (environment: any)
=========================================================
Fig. - The value of adaptive reconstruction: static (frozen) versus adaptive
system over the Phase 2 rolling period. (a) archive-to-data matching distance,
(b) mean predictive entropy, (c) 95th-percentile KLD vs the HMM prior. Vertical
dashed lines mark cycles where reconstruction was triggered; dotted horizontal
lines mark the alert thresholds.

Inputs (results/):
    archive_data_distance.csv   (compute_matching_distance.py)
    static_history.csv          (compute_static_metrics.py)
    performance_history.csv     (adaptive run, from phase 2)
Output: results/adaptive_reconstruction_value.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DIST_CSV = os.path.join(RESULTS_DIR, "archive_data_distance.csv")
STATIC_CSV = os.path.join(RESULTS_DIR, "static_history.csv")
ADAPT_CSV = os.path.join(RESULTS_DIR, "performance_history.csv")
FIG_BASENAME = "adaptive_reconstruction_value"

H_ALERT, KLD_ALERT = 0.30, 0.10
COLORS = ['#4E6597', '#8A8CBD', '#B7A9CD', '#FECEA0', '#EFA485', '#B6766A', '#555555']
C_ADAPT, C_STATIC, C_THRESH, C_RECON = COLORS[0], COLORS[5], COLORS[4], COLORS[6]
LEG = {'weight': 'bold', 'size': 10}

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'mathtext.fontset': 'stix', 'axes.linewidth': 1.0,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11})


def main():
    dist = pd.read_csv(DIST_CSV)
    stat = pd.read_csv(STATIC_CSV)[['Model_Version', 'H_Mean', 'KLD_95']].rename(
        columns={'H_Mean': 'H_static', 'KLD_95': 'KLD_static'})
    adapt = pd.read_csv(ADAPT_CSV)
    recon_versions = adapt.loc[adapt['Reconstruction_Triggered'].astype(str).str.upper() == 'TRUE',
                               'Model_Version'].tolist()
    adapt = adapt[['Model_Version', 'H_Mean', 'KLD_95']].rename(
        columns={'H_Mean': 'H_adap', 'KLD_95': 'KLD_adap'})

    df = dist.merge(stat, on='Model_Version', how='left').merge(adapt, on='Model_Version', how='left')
    x = np.arange(len(df))
    recon_idx = [list(df['Model_Version']).index(v) for v in recon_versions
                 if v in df['Model_Version'].values]

    def recon_lines(ax, lab=False):
        for k, i in enumerate(recon_idx):
            ax.axvline(i, ls='--', color=C_RECON, lw=1.3, alpha=0.85,
                       label=('Reconstruction triggered' if (lab and k == 0) else None))

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.2), sharex=True)

    ax = axes[0]; recon_lines(ax, lab=True)
    ax.plot(x, df['Static_Dist'], 'o-', color=C_STATIC, lw=2, ms=6, label='Static')
    ax.plot(x, df['Adaptive_Dist'], 's-', color=C_ADAPT, lw=2, ms=6, label='Adaptive')
    ax.set_ylabel('Archive–data distance', fontsize=13, fontweight='bold')
    ax.legend(prop=LEG, frameon=True, framealpha=0.5, edgecolor='#cccccc', loc='lower left')

    ax = axes[1]; recon_lines(ax)
    ax.plot(x, df['H_static'], 'o-', color=C_STATIC, lw=2, ms=6, label='Static')
    ax.plot(x, df['H_adap'], 's-', color=C_ADAPT, lw=2, ms=6, label='Adaptive')
    ax.axhline(H_ALERT, ls=':', color=C_THRESH, lw=1.8, label=f'Alert threshold ({H_ALERT})')
    ax.set_ylabel('Mean predictive\nentropy', fontsize=13, fontweight='bold')
    ax.legend(prop=LEG, frameon=True, framealpha=0.5, edgecolor='#cccccc', loc='upper left')

    ax = axes[2]; recon_lines(ax)
    ax.plot(x, df['KLD_static'], 'o-', color=C_STATIC, lw=2, ms=6, label='Static')
    ax.plot(x, df['KLD_adap'], 's-', color=C_ADAPT, lw=2, ms=6, label='Adaptive')
    ax.axhline(KLD_ALERT, ls=':', color=C_THRESH, lw=1.8, label=f'Alert threshold ({KLD_ALERT})')
    ax.set_ylabel('KLD (95th pct)\nvs HMM prior', fontsize=13, fontweight='bold')
    ax.legend(prop=LEG, frameon=True, framealpha=0.5, edgecolor='#cccccc', loc='upper left')

    for ax, lab in zip(axes, ['(a)', '(b)', '(c)']):
        ax.text(-0.10, 1.03, lab, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.grid(True, ls='--', alpha=0.3); ax.tick_params(direction='out', length=4)

    axes[-1].set_xticks(x); axes[-1].set_xticklabels(df['Model_Version'], rotation=45)
    axes[-1].set_xlabel('Update cycle', fontsize=13, fontweight='bold')
    fig.tight_layout(h_pad=1.1)
    fig.canvas.draw()
    for ax in axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
