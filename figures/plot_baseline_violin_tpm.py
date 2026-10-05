"""
plot_baseline_violin_tpm.py
===========================
Fig. 6 - Analysis of the initial baseline states:
(a) violin plots of the six water-quality/hydraulic parameters per baseline
    state, and (b) the empirical transition probability matrix (TPM) between
    states.

Inputs (from phase 1, under results/):
    outlet_phase1_cleaned.csv
    posterior_all_for_clustering.csv
    state_features_and_merged_clusters.csv
Output: results/advanced_visualizations.{png,pdf}
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_BASENAME = "advanced_visualizations"

CLEAN_CSV = os.path.join(RESULTS_DIR, "outlet_phase1_cleaned.csv")
POSTERIOR_CSV = os.path.join(RESULTS_DIR, "posterior_all_for_clustering.csv")
CLUSTERS_CSV = os.path.join(RESULTS_DIR, "state_features_and_merged_clusters.csv")

FEATURE_NAMES = ['Temp', 'NH3', 'Turb', 'Cond', 'pH', 'Flow']
AXIS_LABELS = {
    'Temp': 'Water Temperature (°C)',
    'NH3':  'NH₃-N (mg/L)',
    'Turb': 'Turbidity (NTU)',
    'Cond': 'EC (µS/cm)',
    'pH':   'pH',
    'Flow': 'Flow Rate (m³/h)',
}
PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A', '#8A8CBD', '#FECEA0', '#E6BEC7']
AXIS_LABEL_FS, TICK_FS = 20, 17


def main():
    df_clean = pd.read_csv(CLEAN_CSV, parse_dates=["Datetime"])
    df_post = pd.read_csv(POSTERIOR_CSV, parse_dates=["Datetime"])
    df_clusters = pd.read_csv(CLUSTERS_CSV)

    # map (Segment, State) -> MergedState_final onto every timestep
    mp = df_clusters[['Segment', 'State', 'MergedState_final']].rename(columns={'State': 'PredictedState'})
    df_post_mapped = pd.merge(df_post, mp, on=['Segment', 'PredictedState'], how='left')
    df_final = pd.merge(df_clean, df_post_mapped[['Datetime', 'Segment', 'MergedState_final']],
                        on=['Datetime', 'Segment'], how='inner').dropna(subset=['MergedState_final'])
    # align to paper state indices 0..3
    df_final['MergedState_final'] = df_final['MergedState_final'].map({1: 0, 2: 1, 3: 2, 4: 3})
    df_final = df_final.dropna(subset=['MergedState_final'])
    df_final['MergedState_final'] = df_final['MergedState_final'].astype(int)
    df_final = df_final.sort_values(by='MergedState_final')
    # NOTE: no unit conversion - all data are already in m3/h.

    unique_states = sorted(df_final['MergedState_final'].unique())
    state_palette = PALETTE[:len(unique_states)]
    state_labels = [f'State {s}' for s in unique_states]

    # empirical TPM (per Segment, consecutive timesteps)
    idx = {s: i for i, s in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)))
    for _, g in df_final.groupby('Segment'):
        seq = g.sort_values('Datetime')['MergedState_final'].map(idx).to_numpy()
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1
    rs = counts.sum(axis=1, keepdims=True)
    tpm = pd.DataFrame(np.divide(counts, rs, where=rs > 0),
                       index=unique_states, columns=unique_states)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 19,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.weight': 'bold',
        'axes.labelsize': 20, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    })

    fig = plt.figure(figsize=(20, 12))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.8, 1],
                                 left=0.06, right=0.97, top=0.94, bottom=0.08, wspace=0.19)
    gs_violin = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_outer[0],
                                                 hspace=0.18, wspace=0.20)

    for i, feature in enumerate(FEATURE_NAMES):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs_violin[row, col])
        sns.violinplot(data=df_final, x='MergedState_final', y=feature, ax=ax,
                       palette=state_palette, inner='quartile', linewidth=1.0,
                       density_norm='area', cut=0)
        ax.set_ylabel(AXIS_LABELS.get(feature, feature), fontsize=AXIS_LABEL_FS, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(state_labels, fontsize=TICK_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if row == 2:
            ax.set_xlabel('Baseline State', fontsize=AXIS_LABEL_FS, fontweight='bold')
    fig.text(0.01, 0.97, '(a)', fontsize=22, fontweight='bold', va='top', ha='left', color='#333333')

    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], height_ratios=[1.2, 1])
    ax_tpm = fig.add_subplot(gs_right[0])
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#B7A9CD', '#4E6597'], N=256)
    sns.heatmap(tpm, annot=True, fmt='.1%', cmap=cmap, vmin=0, vmax=1, square=True,
                cbar_kws={'label': 'Transition Probability', 'shrink': 0.85},
                xticklabels=state_labels, yticklabels=state_labels,
                linewidths=1.5, linecolor='white',
                annot_kws={'size': TICK_FS, 'weight': 'bold'}, ax=ax_tpm)
    n = tpm.shape[0]; ax_tpm.set_xlim(0, n); ax_tpm.set_ylim(n, 0)
    for sp in ax_tpm.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.6); sp.set_color('#333333'); sp.set_zorder(10)
    ax_tpm.tick_params(axis='both', which='major', length=4, width=1.0, color='#333333', pad=8, labelsize=TICK_FS)
    ax_tpm.set_xlabel('Next Baseline State $(t+1)$', fontsize=AXIS_LABEL_FS, fontweight='bold')
    ax_tpm.set_ylabel('Current Baseline State $(t)$', fontsize=AXIS_LABEL_FS, fontweight='bold')
    ax_tpm.set_xticklabels(state_labels, fontsize=TICK_FS, rotation=0)
    ax_tpm.set_yticklabels(state_labels, fontsize=TICK_FS, rotation=0)
    cbar = ax_tpm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_label('Transition Probability', fontsize=AXIS_LABEL_FS, fontweight='bold')
    ax_tpm.text(-0.25, 1.05, '(b)', transform=ax_tpm.transAxes, fontsize=22,
                fontweight='bold', va='top', ha='left', color='#333333')

    fig.canvas.draw()
    for ax in fig.axes:
        if ax.xaxis.label.get_text():
            ax.xaxis.label.set_fontweight('bold'); ax.xaxis.label.set_fontsize(AXIS_LABEL_FS)
        if ax.yaxis.label.get_text():
            ax.yaxis.label.set_fontweight('bold'); ax.yaxis.label.set_fontsize(AXIS_LABEL_FS)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold'); lbl.set_fontsize(TICK_FS)

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
