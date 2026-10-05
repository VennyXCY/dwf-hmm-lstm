"""
plot_elbow_dendrogram.py
========================
Phase 1 figure (Fig. 4): (a) merge-distance elbow used to choose the number of
merged baseline states, and (b) the Ward dendrogram of the per-state feature
vectors. The clustering is recomputed here from the saved intermediate tables
so the figure is self-contained; it reproduces exactly the merging performed in
build_baseline_archive.py.

Input : results/posterior_all_for_clustering.csv
        results/state_means_all_for_clustering.csv
Output: results/elbow_and_dendrogram.{png,pdf}

Environment: dwf-hmm (see requirements-hmm.txt).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

RESULTS_DIR = "results"
POSTERIOR_CSV = os.path.join(RESULTS_DIR, "posterior_all_for_clustering.csv")
STATE_MEANS_CSV = os.path.join(RESULTS_DIR, "state_means_all_for_clustering.csv")
FIG_BASENAME = "elbow_and_dendrogram"

FEATURE_NAMES = ['Temp', 'NH3', 'Turb', 'Cond', 'pH', 'Flow']
TIME_PCA_COMPONENTS = 3
MAX_CLUSTERS = 8
CLUSTER_PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A',
                   '#8A8CBD', '#FECEA0', '#E6BEC7']


def build_state_features(posterior_all, state_means_all):
    """Reproduce the per-state feature matrix and sample-count weights used for
    hierarchical clustering (means + hour-of-day PCA)."""
    posterior_all['Hour'] = pd.to_datetime(posterior_all['Datetime']).dt.hour
    time_dist = (posterior_all.groupby(['Segment', 'PredictedState'])['Hour']
                 .value_counts(normalize=True).unstack(fill_value=0))
    for h in range(24):
        if h not in time_dist.columns:
            time_dist[h] = 0
    time_dist = time_dist.reindex(sorted(time_dist.columns), axis=1)
    time_cols = [f'Hour_{i}' for i in range(24)]
    time_dist.columns = time_cols

    feat = pd.merge(state_means_all, time_dist, how='left',
                    left_on=['Segment', 'State'], right_index=True)
    feat[time_cols] = feat[time_cols].fillna(0)

    counts = (posterior_all.groupby(['Segment', 'PredictedState'])
              .size().reset_index(name='SampleCount'))
    feat = pd.merge(feat, counts, how='left', left_on=['Segment', 'State'],
                    right_on=['Segment', 'PredictedState'])
    feat['SampleCount'] = feat['SampleCount'].fillna(0)
    weights = feat['SampleCount'] / feat['SampleCount'].max()
    weights[weights == 0] = 0.01

    n_time_pca = min(TIME_PCA_COMPONENTS, len(feat), len(time_cols))
    time_pca_cols = []
    if n_time_pca >= 1:
        time_pca = PCA(n_components=n_time_pca).fit_transform(feat[time_cols])
        time_pca_cols = [f'Time_PC{i + 1}' for i in range(n_time_pca)]
        feat = pd.concat([feat, pd.DataFrame(time_pca, columns=time_pca_cols,
                                             index=feat.index)], axis=1)

    x = feat[FEATURE_NAMES + time_pca_cols]
    x_scaled = StandardScaler().fit_transform(x)
    if time_pca_cols:
        idx = [x.columns.get_loc(c) for c in time_pca_cols]
        x_scaled[:, idx] *= weights.values[:, np.newaxis]
    return feat, x_scaled


def main():
    plt.rcParams.update({'font.family': 'Times New Roman'})
    posterior_all = pd.read_csv(POSTERIOR_CSV)
    state_means_all = pd.read_csv(STATE_MEANS_CSV)

    feat, x_scaled = build_state_features(posterior_all, state_means_all)
    linked = linkage(x_scaled, method='ward')

    merge_d = linked[:, 2]
    elbow_idx = int(np.argmax(np.diff(merge_d)))
    suggested_k = len(merge_d) - elbow_idx
    n_clusters = max(1, min(suggested_k, MAX_CLUSTERS, len(x_scaled)))

    # Display segments as consecutive indices (0, 1, 2, ...) in the dendrogram.
    remap = {seg: i for i, seg in enumerate(sorted(feat['Segment'].unique()))}
    display_labels = [f"{remap[s]}_{st}"
                      for s, st in zip(feat['Segment'], feat['State'])]

    fig_height_dendro = max(3, len(feat) * 0.32)
    fig = plt.figure(figsize=(10, 4.5 + fig_height_dendro))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, fig_height_dendro / 4.5],
                           left=0.10, right=0.97, top=0.96, bottom=0.07,
                           hspace=0.22)

    # (a) Elbow
    ax_e = fig.add_subplot(gs[0, 0])
    ax_e.plot(range(1, len(merge_d) + 1), merge_d, color='#4E6597',
              linewidth=1.5, marker='o', markersize=4, markerfacecolor='#4E6597')
    ax_e.axvline(x=elbow_idx + 1, color='#B6766A', linestyle='--', linewidth=1.4,
                 label=f'Suggested $k$ = {suggested_k}')
    ax_e.set_xlabel('Merging step', fontsize=12)
    ax_e.set_ylabel('Merge distance', fontsize=12)
    ax_e.tick_params(axis='both', labelsize=11)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax_e.set_axisbelow(True)
    ax_e.legend(fontsize=10, frameon=True, framealpha=0.85, edgecolor='#cccccc',
                handlelength=1.5)
    ax_e.text(-0.08, 1.06, '(a)', transform=ax_e.transAxes, fontsize=14,
              fontweight='bold', va='top', ha='left')

    # (b) Dendrogram
    ax_d = fig.add_subplot(gs[1, 0])
    set_link_color_palette(CLUSTER_PALETTE[:n_clusters])
    dendrogram(Z=linked, labels=display_labels, ax=ax_d, leaf_rotation=0,
               leaf_font_size=10, orientation='right',
               color_threshold=linked[-(n_clusters - 1), 2],
               above_threshold_color='#aaaaaa')
    set_link_color_palette(None)
    ax_d.set_xlabel('Distance', fontsize=12)
    ax_d.tick_params(axis='both', labelsize=10)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax_d.set_axisbelow(True)
    ax_d.text(-0.08, 1.04, '(b)', transform=ax_d.transAxes, fontsize=14,
              fontweight='bold', va='top', ha='left')

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw)
    fig.savefig(pdf, **kw)
    print(f"Elbow-suggested k = {suggested_k} (using {n_clusters}).")
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
