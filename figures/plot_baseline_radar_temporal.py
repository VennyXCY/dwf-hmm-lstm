"""
plot_baseline_radar_temporal.py
===============================
Fig. 5 - Spatiotemporal characteristics of the baseline states:
(a) hydro-chemical/hydraulic fingerprints (per-feature radar, raw values
    annotated, min-max normalized across states for shape), and
(b) hour-of-day occurrence patterns (one bar chart per state).

Input : results/merged_state_features_physical_and_time.csv  (from phase 1)
Output: results/radar_and_temporal.{png,pdf}

Note: for a sparse state, the archive may contain slightly negative feature
means (a PCA-reconstruction artifact). They are clamped to 0 here for display
only; the pipeline archive itself is left unchanged.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
ARCHIVE_CSV = os.path.join(RESULTS_DIR, "merged_state_features_physical_and_time.csv")
FIG_BASENAME = "radar_and_temporal"

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 20,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.weight': 'bold',
    'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
})

PALETTE = ['#4E6597', '#EFA485', '#B7A9CD', '#B6766A']
RADAR_PALETTE = ['#4E6597', '#8A8CBD', '#B7A9CD', '#FECEA0', '#EFA485', '#B6766A']
PHYS_FEATURES = ['Temp', 'NH3', 'Turb', 'Cond', 'pH', 'Flow']
PHYS_LABELS = {
    'Temp': 'Water Temperature (°C)',
    'NH3': 'NH₃-N (mg/L)',
    'Turb': 'Turbidity (NTU)',
    'Cond': 'EC (µS/cm)',
    'pH': 'pH',
    'Flow': 'Flow Rate (m³/h)',
}


def main():
    df = pd.read_csv(ARCHIVE_CSV)
    if df['MergedState_final'].min() == 1:
        df['MergedState_final'] = df['MergedState_final'] - 1
    # Clamp physically-impossible negative means to 0 (display only).
    df[PHYS_FEATURES] = df[PHYS_FEATURES].clip(lower=0)

    time_cols = [f'Hour_{i}' for i in range(24)]
    hours = np.arange(24)
    states = sorted(df['MergedState_final'].unique())
    num_states = len(states)
    angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(16, 20))
    outer_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.3, 1],
                                 hspace=0.13, left=0.07, right=0.97, top=0.97, bottom=0.05)
    inner_gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0],
                                                  wspace=0.16, hspace=0.40)
    inner_gs_b = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1],
                                                  wspace=0.22, hspace=0.40)

    # (a) radar
    axes_a = []
    for i, feature in enumerate(PHYS_FEATURES):
        ax = fig.add_subplot(inner_gs_a[i // 3, i % 3], polar=True)
        axes_a.append(ax)
        color = RADAR_PALETTE[i % len(RADAR_PALETTE)]
        vals_raw = np.array([df.loc[df['MergedState_final'] == s, feature].values[0] for s in states])
        vmin, vmax = vals_raw.min(), vals_raw.max()
        vals_norm = (vals_raw - vmin) / (vmax - vmin + 1e-9)
        vals_norm_closed = vals_norm.tolist() + [vals_norm[0]]
        ax.plot(angles, vals_norm_closed, color=color, linewidth=2.2, linestyle='solid')
        ax.fill(angles, vals_norm_closed, color=color, alpha=0.30)
        ax.plot(angles, vals_norm_closed, 'o', color=color, markersize=7)
        for j, (ang, raw_val) in enumerate(zip(angles[:-1], vals_raw)):
            ax.text(ang, vals_norm[j] + 0.16, f'{raw_val:.1f}', ha='center', va='center',
                    fontsize=20, fontweight='bold', color='#444444')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'S{s}' for s in states], fontsize=20, fontweight='bold')
        ax.set_yticklabels([]); ax.set_ylim(0, 1.42)
        ax.set_title(PHYS_LABELS[feature], fontsize=26, fontweight='bold', pad=16, color=color)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.7)
        ax.spines['polar'].set_linewidth(0.6)
    fig.text(0.01, 0.975, '(a)', fontsize=28, fontweight='bold', va='top', ha='left')

    # divider between the two radar rows
    row0, row1 = axes_a[0:3], axes_a[3:6]
    mid_y = (min(ax.get_position().y0 for ax in row0) + max(ax.get_position().y1 for ax in row1)) / 2
    left_x = min(ax.get_position().x0 for ax in axes_a) - 0.02
    right_x = max(ax.get_position().x1 for ax in axes_a) + 0.02
    fig.add_artist(mlines.Line2D([left_x, right_x], [mid_y, mid_y], transform=fig.transFigure,
                                 color='#999999', linewidth=1.2, linestyle='-'))

    # (b) hour-of-day bars
    for i, state in enumerate(states):
        ax = fig.add_subplot(inner_gs_b[i // 2, i % 2])
        row = df[df['MergedState_final'] == state]
        if not row.empty:
            values = row[time_cols].values.flatten()
            color = PALETTE[i % len(PALETTE)]
            ax.bar(hours, values, color=color, width=0.75, alpha=0.88,
                   edgecolor='white', linewidth=0.3)
            ax.set_title(f'State {state}', fontsize=26, fontweight='bold', color=color, pad=9)
            ax.set_ylabel('Probability', fontsize=22, fontweight='bold')
            ax.set_xlabel('Hour of day', fontsize=22, fontweight='bold')
            ax.set_xticks(np.arange(0, 25, 4)); ax.set_xlim(-0.8, 23.8)
            ax.tick_params(axis='both', labelsize=20)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
            ax.set_axisbelow(True)
    fig.text(0.01, 0.455, '(b)', fontsize=28, fontweight='bold', va='top', ha='left')

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw); fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
