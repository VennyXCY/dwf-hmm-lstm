"""
plot_posterior_segments.py
==========================
Phase 1 figure: per-segment HMM posterior probabilities over time, drawn with
broken time axes so that segments separated by long data gaps share one row.

Input : results/posterior_all_for_clustering.csv  (from build_baseline_archive.py)
Output: results/posterior_probabilities_segments.{png,pdf}

Environment: dwf-hmm (see requirements-hmm.txt).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

RESULTS_DIR = "results"
INPUT_CSV = os.path.join(RESULTS_DIR, "posterior_all_for_clustering.csv")
FIG_BASENAME = "posterior_probabilities_segments"

PALETTE = ['#2D4F8E', '#6557A5', '#9B7DC0', '#E8971E', '#D4562B', '#963028', '#C45E7A']

# Layout (physical inches)
ROW_HEIGHT = 1.8       # height of each segment row
GAP_HEIGHT = 0.5       # vertical gap between rows
MARGIN_TOP = 0.4
MARGIN_BOTTOM = 0.8
GAP_THRESHOLD = pd.Timedelta('2 days')   # split a row into panels at gaps larger than this
BUFFER = pd.Timedelta('12h')             # padding on each side of a panel


def draw_break(ax, side):
    """Draw a diagonal axis-break marker on the given side of an axis."""
    break_d, break_h, break_gap = 0.022, 0.06, 0.018
    kw = dict(transform=ax.transAxes, color='#444444', linewidth=1.5,
              clip_on=False, zorder=10)
    x0 = 0.0 if side == 'left' else 1.0
    for offset in (-break_gap / 2, break_gap / 2):
        cx = x0 + offset
        ax.plot([cx - break_d, cx + break_d], [-break_h, break_h], **kw)


def main():
    plt.rcParams.update({'font.family': 'Times New Roman'})
    posterior_all = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])

    # Display segments as consecutive indices (0, 1, 2, ...) even if some raw
    # segment ids were dropped by the minimum-duration filter upstream.
    unique_segments = sorted(posterior_all['Segment'].unique())
    seg_display = {seg: f'Segment {i}' for i, seg in enumerate(unique_segments)}
    n_segs = len(unique_segments)

    # Split the full time axis into panels at large data gaps.
    times = posterior_all['Datetime'].sort_values().reset_index(drop=True)
    gaps = times.diff()
    gap_pos = gaps[gaps > GAP_THRESHOLD].index.tolist()
    starts, ends = [times.iloc[0]], []
    for pos in gap_pos:
        ends.append(times.iloc[pos - 1])
        starts.append(times.iloc[pos])
    ends.append(times.iloc[-1])
    windows = [(s - BUFFER, e + BUFFER) for s, e in zip(starts, ends)]
    n_windows = len(windows)

    spans = [(e - s).total_seconds() for s, e in windows]
    width_ratios = [s / max(spans) for s in spans]

    fig_height = (MARGIN_TOP + MARGIN_BOTTOM + n_segs * ROW_HEIGHT
                  + (n_segs - 1) * GAP_HEIGHT)
    gs = gridspec.GridSpec(
        n_segs, n_windows, width_ratios=width_ratios,
        hspace=GAP_HEIGHT / ROW_HEIGHT, wspace=0.06,
        left=0.09, right=0.97,
        top=1.0 - MARGIN_TOP / fig_height, bottom=MARGIN_BOTTOM / fig_height)
    fig = plt.figure(figsize=(14, fig_height))

    for row_idx, seg in enumerate(unique_segments):
        seg_df = posterior_all[posterior_all['Segment'] == seg]
        state_cols = [c for c in seg_df.columns
                      if c.startswith('State_') and not seg_df[c].isnull().all()]

        # Centered row title
        row_ax = fig.add_subplot(gs[row_idx, :], frameon=False)
        row_ax.set_xticks([]); row_ax.set_yticks([])
        row_ax.annotate(seg_display[seg], xy=(0.5, 1.0), xycoords='axes fraction',
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', va='bottom', fontsize=13,
                        fontweight='bold', color='#333333')

        is_last = (row_idx == n_segs - 1)
        for win_idx, (w_start, w_end) in enumerate(windows):
            ax = fig.add_subplot(gs[row_idx, win_idx])
            for j, state in enumerate(state_cols):
                ax.plot(seg_df['Datetime'], seg_df[state],
                        color=PALETTE[j % len(PALETTE)], linewidth=1.2,
                        alpha=0.9, label=state.replace('_', ' '))

            ax.set_xlim(w_start, w_end)
            span_days = (w_end - w_start).days
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, span_days // 4)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if not is_last:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=10, labelcolor='#333333')
                for lbl in ax.get_xticklabels():
                    lbl.set_fontweight('bold')

            ax.set_ylim(-0.05, 1.1)
            ax.set_yticks([0, 0.5, 1.0])
            if win_idx == 0:
                ax.set_ylabel('Posterior\nprobability', fontsize=11, fontweight='bold')
            else:
                ax.tick_params(labelleft=False)
                ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)

            # Legend once per row, placed where the data leaves room
            if is_last and win_idx == 0:
                ax.legend(loc='upper left', fontsize=10, frameon=True,
                          framealpha=0.8, edgecolor='#cccccc', handlelength=1.0)
            elif not is_last and win_idx == n_windows - 1:
                ax.legend(loc='upper right', fontsize=10, frameon=True,
                          framealpha=0.8, edgecolor='#cccccc', handlelength=1.0)

            if win_idx > 0:
                draw_break(ax, 'left')
            if win_idx < n_windows - 1:
                draw_break(ax, 'right')

    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12, fontweight='bold')

    png = os.path.join(RESULTS_DIR, FIG_BASENAME + ".png")
    pdf = os.path.join(RESULTS_DIR, FIG_BASENAME + ".pdf")
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(png, dpi=600, **kw)
    fig.savefig(pdf, **kw)
    print(f"Saved:\n  {png}\n  {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
