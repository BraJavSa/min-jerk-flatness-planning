#!/usr/bin/env python3
"""
plot_response_times.py

Compares the computation / response times (solve_ms) of the RealtimeController
across the three cases (Mass Symmetry, Pseudo-Flatness, Fictitious-Input).
Produces a single academic publication-quality vector PDF and high-res PNG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Style configuration for academic papers
plt.rcParams.update({
    'font.size': 9.5,
    'axes.labelsize': 10.0,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.9,
})

CASES = [
    {
        'key': 'case1',
        'title': 'Case 1: Mass Symmetry',
        'short_name': 'Mass Symmetry',
        'file': PROJECT_DIR / 'Mass_Symmetry' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_main': '#1E40AF',
        'color_fill': '#DBEAFE',
    },
    {
        'key': 'case2',
        'title': 'Case 2: Pseudo-Flatness',
        'short_name': 'Pseudo-Flatness',
        'file': PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_main': '#047857',
        'color_fill': '#D1FAE5',
    },
    {
        'key': 'case3',
        'title': 'Case 3: Fictitious-Input',
        'short_name': 'Fictitious-Input',
        'file': PROJECT_DIR / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_main': '#D97706',
        'color_fill': '#FEF3C7',
    },
]

T_BUDGET_MS = 1000.0 / 30.0  # 33.333 ms for 30 Hz control loop

def plot_response_times_comparison(output_pdf=None, output_png=None):
    if output_pdf is None:
        output_pdf = SCRIPT_DIR / 'comparison_response_times.pdf'
    if output_png is None:
        output_png = SCRIPT_DIR / 'comparison_response_times.png'

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=300)

    stats = []
    plot_positions = [axes[0, 0], axes[0, 1], axes[1, 0]]

    # Y-axis upper limit for all time series to share consistent scale
    y_max = 42.0

    for idx, (case_info, ax) in enumerate(zip(CASES, plot_positions)):
        df = pd.read_csv(case_info['file'])
        t = df['t'].to_numpy()
        solve_ms = df['solve_ms'].to_numpy()

        mean_ms = np.mean(solve_ms)
        max_ms = np.max(solve_ms)
        p95_ms = np.percentile(solve_ms, 95)
        std_ms = np.std(solve_ms)
        headroom_pct = (1.0 - mean_ms / T_BUDGET_MS) * 100.0

        stats.append({
            'short_name': case_info['short_name'],
            'mean': mean_ms,
            'max': max_ms,
            'p95': p95_ms,
            'std': std_ms,
            'headroom': headroom_pct,
            'color': case_info['color_main']
        })

        # Plot time series
        ax.plot(
            t, solve_ms,
            color=case_info['color_main'],
            linestyle='-',
            linewidth=0.9,
            alpha=0.85,
            label=r'Solve Time $t_{\mathrm{solve}}$',
            zorder=3
        )
        ax.fill_between(t, 0, solve_ms, color=case_info['color_fill'], alpha=0.55, zorder=2)

        # 30 Hz Budget reference line
        ax.axhline(
            T_BUDGET_MS,
            color='#DC2626',
            linestyle='--',
            linewidth=1.3,
            label=rf'$30\,\mathrm{{Hz}}\ \mathrm{{Budget}}\ ({T_BUDGET_MS:.2f}\,\mathrm{{ms}})$',
            zorder=4
        )

        # Mean and Peak lines
        ax.axhline(
            mean_ms,
            color='#0F172A',
            linestyle='-.',
            linewidth=1.2,
            label=rf'Mean: {mean_ms:.2f} ms',
            zorder=5
        )
        ax.axhline(
            max_ms,
            color='#B45309',
            linestyle=':',
            linewidth=1.2,
            label=rf'Peak: {max_ms:.2f} ms',
            zorder=5
        )

        ax.set_title(case_info['title'], fontsize=10.5, pad=7, fontweight='medium')
        ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')
        ax.set_ylabel(r'Solve Time $t_{\mathrm{solve}} \ [\mathrm{ms}]$')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        badge_txt = (
            rf"$\mathbf{{Mean:}}\ {mean_ms:.2f}\,\mathrm{{ms}}$" "\n"
            rf"$\mathbf{{Max:}}\ {max_ms:.2f}\,\mathrm{{ms}}$" "\n"
            rf"$\mathbf{{Headroom:}}\ {headroom_pct:.1f}\% $"
        )
        ax.text(
            0.03, 0.94, badge_txt,
            transform=ax.transAxes,
            fontsize=7.8,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#F8FAFC', edgecolor='#94A3B8', alpha=0.92)
        )

        ax.legend(loc='upper right', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.2)

    # 4th subplot: Comparative Bar Summary
    ax_bar = axes[1, 1]
    labels = [s['short_name'] for s in stats]
    x = np.arange(len(labels))
    bar_width = 0.25

    means = [s['mean'] for s in stats]
    p95s = [s['p95'] for s in stats]
    maxs = [s['max'] for s in stats]

    b1 = ax_bar.bar(x - bar_width, means, bar_width, label='Mean', color='#1E40AF', edgecolor='black', linewidth=0.8)
    b2 = ax_bar.bar(x,            p95s,  bar_width, label='95th %-tile', color='#047857', edgecolor='black', linewidth=0.8)
    b3 = ax_bar.bar(x + bar_width, maxs,  bar_width, label='Peak (Max)', color='#D97706', edgecolor='black', linewidth=0.8)

    ax_bar.axhline(
        T_BUDGET_MS,
        color='#DC2626',
        linestyle='--',
        linewidth=1.3,
        label=rf'$30\,\mathrm{{Hz}}\ \mathrm{{Limit}}\ ({T_BUDGET_MS:.2f}\,\mathrm{{ms}})$',
        zorder=5
    )

    ax_bar.set_title('RealtimeController Benchmark Summary', fontsize=10.5, pad=7, fontweight='medium')
    ax_bar.set_ylabel(r'Computation Time $[\mathrm{ms}]$')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=0, fontsize=8.8)
    ax_bar.set_ylim(0, y_max)
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.55)
    ax_bar.set_axisbelow(True)

    # Annotate numeric values on top of bars
    for bar_group in (b1, b2, b3):
        for rect in bar_group:
            val = rect.get_height()
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2.0,
                val + 0.8,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=7.2,
                fontfamily='serif'
            )

    ax_bar.legend(loc='upper left', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.2)

    fig.suptitle(r'Real-Time Controller Response Time ($t_{\mathrm{solve}}$) & 30 Hz Budget Headroom', fontsize=12.5, y=0.995)
    plt.tight_layout()

    fig.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Response Times] Vector PDF saved to: {output_pdf}")
    print(f"[Response Times] High-res PNG saved to: {output_png}")

if __name__ == '__main__':
    plot_response_times_comparison()
