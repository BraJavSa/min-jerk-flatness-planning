#!/usr/bin/env python3
"""
plot_realtime_velocities.py

Compares the 3 velocities (u, v, r) of RealtimeController across all three cases.
Produces a single vector PDF with 9 subplots (3 rows x 3 columns):
  - Columns: Case 1 (Mass Symmetry), Case 2 (Pseudo-Flatness), Case 3 (Fictitious-Input)
  - Rows: Surge velocity u [m/s], Sway velocity v [m/s], Yaw rate r [rad/s]
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
        'file': PROJECT_DIR / 'Mass_Symmetry' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_real': '#000000',
        'color_ref': '#B22222',
    },
    {
        'key': 'case2',
        'title': 'Case 2: Pseudo-Flatness',
        'file': PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_real': '#000000',
        'color_ref': '#B22222',
    },
    {
        'key': 'case3',
        'title': 'Case 3: Fictitious-Input',
        'file': PROJECT_DIR / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color_real': '#000000',
        'color_ref': '#B22222',
    },
]

def clean_outliers(signal, max_value=5.0):
    signal = np.asarray(signal, dtype=float)
    outliers = np.abs(signal) > max_value
    if not np.any(outliers):
        return signal
    cleaned = signal.copy()
    valid = np.where(~outliers)[0]
    for idx in np.where(outliers)[0]:
        if valid.size:
            nearest = valid[np.argmin(np.abs(valid - idx))]
            cleaned[idx] = signal[nearest]
    return cleaned

def plot_velocities_comparison(output_pdf=None, output_png=None):
    if output_pdf is None:
        output_pdf = SCRIPT_DIR / 'comparison_realtime_velocities.pdf'
    if output_png is None:
        output_png = SCRIPT_DIR / 'comparison_realtime_velocities.png'

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.5), dpi=300, sharex='col')

    velocities = [
        {
            'row': 0,
            'var_ref': 'u_ref',
            'var_real': 'u_real',
            'label': r'Surge Velocity $u \ [\mathrm{m/s}]$',
            'ref_label': r'$u_{\mathrm{ref}}$',
            'real_label': r'$u_{\mathrm{real}}$',
            'y_fmt': '%.2f',
        },
        {
            'row': 1,
            'var_ref': 'v_ref',
            'var_real': 'v_real',
            'label': r'Sway Velocity $v \ [\mathrm{m/s}]$',
            'ref_label': r'$v_{\mathrm{ref}}$',
            'real_label': r'$v_{\mathrm{real}}$',
            'y_fmt': '%.2f',
        },
        {
            'row': 2,
            'var_ref': 'r_ref',
            'var_real': 'r_real',
            'label': r'Yaw Rate $r \ [\mathrm{rad/s}]$',
            'ref_label': r'$r_{\mathrm{ref}}$',
            'real_label': r'$r_{\mathrm{real}}$',
            'y_fmt': '%.2f',
        }
    ]

    for col_idx, case_info in enumerate(CASES):
        df = pd.read_csv(case_info['file'])
        t = df['t'].to_numpy()

        for vel in velocities:
            row_idx = vel['row']
            ax = axes[row_idx, col_idx]

            ref = clean_outliers(df[vel['var_ref']].to_numpy())
            real = clean_outliers(df[vel['var_real']].to_numpy())

            rmse = np.sqrt(np.mean((real - ref)**2))

            line_ref, = ax.plot(
                t, ref,
                color=case_info['color_ref'],
                linestyle='-',
                linewidth=1.7,
                label=rf'Desired ({vel["ref_label"]})',
                zorder=3
            )

            line_real, = ax.plot(
                t, real,
                color=case_info['color_real'],
                linestyle='--',
                linewidth=1.2,
                label=rf'Closed-Loop ({vel["real_label"]})',
                zorder=4
            )

            ax.grid(True, linestyle='--', alpha=0.55)
            ax.set_xlim(0, t[-1])
            ax.yaxis.set_major_formatter(FormatStrFormatter(vel['y_fmt']))

            # Add RMSE annotation badge
            rmse_str = f'RMSE: {rmse:.4f}'
            ax.text(
                0.03, 0.92, rmse_str,
                transform=ax.transAxes,
                fontsize=7.8,
                verticalalignment='top',
                fontfamily='serif',
                bbox=dict(boxstyle='square,pad=0.25', facecolor='#F8FAFC', edgecolor='#94A3B8', alpha=0.9)
            )

            # Titles on top row
            if row_idx == 0:
                ax.set_title(case_info['title'], fontsize=11.0, pad=9, fontweight='medium')

            # Y-labels on leftmost column
            if col_idx == 0:
                ax.set_ylabel(vel['label'])

            # X-labels on bottom row
            if row_idx == 2:
                ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')

            # Legend on upper row, rightmost column (or in each row)
            ax.legend(loc='lower right', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.2)

    fig.suptitle('RealtimeController Velocity Tracking Performance ($u, v, r$ Comparison)', fontsize=12.5, y=0.995)
    plt.tight_layout()

    fig.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Realtime Velocities] Vector PDF saved to: {output_pdf}")
    print(f"[Realtime Velocities] High-res PNG saved to: {output_png}")

if __name__ == '__main__':
    plot_velocities_comparison()
