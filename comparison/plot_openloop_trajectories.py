#!/usr/bin/env python3
"""
plot_openloop_trajectories.py

Plots the planned reference trajectory first, followed by the three open-loop
trajectories (Case 1: Mass Symmetry, Case 2: Pseudo-Flatness, Case 3: Fictitious-Input).
Outputs a single academic publication-quality vector PDF and high-res PNG.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Publication-grade typography and style settings
plt.rcParams.update({
    'font.size': 10.5,
    'axes.labelsize': 11.0,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
})

# Curated academic color palette
COLOR_PLANNED = '#B22222'      # Firebrick / Crimson for reference
COLOR_CASE1 = '#1E40AF'        # Sapphire Blue (Mass Symmetry)
COLOR_CASE2 = '#047857'        # Emerald Green (Pseudo-Flatness)
COLOR_CASE3 = '#D97706'        # Dark Amber (Fictitious-Input)
COLOR_WP = '#FACC15'           # Yellow for waypoints
COLOR_START = '#10B981'        # Green for start
COLOR_GOAL = '#DC2626'         # Red for goal

def clean_outliers(signal, max_value=100.0):
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

def load_data():
    # 1. Planned Reference Trajectory
    ref_path = PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output' / 'planned_trajectory_reference.csv'
    if not ref_path.exists():
        ref_path = PROJECT_DIR / 'Mass_Symmetry' / 'RealtimeController' / 'output' / 'planned_trajectory_reference.csv'
    df_ref = pd.read_csv(ref_path)

    # 2. Waypoints
    wp_path = PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output' / 'waypoints.csv'
    if wp_path.exists():
        try:
            wp_data = np.loadtxt(wp_path, delimiter=',')
            wp_x, wp_y = wp_data[:, 0], wp_data[:, 1]
        except Exception:
            wp_x, wp_y = None, None
    else:
        wp_x, wp_y = None, None

    if wp_x is None:
        base_wp = np.array([
            [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
            [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
        ])
        wp_x, wp_y = base_wp[:, 0], base_wp[:, 1]

    # 3. Open-loop trajectories
    c1_path = PROJECT_DIR / 'Mass_Symmetry' / 'OpenloopResponse' / 'case1_openloop_results.csv'
    c2_path = PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'OpenloopResponse' / 'case2_openloop_results.csv'
    c3_path = PROJECT_DIR / 'Fictitious-Input_Full_Actuation' / 'OpenloopResponse' / 'case3_openloop_results.csv'

    df_c1 = pd.read_csv(c1_path)
    df_c2 = pd.read_csv(c2_path)
    df_c3 = pd.read_csv(c3_path)

    return df_ref, (wp_x, wp_y), df_c1, df_c2, df_c3

def plot_openloop_comparison(output_pdf=None, output_png=None):
    if output_pdf is None:
        output_pdf = SCRIPT_DIR / 'comparison_openloop_trajectories.pdf'
    if output_png is None:
        output_png = SCRIPT_DIR / 'comparison_openloop_trajectories.png'

    df_ref, (wp_x, wp_y), df_c1, df_c2, df_c3 = load_data()

    x_ref = df_ref['x_ref'].to_numpy()
    y_ref = df_ref['y_ref'].to_numpy()

    x_c1 = clean_outliers(df_c1['x_real'].to_numpy())
    y_c1 = clean_outliers(df_c1['y_real'].to_numpy())

    x_c2 = clean_outliers(df_c2['x_real'].to_numpy())
    y_c2 = clean_outliers(df_c2['y_real'].to_numpy())

    x_c3 = clean_outliers(df_c3['x_real'].to_numpy())
    y_c3 = clean_outliers(df_c3['y_real'].to_numpy())

    # Square/balanced rectangular figure (7.2 x 6.4 in)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)

    # 1. Planned reference trajectory plotted FIRST (per user instruction)
    line_plan, = ax.plot(
        x_ref, y_ref,
        color=COLOR_PLANNED,
        linestyle='-',
        linewidth=2.2,
        label=r'Planned Reference $\mathbf{p}_{\mathrm{ref}}(t)$',
        zorder=3
    )

    # 2. Then the 3 open-loop trajectories
    line_c1, = ax.plot(
        x_c1, y_c1,
        color=COLOR_CASE1,
        linestyle='-.',
        linewidth=1.5,
        label='Case 1: Mass Symmetry (Open-Loop)',
        zorder=4
    )

    line_c2, = ax.plot(
        x_c2, y_c2,
        color=COLOR_CASE2,
        linestyle='--',
        linewidth=1.7,
        label='Case 2: Pseudo-Flatness (Open-Loop)',
        zorder=5
    )

    line_c3, = ax.plot(
        x_c3, y_c3,
        color=COLOR_CASE3,
        linestyle=':',
        linewidth=1.8,
        label='Case 3: Fictitious-Input (Open-Loop)',
        zorder=4
    )

    # 3. Waypoints and Key Points
    ax.scatter(
        wp_x, wp_y,
        color=COLOR_WP,
        s=55,
        marker='o',
        edgecolors='black',
        linewidths=1.0,
        label='Waypoints',
        zorder=7
    )

    ax.scatter(
        x_ref[0], y_ref[0],
        color=COLOR_START,
        s=65,
        marker='o',
        edgecolors='black',
        linewidths=1.0,
        label='Start $(x_0, y_0)$',
        zorder=8
    )

    ax.scatter(
        x_ref[-1], y_ref[-1],
        color=COLOR_GOAL,
        s=80,
        marker='X',
        edgecolors='black',
        linewidths=1.0,
        label='Goal $(x_f, y_f)$',
        zorder=8
    )

    # Aesthetic annotations
    ax.set_xlabel(r'Position $X \ [\mathrm{m}]$')
    ax.set_ylabel(r'Position $Y \ [\mathrm{m}]$')
    ax.set_title(r'Open-Loop Response vs. Planned Reference Trajectory', fontsize=11.5, pad=10)

    ax.axis('equal')
    ax.grid(True, which='both', linestyle='--', alpha=0.55)
    ax.set_axisbelow(True)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Legend in academic box
    ax.legend(
        loc='upper left',
        frameon=True,
        edgecolor='black',
        facecolor='white',
        framealpha=0.95,
        fancybox=False,
        fontsize=8.8
    )

    fig.tight_layout()

    fig.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[OpenLoop Trajectories] Vector PDF saved to: {output_pdf}")
    print(f"[OpenLoop Trajectories] High-res PNG saved to: {output_png}")

if __name__ == '__main__':
    plot_openloop_comparison()
