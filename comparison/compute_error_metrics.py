#!/usr/bin/env python3
"""
compute_error_metrics.py

Computes comprehensive tracking and actuation error metrics between planned and applied
signals for all three cases (Mass Symmetry, Pseudo-Flatness, Fictitious-Input).
Saves a structured JSON file (error_metrics_summary.json) and generates a publication-grade
vector PDF (comparison_error_metrics.pdf) and high-res PNG.
"""

import json
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
        'color': '#1E40AF',
        'linestyle': '-.',
    },
    {
        'key': 'case2',
        'title': 'Case 2: Pseudo-Flatness',
        'short_name': 'Pseudo-Flatness',
        'file': PROJECT_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color': '#047857',
        'linestyle': '-',
    },
    {
        'key': 'case3',
        'title': 'Case 3: Fictitious-Input',
        'short_name': 'Fictitious-Input',
        'file': PROJECT_DIR / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'output' / 'realtime_tracking_results_30Hz.csv',
        'color': '#D97706',
        'linestyle': ':',
    },
]

T_BUDGET_MS = 1000.0 / 30.0  # 33.333 ms

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_metrics():
    summary = {}
    curves_data = []

    for c in CASES:
        df = pd.read_csv(c['file'])
        t = df['t'].to_numpy()
        n_samples = len(t)

        # 1. Position tracking error
        x_real = df['x_real'].to_numpy()
        y_real = df['y_real'].to_numpy()
        x_ref = df['x_ref'].to_numpy()
        y_ref = df['y_ref'].to_numpy()

        err_x = x_real - x_ref
        err_y = y_real - y_ref
        err_pos = np.hypot(err_x, err_y)
        cum_sq = np.cumsum(err_pos**2)
        idx_vec = np.arange(1, n_samples + 1)
        cum_rmse_pos = np.sqrt(cum_sq / idx_vec)

        rmse_pos = float(cum_rmse_pos[-1])
        mean_pos = float(np.mean(err_pos))
        max_pos = float(np.max(err_pos))
        std_pos = float(np.std(err_pos))
        final_pos = float(err_pos[-1])

        # 2. Heading error
        psi_real = df['psi_real'].to_numpy()
        psi_ref = df['psi_ref'].to_numpy()
        err_psi = wrap_angle(psi_real - psi_ref)
        rmse_psi_rad = float(np.sqrt(np.mean(err_psi**2)))
        rmse_psi_deg = float(np.rad2deg(rmse_psi_rad))
        max_psi_deg = float(np.rad2deg(np.max(np.abs(err_psi))))

        # 3. Velocity tracking errors
        u_real = df['u_real'].to_numpy()
        u_ref = df['u_ref'].to_numpy()
        v_real = df['v_real'].to_numpy()
        v_ref = df['v_ref'].to_numpy()
        r_real = df['r_real'].to_numpy()
        r_ref = df['r_ref'].to_numpy()

        rmse_u = float(np.sqrt(np.mean((u_real - u_ref)**2)))
        rmse_v = float(np.sqrt(np.mean((v_real - v_ref)**2)))
        rmse_r = float(np.sqrt(np.mean((r_real - r_ref)**2)))

        # 4. Actuation error (Applied vs Planned)
        tau_u_applied = df['tau_u_applied'].to_numpy()
        tau_u_ref = df['tau_u_ref'].to_numpy()
        tau_r_applied = df['tau_r_applied'].to_numpy()
        tau_r_ref = df['tau_r_ref'].to_numpy()

        err_tau_u = tau_u_applied - tau_u_ref
        err_tau_r = tau_r_applied - tau_r_ref

        rmse_tau_u = float(np.sqrt(np.mean(err_tau_u**2)))
        mae_tau_u = float(np.mean(np.abs(err_tau_u)))
        max_tau_u = float(np.max(np.abs(err_tau_u)))
        mean_tau_u = float(np.mean(err_tau_u))
        std_tau_u = float(np.std(err_tau_u))

        rmse_tau_r = float(np.sqrt(np.mean(err_tau_r**2)))
        mae_tau_r = float(np.mean(np.abs(err_tau_r)))
        max_tau_r = float(np.max(np.abs(err_tau_r)))
        mean_tau_r = float(np.mean(err_tau_r))
        std_tau_r = float(np.std(err_tau_r))

        # 5. Timing & Headroom
        solve_ms = df['solve_ms'].to_numpy()
        mean_solve_ms = float(np.mean(solve_ms))
        max_solve_ms = float(np.max(solve_ms))
        p95_solve_ms = float(np.percentile(solve_ms, 95))
        headroom_pct = float((1.0 - mean_solve_ms / T_BUDGET_MS) * 100.0)

        summary[c['key']] = {
            'case_name': c['title'],
            'short_name': c['short_name'],
            'samples_executed': int(n_samples),
            'duration_s': float(t[-1]),
            'position_tracking_error_m': {
                'rmse_m': rmse_pos,
                'mean_m': mean_pos,
                'max_m': max_pos,
                'std_m': std_pos,
                'final_m': final_pos
            },
            'heading_tracking_error': {
                'rmse_deg': rmse_psi_deg,
                'max_deg': max_psi_deg,
                'rmse_rad': rmse_psi_rad
            },
            'velocity_tracking_rmse': {
                'u_rmse_mps': rmse_u,
                'v_rmse_mps': rmse_v,
                'r_rmse_radps': rmse_r
            },
            'actuation_discrepancy_applied_vs_planned': {
                'tau_u_surge_N': {
                    'rmse_N': rmse_tau_u,
                    'mae_N': mae_tau_u,
                    'max_abs_N': max_tau_u,
                    'mean_N': mean_tau_u,
                    'std_N': std_tau_u
                },
                'tau_r_yaw_Nm': {
                    'rmse_Nm': rmse_tau_r,
                    'mae_Nm': mae_tau_r,
                    'max_abs_Nm': max_tau_r,
                    'mean_Nm': mean_tau_r,
                    'std_Nm': std_tau_r
                }
            },
            'realtime_computation_timing_ms': {
                'mean_ms': mean_solve_ms,
                'p95_ms': p95_solve_ms,
                'max_ms': max_solve_ms,
                'control_period_budget_ms': float(T_BUDGET_MS),
                'headroom_margin_pct': headroom_pct
            }
        }

        curves_data.append({
            'key': c['key'],
            'short_name': c['short_name'],
            'color': c['color'],
            'linestyle': c['linestyle'],
            't': t,
            'err_pos': err_pos,
            'cum_rmse_pos': cum_rmse_pos,
            'err_tau_u': err_tau_u,
            'err_tau_r': err_tau_r,
            'rmse_pos': rmse_pos,
            'max_pos': max_pos,
            'mae_tau_u': mae_tau_u,
            'mae_tau_r': mae_tau_r,
        })

    return summary, curves_data

def plot_error_metrics(curves_data, output_pdf=None, output_png=None):
    if output_pdf is None:
        output_pdf = SCRIPT_DIR / 'comparison_error_metrics.pdf'
    if output_png is None:
        output_png = SCRIPT_DIR / 'comparison_error_metrics.png'

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=300)

    # Subplot [0, 0]: Instantaneous Position Error e_pos(t)
    ax = axes[0, 0]
    for cd in curves_data:
        ax.plot(
            cd['t'], cd['err_pos'],
            color=cd['color'],
            linestyle=cd['linestyle'],
            linewidth=1.3,
            label=rf"{cd['short_name']} (Peak: {cd['max_pos']:.3f} m)",
            alpha=0.9
        )
    ax.set_title(r'Instantaneous Position Error $e_{\mathrm{pos}}(t) = \|\mathbf{p}_{\mathrm{real}} - \mathbf{p}_{\mathrm{ref}}\|$', fontsize=10.0, pad=7)
    ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')
    ax.set_ylabel(r'Position Error $[\mathrm{m}]$')
    ax.set_xlim(0, curves_data[0]['t'][-1])
    ax.set_ylim(0, 0.14)
    ax.grid(True, linestyle='--', alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.5)

    # Subplot [0, 1]: Cumulative RMSE Position
    ax = axes[0, 1]
    for cd in curves_data:
        ax.plot(
            cd['t'], cd['cum_rmse_pos'],
            color=cd['color'],
            linestyle=cd['linestyle'],
            linewidth=1.8,
            label=rf"{cd['short_name']} (Final RMSE: {cd['rmse_pos']:.4f} m)",
            alpha=0.95
        )
    ax.set_title(r'Cumulative Position Tracking RMSE $\mathrm{RMSE}_{\mathrm{pos}}(t)$', fontsize=10.0, pad=7)
    ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')
    ax.set_ylabel(r'Cumulative RMSE $[\mathrm{m}]$')
    ax.set_xlim(0, curves_data[0]['t'][-1])
    ax.set_ylim(0, 0.10)
    ax.grid(True, linestyle='--', alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='upper right', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.5)

    # Subplot [1, 0]: Surge Actuation Error e_tau_u(t)
    ax = axes[1, 0]
    for cd in curves_data:
        ax.plot(
            cd['t'], cd['err_tau_u'],
            color=cd['color'],
            linestyle=cd['linestyle'],
            linewidth=1.0,
            label=rf"{cd['short_name']} (MAE: {cd['mae_tau_u']:.2f} N)",
            alpha=0.85
        )
    ax.axhline(0, color='black', linestyle='-', linewidth=0.7, alpha=0.5)
    ax.set_title(r'Surge Force Discrepancy $e_{\tau_u}(t) = \tau_{u,\mathrm{applied}} - \tau_{u,\mathrm{ref}}$', fontsize=10.0, pad=7)
    ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')
    ax.set_ylabel(r'Error in Surge Force $[\mathrm{N}]$')
    ax.set_xlim(0, curves_data[0]['t'][-1])
    ax.grid(True, linestyle='--', alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='lower left', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.5)

    # Subplot [1, 1]: Yaw Moment Actuation Error e_tau_r(t)
    ax = axes[1, 1]
    for cd in curves_data:
        ax.plot(
            cd['t'], cd['err_tau_r'],
            color=cd['color'],
            linestyle=cd['linestyle'],
            linewidth=1.0,
            label=rf"{cd['short_name']} (MAE: {cd['mae_tau_r']:.2f} $\mathrm{{N\cdot m}}$)",
            alpha=0.85
        )
    ax.axhline(0, color='black', linestyle='-', linewidth=0.7, alpha=0.5)
    ax.set_title(r'Yaw Moment Discrepancy $e_{\tau_r}(t) = \tau_{r,\mathrm{applied}} - \tau_{r,\mathrm{ref}}$', fontsize=10.0, pad=7)
    ax.set_xlabel(r'Time $t \ [\mathrm{s}]$')
    ax.set_ylabel(r'Error in Yaw Moment $[\mathrm{N\cdot m}]$')
    ax.set_xlim(0, curves_data[0]['t'][-1])
    ax.grid(True, linestyle='--', alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='lower left', frameon=True, edgecolor='#CBD5E1', facecolor='white', framealpha=0.9, fontsize=7.5)

    fig.suptitle(r'Tracking Error & Actuation Discrepancy Performance Comparison', fontsize=12.5, y=0.995)
    plt.tight_layout()

    fig.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Error Metrics] Vector PDF saved to: {output_pdf}")
    print(f"[Error Metrics] High-res PNG saved to: {output_png}")

def main():
    summary_json_path = SCRIPT_DIR / 'error_metrics_summary.json'
    summary, curves_data = compute_metrics()

    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[Error Metrics] Structured JSON saved to: {summary_json_path}")

    plot_error_metrics(curves_data)

if __name__ == '__main__':
    main()
