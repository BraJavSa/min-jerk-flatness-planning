#!/usr/bin/env python3
"""
Academic visualization script for Real-Time Tracking Results (ROS 2 / WAM-V).
Generates a single figure with 6 comprehensive subplots:
  1. 2D Trajectory Tracking (Ideal vs Real)
  2. Surge Velocity u (Ideal vs Real)
  3. Sway Velocity v (Ideal vs Real)
  4. Yaw Rate r (Ideal vs Real)
  5. Position RMSE over Time
  6. Control Actions tau_u and tau_r (Ideal vs Realized)

Usage:
    python3 plot_realtime_results.py [tracking_csv_path] [output_png_path]
"""

import os
import sys
from pathlib import Path
import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot Aesthetics & Formatting
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5

# Color Palette
C_REF = '#2563EB'       # Royal Blue (Reference/Ideal)
C_REAL = '#DC2626'      # Crimson Red (Realized/Actual)
C_EMERALD = '#059669'   # Emerald Green (Cumulative RMSE / Torque Ref)
C_AMBER = '#D97706'     # Amber / Orange (Torque Realized)
C_GRAY = '#6B7280'      # Neutral Gray (Instantaneous error)


def clean_outliers(signal, max_value=5.0):
    """Filter transient numerical outliers as in identify_models.py."""
    signal = np.asarray(signal, dtype=float)
    outliers = np.abs(signal) > max_value
    if not np.any(outliers):
        return signal
    cleaned = signal.copy()
    valid_indices = np.where(~outliers)[0]
    for index in np.where(outliers)[0]:
        if valid_indices.size:
            nearest_valid = valid_indices[np.argmin(np.abs(valid_indices - index))]
            cleaned[index] = signal[nearest_valid]
    return cleaned


def load_tracking_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tracking CSV file not found: {csv_path}")

    data = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))

    for col in data:
        data[col] = np.array(data[col])
    return data


def plot_tracking_results(csv_path=None, output_png=None, case_title="Case 2 (6-Param Pseudo-Flatness)"):
    script_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = script_dir / 'realtime_tracking_results_30Hz.csv'
    else:
        csv_path = Path(csv_path)

    if output_png is None:
        output_png = script_dir / 'realtime_tracking_plot.png'
    else:
        output_png = Path(output_png)

    if not csv_path.exists():
        print(f"[Warning] {csv_path} does not exist. Run the real-time node first.")
        return

    data = load_tracking_data(csv_path)
    t = data['t']
    n_samples = len(t)
    if n_samples == 0:
        print(f"[Warning] {csv_path} is empty.")
        return

    # Extract states
    x_real = data['x_real']
    y_real = data['y_real']
    psi_real = data['psi_real']

    # As the new node saves everything directly in NED/planner frame, we don't negate anything here.
    u_real = clean_outliers(data['u_real'])
    v_real = clean_outliers(data['v_real'])
    r_real = clean_outliers(data['r_real'])

    x_ref = data['x_ref']
    y_ref = data['y_ref']
    psi_ref = data['psi_ref']
    u_ref = data['u_ref']
    v_ref = data['v_ref']
    r_ref = data['r_ref']

    # Extract forces
    tau_u_applied = data['tau_u_applied']
    tau_r_applied = data['tau_r_applied']

    # Reference forces: check if present in CSV, otherwise fallback to planned CSV
    if 'tau_u_ref' in data and len(data['tau_u_ref']) == n_samples:
        tau_u_ref = data['tau_u_ref']
        tau_r_ref = data['tau_r_ref']
    else:
        plan_csv = script_dir / 'planned_trajectory_reference.csv'
        if plan_csv.exists():
            plan_data = load_tracking_data(plan_csv)
            tau_u_ref = plan_data['tau_u_ref'][:n_samples]
            tau_r_ref = plan_data['tau_r_ref'][:n_samples]
        else:
            tau_u_ref = np.zeros(n_samples)
            tau_r_ref = np.zeros(n_samples)

    # Compute Position Tracking Errors & Cumulative RMSE
    err_x = x_real - x_ref
    err_y = y_real - y_ref
    err_pos = np.hypot(err_x, err_y)
    cum_sq_err = np.cumsum(err_pos ** 2)
    sample_indices = np.arange(1, n_samples + 1)
    rmse_pos = np.sqrt(cum_sq_err / sample_indices)
    final_rmse = rmse_pos[-1]
    max_err = np.max(err_pos)

    # -----------------------------------------------------------------------
    # Create 6-Subplot Figure (3 Rows x 2 Columns)
    # -----------------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(14, 11), dpi=300)

    # 1. 2D Trajectory (X vs Y)
    ax_traj = axs[0, 0]
    ax_traj.plot(x_ref, y_ref, color=C_REF, lw=2.2, label='Ideal / Reference')
    ax_traj.plot(x_real, y_real, color=C_REAL, lw=1.8, ls='--', label='Realized / Odometry')
    ax_traj.scatter(x_ref[0], y_ref[0], color='#10B981', s=60, zorder=5, marker='o', label='Start')
    ax_traj.scatter(x_ref[-1], y_ref[-1], color='#EF4444', s=70, zorder=5, marker='X', label='Goal')
    ax_traj.set_title('2D Trajectory Tracking', fontweight='bold')
    ax_traj.set_xlabel('X Position [m]', fontweight='bold')
    ax_traj.set_ylabel('Y Position [m]', fontweight='bold')
    ax_traj.grid(True, ls=':', alpha=0.6)
    ax_traj.axis('equal')
    ax_traj.legend(loc='best', frameon=True, facecolor='white')

    # 2. Position Tracking RMSE over Time
    ax_rmse = axs[0, 1]
    ax_rmse.plot(t, rmse_pos, color=C_EMERALD, lw=2.0, label=f'Cumulative RMSE (Final: {final_rmse:.3f} m)')
    ax_rmse.plot(t, err_pos, color=C_GRAY, lw=1.0, ls=':', alpha=0.7, label=f'Instantaneous Error (Max: {max_err:.3f} m)')
    ax_rmse.set_title('Position Tracking RMSE vs Time', fontweight='bold')
    ax_rmse.set_xlabel('Time [s]', fontweight='bold')
    ax_rmse.set_ylabel('Position Error [m]', fontweight='bold')
    ax_rmse.grid(True, ls=':', alpha=0.6)
    ax_rmse.legend(loc='upper right', frameon=True, facecolor='white')

    # 3. Surge Velocity u
    ax_u = axs[1, 0]
    ax_u.plot(t, u_ref, color=C_REF, lw=2.0, label='Ideal $u_{ref}$')
    ax_u.plot(t, u_real, color=C_REAL, lw=1.6, ls='--', label='Realized $u_{real}$')
    ax_u.set_title('Surge Velocity ($u$)', fontweight='bold')
    ax_u.set_xlabel('Time [s]', fontweight='bold')
    ax_u.set_ylabel('$u$ [m/s]', fontweight='bold')
    ax_u.grid(True, ls=':', alpha=0.6)
    ax_u.legend(loc='upper right', frameon=True, facecolor='white')

    # 4. Sway Velocity v
    ax_v = axs[1, 1]
    ax_v.plot(t, v_ref, color=C_REF, lw=2.0, label='Ideal $v_{ref}$')
    ax_v.plot(t, v_real, color=C_REAL, lw=1.6, ls='--', label='Realized $v_{real}$')
    ax_v.set_title('Sway Velocity ($v$)', fontweight='bold')
    ax_v.set_xlabel('Time [s]', fontweight='bold')
    ax_v.set_ylabel('$v$ [m/s]', fontweight='bold')
    ax_v.grid(True, ls=':', alpha=0.6)
    ax_v.legend(loc='upper right', frameon=True, facecolor='white')

    # 5. Yaw Rate r
    ax_r = axs[2, 0]
    ax_r.plot(t, r_ref, color=C_REF, lw=2.0, label='Ideal $r_{ref}$')
    ax_r.plot(t, r_real, color=C_REAL, lw=1.6, ls='--', label='Realized $r_{real}$')
    ax_r.set_title('Yaw Rate ($r$)', fontweight='bold')
    ax_r.set_xlabel('Time [s]', fontweight='bold')
    ax_r.set_ylabel('$r$ [rad/s]', fontweight='bold')
    ax_r.grid(True, ls=':', alpha=0.6)
    ax_r.legend(loc='upper right', frameon=True, facecolor='white')

    # 6. Control Actions tau_u and tau_r (Ideal vs Realized)
    ax_tau = axs[2, 1]
    # Left axis: Surge force tau_u [N]
    line1 = ax_tau.plot(t, tau_u_ref, color=C_REF, lw=1.8, label=r'$\tau_u$ Ideal [N]')
    line2 = ax_tau.plot(t, tau_u_applied, color=C_REAL, lw=1.6, ls='--', label=r'$\tau_u$ Realized [N]')
    ax_tau.set_xlabel('Time [s]', fontweight='bold')
    ax_tau.set_ylabel('Surge Force $\\tau_u$ [N]', fontweight='bold', color='#1E3A8A')
    ax_tau.tick_params(axis='y', labelcolor='#1E3A8A')
    ax_tau.grid(True, ls=':', alpha=0.6)

    # Right axis: Yaw torque tau_r [N·m]
    ax_tau_r = ax_tau.twinx()
    line3 = ax_tau_r.plot(t, tau_r_ref, color=C_EMERALD, lw=1.8, label=r'$\tau_r$ Ideal [N·m]')
    line4 = ax_tau_r.plot(t, tau_r_applied, color=C_AMBER, lw=1.6, ls='--', label=r'$\tau_r$ Realized [N·m]')
    ax_tau_r.set_ylabel('Yaw Moment $\\tau_r$ [N·m]', fontweight='bold', color='#065F46')
    ax_tau_r.tick_params(axis='y', labelcolor='#065F46')

    # Merged legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax_tau.legend(lines, labels, loc='upper right', ncol=2, frameon=True, facecolor='white')
    ax_tau.set_title(r'Control Actions ($\tau_u, \tau_r$)', fontweight='bold')

    fig.suptitle(f'{case_title} - Real-Time Tracking & Dynamics Performance', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] Real-time tracking plot saved to: {output_png}")


if __name__ == '__main__':
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    plot_tracking_results(csv_arg, out_arg, case_title="Case 2 (6-Param Pseudo-Flatness)")
