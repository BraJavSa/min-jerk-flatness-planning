#!/usr/bin/env python3
"""
Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)

Reads the CSVs produced by the C binaries (c2_main, c2_simulate_openloop)
and reproduces the original matplotlib diagnostic figures
(c2_flatness_planning_results.png, c2_openloop_simulation_results.png).

Usage:
    python3 plot_results.py            # looks for CSVs in current dir
    python3 plot_results.py --dir OUT  # looks for CSVs in OUT/
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

T_MAX = 65.92
T_MIN = -49.38

c_blue = '#2563EB'
c_green = '#059669'
c_amber = '#D97706'
c_red = '#DC2626'


def load_csv(path, cols):
    """Load a CSV with a header row into a dict of named columns."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return {c: data[c] for c in cols}


def plot_main(data_dir, out_dir):
    main_csv = os.path.join(data_dir, 'c2_main_results.csv')
    wp_csv = os.path.join(data_dir, 'c2_waypoints.csv')
    if not os.path.exists(main_csv):
        print(f'[plot_results] skip: {main_csv} not found')
        return

    d = load_csv(main_csv, ['t', 'x', 'y', 'psi', 'u', 'v', 'r',
                             'tau_u', 'tau_r', 'T1_plan', 'T2_plan', 'jx', 'jy'])
    wp = np.genfromtxt(wp_csv, delimiter=',', names=True)

    t_sim = d['t']
    eta_ref = np.column_stack([d['x'], d['y'], d['psi']])
    nu_ref = np.column_stack([d['u'], d['v'], d['r']])
    tau_ref = np.column_stack([d['tau_u'], d['tau_r']])
    T_plan = np.column_stack([d['T1_plan'], d['T2_plan']])
    jerk = np.column_stack([d['jx'], d['jy']])
    waypoints = np.column_stack([wp['x'], wp['y']])

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)

    axs[0].plot(eta_ref[:, 0], eta_ref[:, 1], color=c_blue, lw=2.2, label='Planned Path')
    axs[0].scatter(waypoints[:, 0], waypoints[:, 1], color='#111827', s=55, zorder=5, label='Waypoints')
    for idx_wp, (wx, wy) in enumerate(waypoints):
        axs[0].annotate(f'WP{idx_wp}', (wx, wy), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory', fontsize=12, fontweight='bold', pad=8)
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white')

    axs[1].plot(t_sim, nu_ref[:, 0], color=c_blue, lw=1.8, label='$u$ [m/s]')
    axs[1].plot(t_sim, nu_ref[:, 1], color=c_green, lw=1.8, label='$v$ [m/s]')
    axs[1].plot(t_sim, nu_ref[:, 2], color=c_amber, lw=1.8, label='$r$ [rad/s]')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Velocities', fontweight='bold')
    axs[1].set_title('Body Velocities', fontsize=12, fontweight='bold', pad=8)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    axs[2].plot(t_sim, tau_ref[:, 0], color=c_blue, lw=1.8, label='$\\tau_u$ [N]')
    axs[2].plot(t_sim, tau_ref[:, 1], color=c_amber, lw=1.8, label='$\\tau_r$ [N\u00b7m]')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Forces', fontweight='bold')
    axs[2].set_title('Control Forces', fontsize=12, fontweight='bold', pad=8)
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    axs[3].plot(t_sim, T_plan[:, 0], color=c_green, lw=1.8, label='$T_1$ [N]')
    axs[3].plot(t_sim, T_plan[:, 1], color=c_red, lw=1.8, label='$T_2$ [N]')
    axs[3].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[3].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Thrust [N]', fontweight='bold')
    axs[3].set_title('Thruster Allocation', fontsize=12, fontweight='bold', pad=8)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', ncol=4, frameon=True, facecolor='white', fontsize=8.5)

    jerk_mag = np.hypot(jerk[:, 0], jerk[:, 1])
    axs[4].plot(t_sim, jerk[:, 0], color=c_blue, lw=1.5, label='$j_x$ [m/s\u00b3]')
    axs[4].plot(t_sim, jerk[:, 1], color=c_green, lw=1.5, label='$j_y$ [m/s\u00b3]')
    axs[4].plot(t_sim, jerk_mag, color=c_red, lw=1.8, ls='-', label='$|j|$ [m/s\u00b3]')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Jerk [m/s\u00b3]', fontweight='bold')
    axs[4].set_title('Jerk Profiles', fontsize=12, fontweight='bold', pad=8)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'c2_flatness_planning_results.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[plot_results] wrote {out_path}')


def plot_openloop(data_dir, out_dir):
    ol_csv = os.path.join(data_dir, 'c2_openloop_results.csv')
    ap_csv = os.path.join(data_dir, 'c2_openloop_applied.csv')
    wp_csv = os.path.join(data_dir, 'c2_waypoints.csv')
    if not os.path.exists(ol_csv):
        print(f'[plot_results] skip: {ol_csv} not found')
        return

    d = load_csv(ol_csv, ['t', 'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
                           'x_real', 'y_real', 'psi_real', 'u_real', 'v_real', 'r_real',
                           'tau_u_plan', 'tau_r_plan', 'T1_plan', 'T2_plan'])
    ap = load_csv(ap_csv, ['t', 'tau_u_applied', 'tau_r_applied', 'T1_applied', 'T2_applied'])
    wp = np.genfromtxt(wp_csv, delimiter=',', names=True)
    waypoints = np.column_stack([wp['x'], wp['y']])

    t_sim = d['t']
    t_ctrl = ap['t']
    eta_ref = np.column_stack([d['x_ref'], d['y_ref'], d['psi_ref']])
    nu_ref = np.column_stack([d['u_ref'], d['v_ref'], d['r_ref']])
    hist_state_real = np.column_stack([d['x_real'], d['y_real'], d['psi_real'],
                                        d['u_real'], d['v_real'], d['r_real']])
    tau_ref = np.column_stack([d['tau_u_plan'], d['tau_r_plan']])
    T_plan = np.column_stack([d['T1_plan'], d['T2_plan']])
    hist_tau_applied = np.column_stack([ap['tau_u_applied'], ap['tau_r_applied']])
    hist_T_applied = np.column_stack([ap['T1_applied'], ap['T2_applied']])

    c_plan = '#2563EB'
    c_real = '#DC2626'
    c_green_plan = '#059669'
    c_green_real = '#047857'
    c_amber_plan = '#D97706'

    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    axs[0].plot(eta_ref[:, 0], eta_ref[:, 1], color=c_plan, lw=2.2, ls='--', label='Planned')
    axs[0].plot(hist_state_real[:, 0], hist_state_real[:, 1], color=c_real, lw=2.0, label='Real')
    axs[0].scatter(waypoints[:, 0], waypoints[:, 1], color='#111827', s=50, zorder=5, label='Waypoints')
    for idx_wp, (wx, wy) in enumerate(waypoints):
        axs[0].annotate(f'WP{idx_wp}', (wx, wy), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory', fontsize=11, fontweight='bold', pad=6)
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    axs[1].plot(t_sim, nu_ref[:, 0], color=c_plan, ls='--', lw=1.8, label='Planned')
    axs[1].plot(t_sim, hist_state_real[:, 3], color=c_real, lw=1.8, label='Real')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Surge Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[2].plot(t_sim, nu_ref[:, 1], color=c_green_plan, ls='--', lw=1.8, label='Planned')
    axs[2].plot(t_sim, hist_state_real[:, 4], color=c_real, lw=1.8, label='Real')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Sway Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[3].plot(t_sim, nu_ref[:, 2], color=c_amber_plan, ls='--', lw=1.8, label='Planned')
    axs[3].plot(t_sim, hist_state_real[:, 5], color=c_real, lw=1.8, label='Real')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Yaw Rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Yaw Rate', fontsize=11, fontweight='bold', pad=6)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[4].plot(t_sim, tau_ref[:, 0], color=c_plan, ls='--', lw=1.8, label='$\\tau_u$ Planned')
    axs[4].plot(t_ctrl, hist_tau_applied[:, 0], color=c_real, lw=1.8, label='$\\tau_u$ Applied')
    axs[4].plot(t_sim, tau_ref[:, 1], color=c_amber_plan, ls='--', lw=1.8, label='$\\tau_r$ Planned')
    axs[4].plot(t_ctrl, hist_tau_applied[:, 1], color='#B45309', lw=1.8, label='$\\tau_r$ Applied')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Forces', fontweight='bold')
    axs[4].set_title('Control Forces', fontsize=11, fontweight='bold', pad=6)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    axs[5].plot(t_sim, T_plan[:, 0], color=c_green_plan, ls='--', lw=1.8, label='$T_1$ Plan')
    axs[5].plot(t_ctrl, hist_T_applied[:, 0], color=c_green_real, lw=1.8, label='$T_1$ Applied')
    axs[5].plot(t_sim, T_plan[:, 1], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Plan')
    axs[5].plot(t_ctrl, hist_T_applied[:, 1], color=c_real, lw=1.8, label='$T_2$ Applied')
    axs[5].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[5].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[5].set_xlabel('Time [s]', fontweight='bold')
    axs[5].set_ylabel('Thrust [N]', fontweight='bold')
    axs[5].set_title('Thruster Allocation', fontsize=11, fontweight='bold', pad=6)
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'c2_openloop_simulation_results.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[plot_results] wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='directory with the C-generated CSVs')
    ap.add_argument('--out', default=None, help='output directory for PNGs (defaults to --dir)')
    args = ap.parse_args()
    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    plot_main(args.dir, out_dir)
    plot_openloop(args.dir, out_dir)


if __name__ == '__main__':
    main()
