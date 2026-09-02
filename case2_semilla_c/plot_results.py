#!/usr/bin/env python3
"""Reads the CSVs produced by c2_main / c2_simulate_openloop (C port) and
regenerates the same diagnostic plots as the original Python pipeline."""

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

T_MAX, T_MIN = 65.92, -49.38

C_BLUE, C_GREEN, C_AMBER, C_RED = '#2563EB', '#059669', '#D97706', '#DC2626'


def load_csv(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    cols = {k: np.array([float(row[k]) for row in rows]) for k in rows[0].keys()}
    return cols


def plot_planning():
    wp = load_csv('c2_waypoints.csv')
    res = load_csv('c2_main_results.csv')

    waypoints = np.column_stack([wp['x'], wp['y']])
    t_sim = res['t']

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)

    axs[0].plot(res['x'], res['y'], color=C_BLUE, lw=2.2, label='Planned Path')
    axs[0].scatter(waypoints[:, 0], waypoints[:, 1], color='#111827', s=55, zorder=5, label='Waypoints')
    for idx, (wx, wy) in enumerate(waypoints):
        axs[0].annotate(f'WP{idx}', (wx, wy), textcoords="offset points", xytext=(5, 5),
                         fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold'); axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory (C port)', fontsize=12, fontweight='bold', pad=8)
    axs[0].grid(True, ls=':', alpha=0.6); axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white')

    axs[1].plot(t_sim, res['u'], color=C_BLUE, lw=1.8, label='$u$ [m/s]')
    axs[1].plot(t_sim, res['v'], color=C_GREEN, lw=1.8, label='$v$ [m/s]')
    axs[1].plot(t_sim, res['r'], color=C_AMBER, lw=1.8, label='$r$ [rad/s]')
    axs[1].set_xlabel('Time [s]', fontweight='bold'); axs[1].set_ylabel('Velocities', fontweight='bold')
    axs[1].set_title('Body Velocities', fontsize=12, fontweight='bold', pad=8)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    axs[2].plot(t_sim, res['tau_u'], color=C_BLUE, lw=1.8, label=r'$\tau_u$ [N]')
    axs[2].plot(t_sim, res['tau_r'], color=C_AMBER, lw=1.8, label=r'$\tau_r$ [N.m]')
    axs[2].set_xlabel('Time [s]', fontweight='bold'); axs[2].set_ylabel('Forces', fontweight='bold')
    axs[2].set_title('Control Forces', fontsize=12, fontweight='bold', pad=8)
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    axs[3].plot(t_sim, res['T1'], color=C_GREEN, lw=1.8, label='$T_1$ [N]')
    axs[3].plot(t_sim, res['T2'], color=C_RED, lw=1.8, label='$T_2$ [N]')
    axs[3].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[3].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[3].set_xlabel('Time [s]', fontweight='bold'); axs[3].set_ylabel('Thrust [N]', fontweight='bold')
    axs[3].set_title('Thruster Allocation', fontsize=12, fontweight='bold', pad=8)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', ncol=4, frameon=True, facecolor='white', fontsize=8.5)

    jerk_mag = np.hypot(res['jerk_x'], res['jerk_y'])
    axs[4].plot(t_sim, res['jerk_x'], color=C_BLUE, lw=1.5, label='$j_x$ [m/s3]')
    axs[4].plot(t_sim, res['jerk_y'], color=C_GREEN, lw=1.5, label='$j_y$ [m/s3]')
    axs[4].plot(t_sim, jerk_mag, color=C_RED, lw=1.8, label='$|j|$ [m/s3]')
    axs[4].set_xlabel('Time [s]', fontweight='bold'); axs[4].set_ylabel('Jerk [m/s3]', fontweight='bold')
    axs[4].set_title('Jerk Profiles', fontsize=12, fontweight='bold', pad=8)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig('c2_flatness_planning_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved c2_flatness_planning_results.png')


def plot_openloop():
    wp = load_csv('c2_waypoints.csv')
    ref = load_csv('c2_openloop_results.csv')
    app = load_csv('c2_openloop_applied.csv')

    waypoints = np.column_stack([wp['x'], wp['y']])
    t_sim = ref['t']
    t_ctrl = app['t'][:-1]

    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    axs[0].plot(ref['x'], ref['y'], color=C_BLUE, lw=2.2, ls='--', label='Planned')
    axs[0].plot(app['real_x'], app['real_y'], color=C_RED, lw=2.0, label='Real (RK4)')
    axs[0].scatter(waypoints[:, 0], waypoints[:, 1], color='#111827', s=50, zorder=5, label='Waypoints')
    for idx, (wx, wy) in enumerate(waypoints):
        axs[0].annotate(f'WP{idx}', (wx, wy), textcoords="offset points", xytext=(5, 5),
                         fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold'); axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory (C port)', fontsize=11, fontweight='bold', pad=6)
    axs[0].grid(True, ls=':', alpha=0.6); axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    axs[1].plot(t_sim, ref['u'], color=C_BLUE, ls='--', lw=1.8, label='Planned')
    axs[1].plot(t_sim, app['real_u'], color=C_RED, lw=1.8, label='Real')
    axs[1].set_xlabel('Time [s]', fontweight='bold'); axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Surge Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[2].plot(t_sim, ref['v'], color=C_GREEN, ls='--', lw=1.8, label='Planned')
    axs[2].plot(t_sim, app['real_v'], color=C_RED, lw=1.8, label='Real')
    axs[2].set_xlabel('Time [s]', fontweight='bold'); axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Sway Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[3].plot(t_sim, ref['r'], color=C_AMBER, ls='--', lw=1.8, label='Planned')
    axs[3].plot(t_sim, app['real_r'], color=C_RED, lw=1.8, label='Real')
    axs[3].set_xlabel('Time [s]', fontweight='bold'); axs[3].set_ylabel('Yaw Rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Yaw Rate', fontsize=11, fontweight='bold', pad=6)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[4].plot(t_sim, ref['tau_u'], color=C_BLUE, ls='--', lw=1.8, label=r'$\tau_u$ Planned')
    axs[4].plot(t_ctrl, app['tau_u_applied'][:-1], color=C_RED, lw=1.8, label=r'$\tau_u$ Applied')
    axs[4].plot(t_sim, ref['tau_r'], color=C_AMBER, ls='--', lw=1.8, label=r'$\tau_r$ Planned')
    axs[4].plot(t_ctrl, app['tau_r_applied'][:-1], color='#B45309', lw=1.8, label=r'$\tau_r$ Applied')
    axs[4].set_xlabel('Time [s]', fontweight='bold'); axs[4].set_ylabel('Forces', fontweight='bold')
    axs[4].set_title('Control Forces', fontsize=11, fontweight='bold', pad=6)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    axs[5].plot(t_sim, ref['T1'], color=C_GREEN, ls='--', lw=1.8, label='$T_1$ Plan')
    axs[5].plot(t_ctrl, app['T1_applied'][:-1], color='#047857', lw=1.8, label='$T_1$ Applied')
    axs[5].plot(t_sim, ref['T2'], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Plan')
    axs[5].plot(t_ctrl, app['T2_applied'][:-1], color=C_RED, lw=1.8, label='$T_2$ Applied')
    axs[5].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[5].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[5].set_xlabel('Time [s]', fontweight='bold'); axs[5].set_ylabel('Thrust [N]', fontweight='bold')
    axs[5].set_title('Thruster Allocation', fontsize=11, fontweight='bold', pad=6)
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    plt.savefig('c2_openloop_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved c2_openloop_simulation_results.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--planning', action='store_true', help='only regenerate the planning plot')
    ap.add_argument('--openloop', action='store_true', help='only regenerate the open-loop plot')
    args = ap.parse_args()

    if not args.planning and not args.openloop:
        plot_planning()
        plot_openloop()
    else:
        if args.planning:
            plot_planning()
        if args.openloop:
            plot_openloop()
