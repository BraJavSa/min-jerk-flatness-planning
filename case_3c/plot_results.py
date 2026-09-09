#!/usr/bin/env python3
"""
Grafica los resultados generados por los binarios en C para Case 3:
  - case3_planning_results.csv   (main)
  - case3_openloop_results.csv   (simulate_openloop)

Uso:
  python3 plot_results.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

c_blue   = '#2563EB'
c_green  = '#059669'
c_amber  = '#D97706'
c_red    = '#DC2626'
c_purple = '#8B5CF6'

plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2


def load_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def plot_planning(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"[Aviso] No existe {csv_path}, se omite el plot de planificacion.")
        return
    d = load_csv(csv_path)
    t = d['t']

    waypoints = np.array([
        [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
        [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
    ])
    EPSILON_TV = 0.1
    T_MAX = 65.92
    T_MIN = -49.38

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)

    # 1. 2D Path
    axs[0].plot(d['x'], d['y'], color=c_blue, lw=2.2, label='Planned Path (Case 3 NLP)')
    axs[0].scatter(waypoints[:, 0], waypoints[:, 1], color='#111827', s=55, zorder=5, label='Waypoints')
    for idx_wp, (wx, wy) in enumerate(waypoints):
        axs[0].annotate(f'WP{idx_wp}', (wx, wy), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory (B-spline Flat Outputs NLP)', fontsize=12, fontweight='bold', pad=8)
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white')

    # 2. Body velocities
    axs[1].plot(t, d['u'], color=c_blue, lw=1.8, label='$u$ [m/s]')
    axs[1].plot(t, d['v'], color=c_green, lw=1.8, label='$v$ [m/s]')
    axs[1].plot(t, d['r'], color=c_amber, lw=1.8, label='$r$ [rad/s]')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Velocities', fontweight='bold')
    axs[1].set_title('Body Velocities', fontsize=12, fontweight='bold', pad=8)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    # 3. Fictitious sway force
    if 'tau_v' in d.dtype.names:
        axs[2].plot(t, d['tau_v'], color=c_purple, lw=1.8, label=r'$\tau_v$ [N]')
        axs[2].axhline(EPSILON_TV, color=c_red, ls='--', lw=1.2, label=rf'$+\epsilon$ ({EPSILON_TV:.1f} N)')
        axs[2].axhline(-EPSILON_TV, color=c_red, ls='--', lw=1.2, label=rf'$-\epsilon$ ({-EPSILON_TV:.1f} N)')
        axs[2].set_title('Fictitious Sway Force (NLP Inequality Constraint)', fontsize=12, fontweight='bold', pad=8)
    else:
        axs[2].plot(t, d['tau_u'], color=c_blue, lw=1.8, label=r'$\tau_u$ [N]')
        axs[2].plot(t, d['tau_r'], color=c_amber, lw=1.8, label=r'$\tau_r$ [N·m]')
        axs[2].set_title('Control Forces', fontsize=12, fontweight='bold', pad=8)
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Force [N]', fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.5)

    # 4. Thruster allocation
    axs[3].plot(t, d['T1'], color=c_green, lw=1.8, label='$T_1$ [N]')
    axs[3].plot(t, d['T2'], color=c_red, lw=1.8, label='$T_2$ [N]')
    axs[3].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=rf'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[3].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=rf'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Thrust [N]', fontweight='bold')
    axs[3].set_title('Thruster Allocation (Demanded)', fontsize=12, fontweight='bold', pad=8)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', ncol=4, frameon=True, facecolor='white', fontsize=8.5)

    # 5. Jerk profiles
    jerk_mag = np.hypot(d['jerk_x'], d['jerk_y'])
    axs[4].plot(t, d['jerk_x'], color=c_blue, lw=1.5, label='$j_x$ [m/s³]')
    axs[4].plot(t, d['jerk_y'], color=c_green, lw=1.5, label='$j_y$ [m/s³]')
    axs[4].plot(t, jerk_mag, color=c_red, lw=1.8, label='$|j|$ [m/s³]')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Jerk [m/s³]', fontweight='bold')
    axs[4].set_title('Jerk Profiles', fontsize=12, fontweight='bold', pad=8)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot de planificacion guardado en: {out_path}")


def plot_openloop(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"[Aviso] No existe {csv_path}, se omite el plot de open-loop.")
        return
    d = load_csv(csv_path)
    t = d['t']

    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    # 1. 2D Path
    axs[0].plot(d['x_plan'], d['y_plan'], color=c_blue, lw=2.2, ls='--', label='Planned (6-param Flatness NLP)')
    axs[0].plot(d['x_real'], d['y_real'], color=c_red, lw=2.0, label='Real (9-param Plant)')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory Tracking', fontsize=11, fontweight='bold')
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    # 2. Surge
    axs[1].plot(t, d['u_plan'], color=c_blue, ls='--', lw=1.8, label='Planned $u$')
    axs[1].plot(t, d['u_real'], color=c_red, lw=1.8, label='Real $u$')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Surge Velocity Tracking', fontsize=11, fontweight='bold')
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    # 3. Sway
    axs[2].plot(t, d['v_plan'], color=c_green, ls='--', lw=1.8, label='Planned $v$')
    axs[2].plot(t, d['v_real'], color=c_red, lw=1.8, label='Real $v$')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Sway Velocity Tracking', fontsize=11, fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    # 4. Yaw rate
    axs[3].plot(t, d['r_plan'], color=c_amber, ls='--', lw=1.8, label='Planned $r$')
    axs[3].plot(t, d['r_real'], color=c_red, lw=1.8, label='Real $r$')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Yaw rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Yaw Rate Tracking', fontsize=11, fontweight='bold')
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    # 5. Forces
    axs[4].plot(t, d['tau_u_plan'], color=c_blue, ls='--', lw=1.8, label=r'$\tau_u$ Demanded')
    axs[4].plot(t, d['tau_u_applied'], color=c_red, lw=1.8, label=r'$\tau_u$ Applied')
    axs[4].plot(t, d['tau_r_plan'], color=c_amber, ls='--', lw=1.8, label=r'$\tau_r$ Demanded')
    axs[4].plot(t, d['tau_r_applied'], color='#B45309', lw=1.8, label=r'$\tau_r$ Applied')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Forces', fontweight='bold')
    axs[4].set_title('Control Forces', fontsize=11, fontweight='bold')
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    # 6. Thrusters
    axs[5].plot(t, d['T1_plan'], color=c_green, ls='--', lw=1.8, label='$T_1$ Demanded')
    axs[5].plot(t, d['T1_applied'], color='#047857', lw=1.8, label='$T_1$ Applied')
    axs[5].plot(t, d['T2_plan'], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Demanded')
    axs[5].plot(t, d['T2_applied'], color=c_red, lw=1.8, label='$T_2$ Applied')
    axs[5].set_xlabel('Time [s]', fontweight='bold')
    axs[5].set_ylabel('Thrust [N]', fontweight='bold')
    axs[5].set_title('Thruster Allocation', fontsize=11, fontweight='bold')
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot de open-loop guardado en: {out_path}")


if __name__ == '__main__':
    plot_planning(
        os.path.join(script_dir, 'case3_planning_results.csv'),
        os.path.join(script_dir, 'case3_planning_results.png'),
    )
    plot_openloop(
        os.path.join(script_dir, 'case3_openloop_results.csv'),
        os.path.join(script_dir, 'case3_openloop_results.png'),
    )
