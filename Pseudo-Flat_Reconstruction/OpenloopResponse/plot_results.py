#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

c_blue = '#2563EB'
c_green = '#059669'
c_amber = '#D97706'
c_red = '#DC2626'
c_green_real = '#047857'

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader]
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(header)}

def plot_openloop(script_dir):
    csv_path = os.path.join(script_dir, 'case2_openloop_results.csv')
    applied_path = os.path.join(script_dir, 'case2_openloop_applied.csv')
    wp_path = os.path.join(script_dir, 'case2_waypoints.csv')
    if not os.path.exists(csv_path):
        print(f"[plot_results] {csv_path} not found, skipping open-loop plot.")
        return

    d = read_csv(csv_path)
    ap = read_csv(applied_path) if os.path.exists(applied_path) else None
    wp = read_csv(wp_path) if os.path.exists(wp_path) else None

    T_MAX, T_MIN = 65.92, -49.38
    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    axs[0].plot(d['x_ref'], d['y_ref'], color=c_blue, lw=2.2, ls='--', label='Planned (9-Param)')
    axs[0].plot(d['x_real'], d['y_real'], color=c_red, lw=2.0, label='Real (9-Param)')
    if wp is not None:
        axs[0].scatter(wp['x'], wp['y'], color='#111827', s=55, zorder=5, label='Waypoints')
        for idx_wp, (wx, wy) in enumerate(zip(wp['x'], wp['y'])):
            axs[0].annotate(f'WP{idx_wp}', (wx, wy), textcoords="offset points",
                             xytext=(5, 5), fontsize=8, fontweight='bold')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory', fontsize=11, fontweight='bold', pad=6)
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    axs[1].plot(d['t'], d['u_ref'], color=c_blue, ls='--', lw=1.8, label='Planned (9-Param)')
    axs[1].plot(d['t'], d['u_real'], color=c_red, lw=1.8, label='Real (9-Param)')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Surge Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[2].plot(d['t'], d['v_ref'], color=c_green, ls='--', lw=1.8, label='Planned (9-Param)')
    axs[2].plot(d['t'], d['v_real'], color=c_red, lw=1.8, label='Real (9-Param)')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Sway Velocity', fontsize=11, fontweight='bold', pad=6)
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[3].plot(d['t'], d['r_ref'], color=c_amber, ls='--', lw=1.8, label='Planned (9-Param)')
    axs[3].plot(d['t'], d['r_real'], color=c_red, lw=1.8, label='Real (9-Param)')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Yaw Rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Yaw Rate', fontsize=11, fontweight='bold', pad=6)
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    t_ctrl = ap['t'] if ap is not None else d['t']
    tau_u_app = ap['tau_u_applied'] if ap is not None else d['tau_u_ref']
    tau_r_app = ap['tau_r_applied'] if ap is not None else d['tau_r_ref']
    axs[4].plot(d['t'], d['tau_u_ref'], color=c_blue, ls='--', lw=1.8, label=r'$\tau_u$ Planned')
    axs[4].plot(t_ctrl, tau_u_app, color=c_red, lw=1.8, label=r'$\tau_u$ Applied')
    axs[4].plot(d['t'], d['tau_r_ref'], color=c_amber, ls='--', lw=1.8, label=r'$\tau_r$ Planned')
    axs[4].plot(t_ctrl, tau_r_app, color='#B45309', lw=1.8, label=r'$\tau_r$ Applied')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Forces', fontweight='bold')
    axs[4].set_title('Control Forces', fontsize=11, fontweight='bold', pad=6)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    T1_app = ap['T1_applied'] if ap is not None else d['T1_ref']
    T2_app = ap['T2_applied'] if ap is not None else d['T2_ref']
    axs[5].plot(d['t'], d['T1_ref'], color=c_green, ls='--', lw=1.8, label='$T_1$ Plan')
    axs[5].plot(t_ctrl, T1_app, color=c_green_real, lw=1.8, label='$T_1$ Applied')
    axs[5].plot(d['t'], d['T2_ref'], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Plan')
    axs[5].plot(t_ctrl, T2_app, color=c_red, lw=1.8, label='$T_2$ Applied')
    axs[5].axhline(T_MAX, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{max}}$ ({T_MAX:.1f} N)')
    axs[5].axhline(T_MIN, color='#9CA3AF', ls=':', lw=1.2, label=f'$T_{{min}}$ ({T_MIN:.1f} N)')
    axs[5].set_xlabel('Time [s]', fontweight='bold')
    axs[5].set_ylabel('Thrust [N]', fontweight='bold')
    axs[5].set_title('Thruster Allocation', fontsize=11, fontweight='bold', pad=6)
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    out_path = os.path.join(script_dir, 'case2_openloop_results.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = os.path.join(script_dir, 'case2_openloop_results.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot_results] Open-loop plot saved to:\n  PNG: {out_path}\n  PDF: {pdf_path}")

if __name__ == '__main__':
    plot_openloop(script_dir)
