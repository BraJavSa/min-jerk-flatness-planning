#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

c_blue = '#2563EB'
c_green = '#059669'
c_amber = '#D97706'
c_red = '#DC2626'

plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def load_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)

def plot_planning(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} does not exist, skipping planning plot.")
        return
    d = load_csv(csv_path)
    t = d['t']

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)

    wp_path = os.path.join(script_dir, 'case1_waypoints.csv')
    if os.path.exists(wp_path):
        wp = np.genfromtxt(wp_path, delimiter=',')
        wp_x, wp_y = wp[:, 0], wp[:, 1]
    else:
        base_wp = np.array([
            [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
            [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
        ])
        wp_x, wp_y = base_wp[:, 0], base_wp[:, 1]

    axs[0].plot(d['x'], d['y'], color=c_blue, lw=2.2, label='Planned trajectory')
    axs[0].scatter(wp_x, wp_y, color='#111827', s=45, marker='o', edgecolors='white', linewidths=0.9, zorder=5, label='Waypoints')
    axs[0].scatter(d['x'][0], d['y'][0], color='#10B981', s=55, marker='o', edgecolors='black', linewidths=0.8, zorder=6, label='Start')
    axs[0].scatter(d['x'][-1], d['y'][-1], color='#DC2626', s=65, marker='X', edgecolors='black', linewidths=0.8, zorder=6, label='Goal')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory', fontsize=12, fontweight='bold')
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white')

    axs[1].plot(t, d['u'], color=c_blue, lw=1.8, label='$u$ [m/s]')
    axs[1].plot(t, d['v'], color=c_green, lw=1.8, label='$v$ [m/s]')
    axs[1].plot(t, d['r'], color=c_amber, lw=1.8, label='$r$ [rad/s]')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Velocities', fontweight='bold')
    axs[1].set_title('Body Velocities', fontsize=12, fontweight='bold')
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    axs[2].plot(t, d['tau_u'], color=c_blue, lw=1.8, label=r'$\tau_u$ [N]')
    axs[2].plot(t, d['tau_r'], color=c_amber, lw=1.8, label=r'$\tau_r$ [N·m]')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Forces', fontweight='bold')
    axs[2].set_title('Control Forces', fontsize=12, fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    axs[3].plot(t, d['T1'], color=c_green, lw=1.8, label='$T_1$ [N]')
    axs[3].plot(t, d['T2'], color=c_red, lw=1.8, label='$T_2$ [N]')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Thrust [N]', fontweight='bold')
    axs[3].set_title('Thruster Allocation', fontsize=12, fontweight='bold')
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    jerk_mag = np.hypot(d['jerk_x'], d['jerk_y'])
    axs[4].plot(t, d['jerk_x'], color=c_blue, lw=1.5, label='$j_x$ [m/s³]')
    axs[4].plot(t, d['jerk_y'], color=c_green, lw=1.5, label='$j_y$ [m/s³]')
    axs[4].plot(t, jerk_mag, color=c_red, lw=1.8, label='$|j|$ [m/s³]')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Jerk [m/s³]', fontweight='bold')
    axs[4].set_title('Jerk Profiles', fontsize=12, fontweight='bold')
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Planning plot saved to:\n  PNG: {out_path}\n  PDF: {pdf_path}")

if __name__ == '__main__':
    csv_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'case1_planning_results.csv')
    img_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, 'case1_planning_results.png')
    plot_planning(csv_in, img_out)
