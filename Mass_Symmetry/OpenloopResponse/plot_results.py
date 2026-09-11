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

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def load_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)

def plot_openloop(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV does not exist: {csv_path}")
        return
    d = load_csv(csv_path)
    t = d['t']

    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    axs[0].plot(d['x_plan'], d['y_plan'], color=c_blue, lw=2.2, ls='--', label='Planned (5-param)')
    axs[0].plot(d['x_real'], d['y_real'], color=c_red, lw=2.0, label='Real (9-param)')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('2D Trajectory', fontsize=11, fontweight='bold')
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    axs[1].plot(t, d['u_plan'], color=c_blue, ls='--', lw=1.8, label='Planned (5-param)')
    axs[1].plot(t, d['u_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[1].set_xlabel('Time [s]', fontweight='bold')
    axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Surge Velocity', fontsize=11, fontweight='bold')
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[2].plot(t, d['v_plan'], color=c_green, ls='--', lw=1.8, label='Planned (5-param)')
    axs[2].plot(t, d['v_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Sway Velocity', fontsize=11, fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[3].plot(t, d['r_plan'], color=c_amber, ls='--', lw=1.8, label='Planned (5-param)')
    axs[3].plot(t, d['r_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[3].set_xlabel('Time [s]', fontweight='bold')
    axs[3].set_ylabel('Yaw Rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Yaw Rate', fontsize=11, fontweight='bold')
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[4].plot(t, d['tau_u_plan'], color=c_blue, ls='--', lw=1.8, label=r'$\tau_u$ Planned')
    axs[4].plot(t, d['tau_u_applied'], color=c_red, lw=1.8, label=r'$\tau_u$ Applied')
    axs[4].plot(t, d['tau_r_plan'], color=c_amber, ls='--', lw=1.8, label=r'$\tau_r$ Planned')
    axs[4].plot(t, d['tau_r_applied'], color='#B45309', lw=1.8, label=r'$\tau_r$ Applied')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Forces', fontweight='bold')
    axs[4].set_title('Control Forces', fontsize=11, fontweight='bold')
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    axs[5].plot(t, d['T1_plan'], color=c_green, ls='--', lw=1.8, label='$T_1$ Plan')
    axs[5].plot(t, d['T1_applied'], color='#047857', lw=1.8, label='$T_1$ Applied')
    axs[5].plot(t, d['T2_plan'], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Plan')
    axs[5].plot(t, d['T2_applied'], color=c_red, lw=1.8, label='$T_2$ Applied')
    axs[5].set_xlabel('Time [s]', fontweight='bold')
    axs[5].set_ylabel('Thrust [N]', fontweight='bold')
    axs[5].set_title('Thruster Allocation', fontsize=11, fontweight='bold')
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Open-loop plot saved to:\n  PNG: {out_path}\n  PDF: {pdf_path}")

if __name__ == '__main__':
    csv_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'case1_openloop_results.csv')
    img_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, 'case1_openloop_results.png')
    plot_openloop(csv_in, img_out)
