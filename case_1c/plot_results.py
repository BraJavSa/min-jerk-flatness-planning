#!/usr/bin/env python3
"""
Grafica los resultados generados por los binarios en C:
  - case1_planning_results.csv   (main)
  - case1_openloop_results.csv   (simulate_openloop)

Uso:
  python3 plot_results.py
"""

import os
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
        print(f"[Aviso] No existe {csv_path}, se omite el plot de planificacion.")
        return
    d = load_csv(csv_path)
    t = d['t']

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)

    axs[0].plot(d['x'], d['y'], color=c_blue, lw=2.2, label='Trayectoria planeada')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('Trayectoria 2D', fontsize=12, fontweight='bold')
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white')

    axs[1].plot(t, d['u'], color=c_blue, lw=1.8, label='$u$ [m/s]')
    axs[1].plot(t, d['v'], color=c_green, lw=1.8, label='$v$ [m/s]')
    axs[1].plot(t, d['r'], color=c_amber, lw=1.8, label='$r$ [rad/s]')
    axs[1].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[1].set_ylabel('Velocidades', fontweight='bold')
    axs[1].set_title('Velocidades en el cuerpo', fontsize=12, fontweight='bold')
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)

    axs[2].plot(t, d['tau_u'], color=c_blue, lw=1.8, label=r'$\tau_u$ [N]')
    axs[2].plot(t, d['tau_r'], color=c_amber, lw=1.8, label=r'$\tau_r$ [N·m]')
    axs[2].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[2].set_ylabel('Fuerzas', fontweight='bold')
    axs[2].set_title('Fuerzas de control', fontsize=12, fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    axs[3].plot(t, d['T1'], color=c_green, lw=1.8, label='$T_1$ [N]')
    axs[3].plot(t, d['T2'], color=c_red, lw=1.8, label='$T_2$ [N]')
    axs[3].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[3].set_ylabel('Empuje [N]', fontweight='bold')
    axs[3].set_title('Asignacion de empuje', fontsize=12, fontweight='bold')
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=9)

    jerk_mag = np.hypot(d['jerk_x'], d['jerk_y'])
    axs[4].plot(t, d['jerk_x'], color=c_blue, lw=1.5, label='$j_x$ [m/s³]')
    axs[4].plot(t, d['jerk_y'], color=c_green, lw=1.5, label='$j_y$ [m/s³]')
    axs[4].plot(t, jerk_mag, color=c_red, lw=1.8, label='$|j|$ [m/s³]')
    axs[4].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[4].set_ylabel('Jerk [m/s³]', fontweight='bold')
    axs[4].set_title('Perfiles de jerk', fontsize=12, fontweight='bold')
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

    axs[0].plot(d['x_plan'], d['y_plan'], color=c_blue, lw=2.2, ls='--', label='Planeada (5-param)')
    axs[0].plot(d['x_real'], d['y_real'], color=c_red, lw=2.0, label='Real (9-param)')
    axs[0].set_xlabel('X [m]', fontweight='bold')
    axs[0].set_ylabel('Y [m]', fontweight='bold')
    axs[0].set_title('Trayectoria 2D', fontsize=11, fontweight='bold')
    axs[0].grid(True, ls=':', alpha=0.6)
    axs[0].axis('equal')
    axs[0].legend(loc='best', frameon=True, facecolor='white', fontsize=8.5)

    axs[1].plot(t, d['u_plan'], color=c_blue, ls='--', lw=1.8, label='Planeada (5-param)')
    axs[1].plot(t, d['u_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[1].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[1].set_ylabel('Surge $u$ [m/s]', fontweight='bold')
    axs[1].set_title('Velocidad de avance', fontsize=11, fontweight='bold')
    axs[1].grid(True, ls=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[2].plot(t, d['v_plan'], color=c_green, ls='--', lw=1.8, label='Planeada (5-param)')
    axs[2].plot(t, d['v_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[2].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[2].set_ylabel('Sway $v$ [m/s]', fontweight='bold')
    axs[2].set_title('Velocidad lateral', fontsize=11, fontweight='bold')
    axs[2].grid(True, ls=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[3].plot(t, d['r_plan'], color=c_amber, ls='--', lw=1.8, label='Planeada (5-param)')
    axs[3].plot(t, d['r_real'], color=c_red, lw=1.8, label='Real (9-param)')
    axs[3].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[3].set_ylabel('Yaw rate $r$ [rad/s]', fontweight='bold')
    axs[3].set_title('Velocidad de giro', fontsize=11, fontweight='bold')
    axs[3].grid(True, ls=':', alpha=0.6)
    axs[3].legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    axs[4].plot(t, d['tau_u_plan'], color=c_blue, ls='--', lw=1.8, label=r'$\tau_u$ Planeado')
    axs[4].plot(t, d['tau_u_applied'], color=c_red, lw=1.8, label=r'$\tau_u$ Aplicado')
    axs[4].plot(t, d['tau_r_plan'], color=c_amber, ls='--', lw=1.8, label=r'$\tau_r$ Planeado')
    axs[4].plot(t, d['tau_r_applied'], color='#B45309', lw=1.8, label=r'$\tau_r$ Aplicado')
    axs[4].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[4].set_ylabel('Fuerzas', fontweight='bold')
    axs[4].set_title('Fuerzas de control', fontsize=11, fontweight='bold')
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=2, frameon=True, facecolor='white', fontsize=8.5)

    axs[5].plot(t, d['T1_plan'], color=c_green, ls='--', lw=1.8, label='$T_1$ Plan')
    axs[5].plot(t, d['T1_applied'], color='#047857', lw=1.8, label='$T_1$ Aplicado')
    axs[5].plot(t, d['T2_plan'], color='#F43F5E', ls='--', lw=1.8, label='$T_2$ Plan')
    axs[5].plot(t, d['T2_applied'], color=c_red, lw=1.8, label='$T_2$ Aplicado')
    axs[5].set_xlabel('Tiempo [s]', fontweight='bold')
    axs[5].set_ylabel('Empuje [N]', fontweight='bold')
    axs[5].set_title('Asignacion de empuje', fontsize=11, fontweight='bold')
    axs[5].grid(True, ls=':', alpha=0.6)
    axs[5].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=8.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot de open-loop guardado en: {out_path}")


if __name__ == '__main__':
    plot_planning(
        os.path.join(script_dir, 'case1_planning_results.csv'),
        os.path.join(script_dir, 'case1_planning_results.png'),
    )
    plot_openloop(
        os.path.join(script_dir, 'case1_openloop_results.csv'),
        os.path.join(script_dir, 'case1_openloop_results.png'),
    )
