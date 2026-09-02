#!/usr/bin/env python3
# Technique: Case 2 - Minimum Jerk QP Trajectory Planning + 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from c2_usv_params import DT_SIM, T_MAX, T_MIN
from c2_min_jerk_qp import MinJerkTrajectory2D
from c2_flatness_reconstruct import reconstruct_flatness_h2

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    waypoints = np.array([
        [0.0, 0.0],
        [8.0, 0.0],
        [14.0, 5.0],
        [14.0, 13.0],
        [20.0, 17.0],
        [28.0, 17.0],
        [32.0, 10.0],
        [26.0, 4.0],
        [20.0, 1.5]
    ])
    
    base_times = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])
    time_scale = 2.10
    times = base_times * time_scale
    
    v0_vec = (0.1, 0.0)
    dir_f = waypoints[8] - waypoints[7]
    dir_f_unit = dir_f / np.linalg.norm(dir_f)
    vf_vec = tuple(0.01 * dir_f_unit)
    
    t_start = time.perf_counter()
    planner = MinJerkTrajectory2D(waypoints, times, vel_start=v0_vec, vel_end=vf_vec)
    t_sim, pos, vel, acc, jerk = planner.sample(dt=DT_SIM)
    flat_data = reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim)
    solve_time_ms = (time.perf_counter() - t_start) * 1000.0

    
    eta_ref  = flat_data['eta']
    nu_ref   = flat_data['nu']
    tau_ref  = flat_data['tau_plan']
    T_plan   = flat_data['T_plan']
    
    planning_metrics = {
        "case": "Case 2",
        "solver_type": "QP (6-Param Pseudo-Flatness)",
        "solve_time_ms": float(solve_time_ms),
        "total_sim_time_s": float(t_sim[-1] - t_sim[0]),
        "num_samples": int(len(t_sim))
    }
    metrics_file = os.path.join(script_dir, 'planning_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(planning_metrics, f, indent=4)
    print(f"[Main] Trajectory Solver (QP) Compute Time: {solve_time_ms:.4f} ms")

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), dpi=300)
    c_blue  = '#2563EB'
    c_green = '#059669'
    c_amber = '#D97706'
    c_red   = '#DC2626'
    
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
    axs[2].plot(t_sim, tau_ref[:, 1], color=c_amber, lw=1.8, label='$\\tau_r$ [N·m]')
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
    axs[4].plot(t_sim, jerk[:, 0], color=c_blue, lw=1.5, label='$j_x$ [m/s³]')
    axs[4].plot(t_sim, jerk[:, 1], color=c_green, lw=1.5, label='$j_y$ [m/s³]')
    axs[4].plot(t_sim, jerk_mag, color=c_red, lw=1.8, ls='-', label='$|j|$ [m/s³]')
    axs[4].set_xlabel('Time [s]', fontweight='bold')
    axs[4].set_ylabel('Jerk [m/s³]', fontweight='bold')
    axs[4].set_title('Jerk Profiles', fontsize=12, fontweight='bold', pad=8)
    axs[4].grid(True, ls=':', alpha=0.6)
    axs[4].legend(loc='upper right', ncol=3, frameon=True, facecolor='white', fontsize=9)
    
    plt.tight_layout()
    out_img_path = os.path.join(script_dir, 'c2_flatness_planning_results.png')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Main] Planning completed. Diagnostic plot saved to: {out_img_path}")

if __name__ == '__main__':
    main()


