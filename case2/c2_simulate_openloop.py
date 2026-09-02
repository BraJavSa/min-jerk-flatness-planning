#!/usr/bin/env python3
# Technique: Case 2 - Minimum Jerk QP Trajectory Planning + 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from c2_usv_params import (
    m11_real, m22_real, m33_real, Xu_real, Yv_real, Nr_real,
    dP, DT_SIM, T_MAX, T_MIN
)
from c2_min_jerk_qp import MinJerkTrajectory2D
from c2_flatness_reconstruct import reconstruct_flatness_h2

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def real_6param_rk4_step(state, Tu, Tr, dt):
    def deriv(s):
        _, _, p_i, u_i, v_i, r_i = s
        dx = u_i * np.cos(p_i) - v_i * np.sin(p_i)
        dy = u_i * np.sin(p_i) + v_i * np.cos(p_i)
        dpsi = r_i
        du = (Tu + m22_real * v_i * r_i - Xu_real * u_i) / m11_real
        dv = (-m11_real * u_i * r_i - Yv_real * v_i) / m22_real
        dr = (Tr + (m11_real - m22_real) * u_i * v_i - Nr_real * r_i) / m33_real
        return np.array([dx, dy, dpsi, du, dv, dr])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
    n_steps = len(t_sim)


    eta_ref = flat_data['eta']
    nu_ref = flat_data['nu']
    tau_ref = flat_data['tau_plan']
    tau_act_flat = flat_data['tau_act']
    T_plan = flat_data['T_plan']
    T_act_flat = flat_data['T_act']

    X_ref_all = np.column_stack([eta_ref, nu_ref])

    state_real = X_ref_all[0].copy()

    hist_state_real = [state_real.copy()]
    hist_tau_applied = []
    hist_T_applied = []

    for i in range(n_steps - 1):
        Tu_apply, Tr_apply = tau_act_flat[i, 0], tau_act_flat[i, 1]
        hist_tau_applied.append([Tu_apply, Tr_apply])
        hist_T_applied.append([T_act_flat[i, 0], T_act_flat[i, 1]])

        state_real = real_6param_rk4_step(state_real, Tu_apply, Tr_apply, DT_SIM)
        hist_state_real.append(state_real.copy())

    hist_state_real = np.array(hist_state_real)
    hist_tau_applied = np.array(hist_tau_applied)
    hist_T_applied = np.array(hist_T_applied)

    pos_err = np.hypot(hist_state_real[:, 0] - eta_ref[:, 0], hist_state_real[:, 1] - eta_ref[:, 1])
    rmse_pos = float(np.sqrt(np.mean(pos_err**2)))
    max_err_pos = float(np.max(pos_err))

    psi_err = np.arctan2(np.sin(hist_state_real[:, 2] - eta_ref[:, 2]), np.cos(hist_state_real[:, 2] - eta_ref[:, 2]))
    rmse_psi_rad = float(np.sqrt(np.mean(psi_err**2)))
    rmse_psi_deg = float(np.degrees(rmse_psi_rad))

    rmse_u = float(np.sqrt(np.mean((hist_state_real[:, 3] - nu_ref[:, 0])**2)))
    rmse_v = float(np.sqrt(np.mean((hist_state_real[:, 4] - nu_ref[:, 1])**2)))
    rmse_r = float(np.sqrt(np.mean((hist_state_real[:, 5] - nu_ref[:, 2])**2)))

    metrics = {
        "case": "Case 2",
        "solver_type": "QP (6-Param Pseudo-Flatness)",
        "trajectory_solver_time_ms": solve_time_ms,
        "tracking_error": {
            "rmse_position_m": rmse_pos,
            "max_position_error_m": max_err_pos,
            "rmse_heading_rad": rmse_psi_rad,
            "rmse_heading_deg": rmse_psi_deg,
            "rmse_surge_u_mps": rmse_u,
            "rmse_sway_v_mps": rmse_v,
            "rmse_yaw_rate_r_radps": rmse_r
        }
    }

    metrics_path = os.path.join(script_dir, 'openloop_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n==================================================")
    print("CASE 2 OPEN-LOOP TRAJECTORY METRICS:")
    print(f"  - Trajectory Solver Time (QP): {solve_time_ms:.4f} ms")
    print(f"  - Position RMSE:               {rmse_pos:.4f} m")
    print(f"  - Max Position Error:          {max_err_pos:.4f} m")
    print(f"  - Heading RMSE:                {rmse_psi_rad:.4f} rad ({rmse_psi_deg:.2f} deg)")
    print(f"  - Surge Velocity (u) RMSE:     {rmse_u:.4f} m/s")
    print(f"  - Sway Velocity (v) RMSE:      {rmse_v:.4f} m/s")
    print(f"  - Yaw Rate (r) RMSE:           {rmse_r:.4f} rad/s")
    print(f"Metrics saved to: {metrics_path}")
    print("==================================================\n")

    fig, axs = plt.subplots(6, 1, figsize=(10, 19), dpi=300)

    c_plan = '#2563EB'
    c_real = '#DC2626'
    c_green_plan = '#059669'
    c_green_real = '#047857'
    c_amber_plan = '#D97706'

    t_ctrl = t_sim[:-1]

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
    out_img_path = os.path.join(script_dir, 'c2_openloop_simulation_results.png')
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Open-loop simulation completed. Plot saved to: {out_img_path}")

if __name__ == '__main__':
    main()


