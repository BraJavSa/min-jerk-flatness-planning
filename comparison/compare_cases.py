#!/usr/bin/env python3
# Technique: USV Trajectory Performance Benchmarking & Comparative Diagnostics (Case 1 vs Case 2 vs Case 3)

import os
import sys
import json
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLATNESS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

CASE1_DIR = os.path.join(FLATNESS_DIR, 'case1')
CASE2_DIR = os.path.join(FLATNESS_DIR, 'case2')
CASE3_DIR = os.path.join(FLATNESS_DIR, 'case3')

def load_metrics(case_dir, case_name):
    metrics_path = os.path.join(case_dir, 'openloop_metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found for {case_name} at {metrics_path}. Please run simulation first.")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def import_from_dir(dir_path, module_name):
    file_path = os.path.join(dir_path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"{os.path.basename(dir_path)}_{module_name}", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_openloop_simulation(case_dir, case_key):
    prev_path = list(sys.path)
    sys.path.insert(0, case_dir)
    
    for mod_key in ['usv_params', 'c2_usv_params', 'min_jerk_qp', 'c2_min_jerk_qp', 'trajectory_nlp', 'flatness_reconstruct', 'c2_flatness_reconstruct', 'simulate_openloop', 'c2_simulate_openloop']:
        if mod_key in sys.modules:
            del sys.modules[mod_key]

    waypoints = np.array([
        [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
        [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
    ])
    base_times = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])
    times = base_times * 2.10
    v0_vec = (0.1, 0.0)
    dir_f = waypoints[8] - waypoints[7]
    dir_f_unit = dir_f / np.linalg.norm(dir_f)
    vf_vec = tuple(0.01 * dir_f_unit)

    if case_key == 'case1':
        import min_jerk_qp
        import flatness_reconstruct
        import simulate_openloop
        import usv_params
        
        planner = min_jerk_qp.MinJerkTrajectory2D(waypoints, times, vel_start=v0_vec, vel_end=vf_vec)
        t_sim, pos, vel, acc, jerk = planner.sample(dt=usv_params.DT_SIM)
        flat_data = flatness_reconstruct.reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim)
        rk4_func = simulate_openloop.real_6param_rk4_step
        dt_sim = usv_params.DT_SIM

    elif case_key == 'case2':
        import c2_min_jerk_qp
        import c2_flatness_reconstruct
        import c2_simulate_openloop
        import c2_usv_params

        planner = c2_min_jerk_qp.MinJerkTrajectory2D(waypoints, times, vel_start=v0_vec, vel_end=vf_vec)
        t_sim, pos, vel, acc, jerk = planner.sample(dt=c2_usv_params.DT_SIM)
        flat_data = c2_flatness_reconstruct.reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim)
        rk4_func = c2_simulate_openloop.real_6param_rk4_step
        dt_sim = c2_usv_params.DT_SIM

    elif case_key == 'case3':
        import trajectory_nlp
        import flatness_reconstruct
        import simulate_openloop
        import usv_params

        planner = trajectory_nlp.FlatnessNLP(waypoints, times, vel_start=v0_vec, vel_end=vf_vec, epsilon=0.1)
        t_sim, pos, vel, acc, jerk = planner.sample(dt_sim=usv_params.DT_SIM)
        flat_data = flatness_reconstruct.reconstruct_flatness_full(pos, vel, acc, jerk, t_sim)
        rk4_func = simulate_openloop.real_6param_rk4_step
        dt_sim = usv_params.DT_SIM

    eta_ref = flat_data['eta']
    tau_act_flat = flat_data['tau_act']
    state_real = np.column_stack([eta_ref, flat_data['nu']])[0].copy()
    hist_state_real = [state_real.copy()]

    for i in range(len(t_sim) - 1):
        Tu_apply, Tr_apply = tau_act_flat[i, 0], tau_act_flat[i, 1]
        state_real = rk4_func(state_real, Tu_apply, Tr_apply, dt_sim)
        hist_state_real.append(state_real.copy())

    sys.path = prev_path
    return waypoints, eta_ref, np.array(hist_state_real)

def main():
    print("[Comparison] Loading metrics from Case 1, Case 2, and Case 3...")
    m1 = load_metrics(CASE1_DIR, "Case 1")
    m2 = load_metrics(CASE2_DIR, "Case 2")
    m3 = load_metrics(CASE3_DIR, "Case 3")

    cases = ["Case 1", "Case 2", "Case 3"]
    colors = ['#2563EB', '#059669', '#8B5CF6']
    
    solve_times = [
        m1["trajectory_solver_time_ms"],
        m2["trajectory_solver_time_ms"],
        m3["trajectory_solver_time_ms"]
    ]
    
    rmse_pos = [
        m1["tracking_error"]["rmse_position_m"],
        m2["tracking_error"]["rmse_position_m"],
        m3["tracking_error"]["rmse_position_m"]
    ]
    
    max_pos = [
        m1["tracking_error"]["max_position_error_m"],
        m2["tracking_error"]["max_position_error_m"],
        m3["tracking_error"]["max_position_error_m"]
    ]
    
    rmse_psi_deg = [
        m1["tracking_error"]["rmse_heading_deg"],
        m2["tracking_error"]["rmse_heading_deg"],
        m3["tracking_error"]["rmse_heading_deg"]
    ]
    
    u_rmse = [m1["tracking_error"]["rmse_surge_u_mps"], m2["tracking_error"]["rmse_surge_u_mps"], m3["tracking_error"]["rmse_surge_u_mps"]]
    v_rmse = [m1["tracking_error"]["rmse_sway_v_mps"], m2["tracking_error"]["rmse_sway_v_mps"], m3["tracking_error"]["rmse_sway_v_mps"]]
    r_rmse = [m1["tracking_error"]["rmse_yaw_rate_r_radps"], m2["tracking_error"]["rmse_yaw_rate_r_radps"], m3["tracking_error"]["rmse_yaw_rate_r_radps"]]

    print("[Comparison] Extracting 2D spatial trajectories for comparative plot...")
    wps, eta_ref1, real1 = run_openloop_simulation(CASE1_DIR, 'case1')
    _, _, real2 = run_openloop_simulation(CASE2_DIR, 'case2')
    _, _, real3 = run_openloop_simulation(CASE3_DIR, 'case3')

    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.suptitle("USV Performance Benchmarking (Case 1 vs Case 2 vs Case 3)", fontsize=15, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    bars1 = ax1.bar(cases, solve_times, color=colors, width=0.55, edgecolor='#111827', linewidth=1.1)
    ax1.set_ylabel("Solve Time [ms]", fontweight='bold')
    ax1.set_title("Computation Time", fontsize=11, fontweight='bold', pad=8)
    ax1.grid(True, axis='y', ls=':', alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(solve_times)*0.02), f"{yval:.1f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, max(solve_times) * 1.15)

    ax2 = fig.add_subplot(2, 3, 2)
    bars2 = ax2.bar(cases, rmse_pos, color=colors, width=0.55, edgecolor='#111827', linewidth=1.1)
    ax2.set_ylabel("RMSE Position [m]", fontweight='bold')
    ax2.set_title("Position Tracking RMSE", fontsize=11, fontweight='bold', pad=8)
    ax2.grid(True, axis='y', ls=':', alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(rmse_pos)*0.02), f"{yval:.4f} m", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(rmse_pos) * 1.15)

    ax3 = fig.add_subplot(2, 3, 3)
    bars3 = ax3.bar(cases, max_pos, color=colors, width=0.55, edgecolor='#111827', linewidth=1.1)
    ax3.set_ylabel("Max Error [m]", fontweight='bold')
    ax3.set_title("Max Position Error", fontsize=11, fontweight='bold', pad=8)
    ax3.grid(True, axis='y', ls=':', alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + (max(max_pos)*0.02), f"{yval:.4f} m", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_ylim(0, max(max_pos) * 1.15)

    ax4 = fig.add_subplot(2, 3, 4)
    bars4 = ax4.bar(cases, rmse_psi_deg, color=colors, width=0.55, edgecolor='#111827', linewidth=1.1)
    ax4.set_ylabel("Heading RMSE [deg]", fontweight='bold')
    ax4.set_title("Heading Tracking RMSE", fontsize=11, fontweight='bold', pad=8)
    ax4.grid(True, axis='y', ls=':', alpha=0.6)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + (max(rmse_psi_deg)*0.02), f"{yval:.2f}°", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, max(rmse_psi_deg) * 1.15)

    ax5 = fig.add_subplot(2, 3, 5)
    x_indices = np.arange(len(cases))
    w = 0.25
    ax5.bar(x_indices - w, u_rmse, width=w, label='$u$ [m/s]', color='#2563EB', edgecolor='#111827')
    ax5.bar(x_indices,     v_rmse, width=w, label='$v$ [m/s]', color='#059669', edgecolor='#111827')
    ax5.bar(x_indices + w, r_rmse, width=w, label='$r$ [rad/s]', color='#D97706', edgecolor='#111827')
    ax5.set_xticks(x_indices)
    ax5.set_xticklabels(cases, fontsize=9, fontweight='bold')
    ax5.set_ylabel("RMSE Velocity", fontweight='bold')
    ax5.set_title("Velocity Errors", fontsize=11, fontweight='bold', pad=8)
    ax5.grid(True, axis='y', ls=':', alpha=0.6)
    ax5.legend(loc='upper right', frameon=True, facecolor='white', fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(eta_ref1[:, 0], eta_ref1[:, 1], color='#6B7280', lw=2.0, ls='--', label='Planned Reference')
    ax6.plot(real1[:, 0], real1[:, 1], color=colors[0], lw=1.8, label='Case 1')
    ax6.plot(real2[:, 0], real2[:, 1], color=colors[1], lw=1.8, label='Case 2')
    ax6.plot(real3[:, 0], real3[:, 1], color=colors[2], lw=1.8, label='Case 3')
    ax6.scatter(wps[:, 0], wps[:, 1], color='#111827', s=45, zorder=5, label='Waypoints')
    ax6.set_xlabel("X [m]", fontweight='bold')
    ax6.set_ylabel("Y [m]", fontweight='bold')
    ax6.set_title("2D Trajectories Overlay", fontsize=11, fontweight='bold', pad=8)
    ax6.grid(True, ls=':', alpha=0.6)
    ax6.axis('equal')
    ax6.legend(loc='best', frameon=True, facecolor='white', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_img = os.path.join(SCRIPT_DIR, 'cases_comparison_results.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()

    summary_data = {
        "case1": m1,
        "case2": m2,
        "case3": m3
    }
    with open(os.path.join(SCRIPT_DIR, 'comparison_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n==================================================")
    print(f"BENCHMARKING COMPARISON COMPLETED!")
    print(f"  - Comparison plot saved to: {out_img}")
    print(f"  - Summary JSON saved to: {os.path.join(SCRIPT_DIR, 'comparison_summary.json')}")
    print(f"==================================================\n")

if __name__ == '__main__':
    main()

