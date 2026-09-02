#!/usr/bin/env python3
# Technique: USV Trajectory Performance Benchmarking & Comparative Diagnostics across 6 Cases
# Case 1 (QP 5-Param Py), Case 2 (QP 6-Param Py), Case 2 C (QP 6-Param C),
# Case 2 Semilla (QP 6-Param SQP Py), Case 2 Semilla C (QP 6-Param SQP C), Case 3 (CasADi NLP IPOPT Py)

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

CASE_CONFIGS = [
    {
        'key': 'case1',
        'dir': os.path.join(FLATNESS_DIR, 'case1'),
        'label': 'Case 1\n(Py)',
        'full_name': 'Case 1 (5-Param Py)',
        'color': '#2563EB'
    },
    {
        'key': 'case2',
        'dir': os.path.join(FLATNESS_DIR, 'case2'),
        'label': 'Case 2\n(Py)',
        'full_name': 'Case 2 (6-Param Py)',
        'color': '#059669'
    },
    {
        'key': 'case2_c',
        'dir': os.path.join(FLATNESS_DIR, 'case2_c'),
        'label': 'Case 2\n(C)',
        'full_name': 'Case 2 (6-Param C)',
        'color': '#10B981'
    },
    {
        'key': 'case2_semilla',
        'dir': os.path.join(FLATNESS_DIR, 'case2_semilla'),
        'label': 'Case 2 Sem\n(Py)',
        'full_name': 'Case 2 Semilla (SQP Py)',
        'color': '#D97706'
    },
    {
        'key': 'case2_semilla_c',
        'dir': os.path.join(FLATNESS_DIR, 'case2_semilla_c'),
        'label': 'Case 2 Sem\n(C)',
        'full_name': 'Case 2 Semilla (SQP C)',
        'color': '#F59E0B'
    },
    {
        'key': 'case3',
        'dir': os.path.join(FLATNESS_DIR, 'case3'),
        'label': 'Case 3\n(Py)',
        'full_name': 'Case 3 (NLP IPOPT Py)',
        'color': '#8B5CF6'
    }
]

def load_metrics(case_dir, case_name):
    metrics_path = os.path.join(case_dir, 'openloop_metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found for {case_name} at {metrics_path}. Please run simulation first.")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def run_openloop_simulation(case_dir, case_key):
    if case_key in ['case2_c', 'case2_semilla_c']:
        ol_csv = os.path.join(case_dir, 'c2_openloop_results.csv')
        ap_csv = os.path.join(case_dir, 'c2_openloop_applied.csv')
        wp_csv = os.path.join(case_dir, 'c2_waypoints.csv')
        
        d = np.genfromtxt(ol_csv, delimiter=',', names=True)
        wp = np.genfromtxt(wp_csv, delimiter=',', names=True) if os.path.exists(wp_csv) else None
        waypoints = np.column_stack([wp['x'], wp['y']]) if wp is not None else np.array([
            [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
            [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
        ])

        x_ref = d['x_ref'] if 'x_ref' in d.dtype.names else d['x']
        y_ref = d['y_ref'] if 'y_ref' in d.dtype.names else d['y']
        psi_ref = d['psi_ref'] if 'psi_ref' in d.dtype.names else d['psi']
        eta_ref = np.column_stack([x_ref, y_ref, psi_ref])

        if 'x_real' in d.dtype.names:
            real_states = np.column_stack([d['x_real'], d['y_real'], d['psi_real']])
        elif os.path.exists(ap_csv):
            ap = np.genfromtxt(ap_csv, delimiter=',', names=True)
            if 'real_x' in ap.dtype.names:
                real_states = np.column_stack([ap['real_x'], ap['real_y'], ap['real_psi']])
            else:
                real_states = eta_ref
        else:
            real_states = eta_ref

        return waypoints, eta_ref, real_states

    prev_path = list(sys.path)
    sys.path.insert(0, case_dir)
    
    for mod_key in ['usv_params', 'c2_usv_params', 'min_jerk_qp', 'c2_min_jerk_qp', 'trajectory_nlp', 'flatness_reconstruct', 'c2_flatness_reconstruct', 'c2_sqp_optimizer', 'simulate_openloop', 'c2_simulate_openloop']:
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

    elif case_key == 'case2_semilla':
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
    print("[Comparison] Loading metrics from all 6 cases...")
    metrics_map = {}
    for cfg in CASE_CONFIGS:
        metrics_map[cfg['key']] = load_metrics(cfg['dir'], cfg['full_name'])

    labels = [cfg['label'] for cfg in CASE_CONFIGS]
    colors = [cfg['color'] for cfg in CASE_CONFIGS]
    keys = [cfg['key'] for cfg in CASE_CONFIGS]
    
    solve_times = [metrics_map[k]["trajectory_solver_time_ms"] for k in keys]
    rmse_pos = [metrics_map[k]["tracking_error"]["rmse_position_m"] for k in keys]
    max_pos = [metrics_map[k]["tracking_error"]["max_position_error_m"] for k in keys]
    rmse_psi_deg = [metrics_map[k]["tracking_error"]["rmse_heading_deg"] for k in keys]
    
    u_rmse = [metrics_map[k]["tracking_error"]["rmse_surge_u_mps"] for k in keys]
    v_rmse = [metrics_map[k]["tracking_error"]["rmse_sway_v_mps"] for k in keys]
    r_rmse = [metrics_map[k]["tracking_error"]["rmse_yaw_rate_r_radps"] for k in keys]

    print("[Comparison] Extracting 2D spatial trajectories for comparative plot...")
    trajectories = {}
    waypoints_ref = None
    eta_plan_ref = None
    for cfg in CASE_CONFIGS:
        wps, eta_ref, real_st = run_openloop_simulation(cfg['dir'], cfg['key'])
        trajectories[cfg['key']] = real_st
        if waypoints_ref is None:
            waypoints_ref = wps
        if eta_plan_ref is None:
            eta_plan_ref = eta_ref

    fig = plt.figure(figsize=(18, 12), dpi=300)
    fig.suptitle("USV Performance Benchmarking (6-Case Comprehensive Comparison)", fontsize=16, fontweight='bold', y=0.98)

    # 1. Computation Time
    ax1 = fig.add_subplot(2, 3, 1)
    bars1 = ax1.bar(labels, solve_times, color=colors, width=0.6, edgecolor='#111827', linewidth=1.1)
    ax1.set_ylabel("Solve Time [ms]", fontweight='bold')
    ax1.set_title("Computation Time", fontsize=12, fontweight='bold', pad=8)
    ax1.grid(True, axis='y', ls=':', alpha=0.6)
    max_st = max(solve_times)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max_st * 0.02), f"{yval:.1f} ms", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax1.set_ylim(0, max_st * 1.18)

    # 2. Position RMSE
    ax2 = fig.add_subplot(2, 3, 2)
    bars2 = ax2.bar(labels, rmse_pos, color=colors, width=0.6, edgecolor='#111827', linewidth=1.1)
    ax2.set_ylabel("RMSE Position [m]", fontweight='bold')
    ax2.set_title("Position Tracking RMSE", fontsize=12, fontweight='bold', pad=8)
    ax2.grid(True, axis='y', ls=':', alpha=0.6)
    max_rp = max(rmse_pos)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max_rp * 0.02), f"{yval:.4f} m", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_ylim(0, max_rp * 1.18)

    # 3. Max Position Error
    ax3 = fig.add_subplot(2, 3, 3)
    bars3 = ax3.bar(labels, max_pos, color=colors, width=0.6, edgecolor='#111827', linewidth=1.1)
    ax3.set_ylabel("Max Error [m]", fontweight='bold')
    ax3.set_title("Max Position Error", fontsize=12, fontweight='bold', pad=8)
    ax3.grid(True, axis='y', ls=':', alpha=0.6)
    max_mp = max(max_pos)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + (max_mp * 0.02), f"{yval:.4f} m", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax3.set_ylim(0, max_mp * 1.18)

    # 4. Heading RMSE
    ax4 = fig.add_subplot(2, 3, 4)
    bars4 = ax4.bar(labels, rmse_psi_deg, color=colors, width=0.6, edgecolor='#111827', linewidth=1.1)
    ax4.set_ylabel("Heading RMSE [deg]", fontweight='bold')
    ax4.set_title("Heading Tracking RMSE", fontsize=12, fontweight='bold', pad=8)
    ax4.grid(True, axis='y', ls=':', alpha=0.6)
    max_psi = max(rmse_psi_deg)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + (max_psi * 0.02), f"{yval:.2f}°", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax4.set_ylim(0, max_psi * 1.18)

    # 5. Velocity Errors
    ax5 = fig.add_subplot(2, 3, 5)
    x_indices = np.arange(len(labels))
    w = 0.25
    ax5.bar(x_indices - w, u_rmse, width=w, label='$u$ [m/s]', color='#2563EB', edgecolor='#111827')
    ax5.bar(x_indices,     v_rmse, width=w, label='$v$ [m/s]', color='#059669', edgecolor='#111827')
    ax5.bar(x_indices + w, r_rmse, width=w, label='$r$ [rad/s]', color='#D97706', edgecolor='#111827')
    ax5.set_xticks(x_indices)
    ax5.set_xticklabels(labels, fontsize=8.5, fontweight='bold')
    ax5.set_ylabel("RMSE Velocity", fontweight='bold')
    ax5.set_title("Velocity Errors", fontsize=12, fontweight='bold', pad=8)
    ax5.grid(True, axis='y', ls=':', alpha=0.6)
    ax5.legend(loc='upper right', frameon=True, facecolor='white', fontsize=8)

    # 6. 2D Trajectories Overlay
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(eta_plan_ref[:, 0], eta_plan_ref[:, 1], color='#6B7280', lw=2.0, ls='--', label='Planned Reference')
    for cfg in CASE_CONFIGS:
        k = cfg['key']
        st = trajectories[k]
        ax6.plot(st[:, 0], st[:, 1], color=cfg['color'], lw=1.6, label=cfg['full_name'])
    ax6.scatter(waypoints_ref[:, 0], waypoints_ref[:, 1], color='#111827', s=45, zorder=5, label='Waypoints')
    ax6.set_xlabel("X [m]", fontweight='bold')
    ax6.set_ylabel("Y [m]", fontweight='bold')
    ax6.set_title("2D Trajectories Overlay", fontsize=12, fontweight='bold', pad=8)
    ax6.grid(True, ls=':', alpha=0.6)
    ax6.axis('equal')
    ax6.legend(loc='best', frameon=True, facecolor='white', fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_img = os.path.join(SCRIPT_DIR, 'cases_comparison_results.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()

    summary_data = {cfg['key']: metrics_map[cfg['key']] for cfg in CASE_CONFIGS}
    summary_path = os.path.join(SCRIPT_DIR, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n==================================================")
    print(f"6-CASE BENCHMARKING COMPARISON COMPLETED!")
    print(f"  - Comparison plot saved to: {out_img}")
    print(f"  - Summary JSON saved to: {summary_path}")
    print(f"==================================================\n")

if __name__ == '__main__':
    main()
