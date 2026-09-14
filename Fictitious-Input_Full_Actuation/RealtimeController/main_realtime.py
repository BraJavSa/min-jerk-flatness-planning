#!/usr/bin/env python3
import os
import sys
import csv
import time
import json
from pathlib import Path
import numpy as np

def _setup_import_paths():
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    case_dir = this_dir.parent
    if str(case_dir) not in sys.path:
        sys.path.insert(1, str(case_dir))
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory('min-jerk-flatness-planning'))
        for sub in ('Fictitious-Input_Full_Actuation/RealtimeController', 'Fictitious-Input_Full_Actuation', 'case_3/realtime_mpc', 'case_3'):
            p = share_dir / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.append(str(p))
    except Exception:
        pass

_setup_import_paths()

from usv_params import DT_SIM
from min_jerk_qp import MinJerkQP
from flatness_reconstruct import reconstruct_flatness_full

WAYPOINTS_LOCAL = np.array([
    [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
    [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
])
BASE_TIMES = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])
TIME_SCALE = 2.10

def build_world_waypoints(x0, y0, psi0):
    c, s = np.cos(psi0), np.sin(psi0)
    R = np.array([[c, -s], [s, c]])
    return (R @ WAYPOINTS_LOCAL.T).T + np.array([x0, y0])

def main():
    if len(sys.argv) < 8:
        print("Usage: main_realtime.py x0 y0 psi0 u0 v0 r0 output_csv [metrics_json]")
        sys.exit(1)

    x0, y0, psi0, u0, v0, r0 = (float(a) for a in sys.argv[1:7])
    out_csv = Path(sys.argv[7]).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_json = Path(sys.argv[8]).resolve() if len(sys.argv) > 8 else out_csv.parent / 'planning_metrics.json'

    waypoints = build_world_waypoints(x0, y0, psi0)
    times = BASE_TIMES * TIME_SCALE

    c0, s0 = np.cos(psi0), np.sin(psi0)
    vx0 = u0 * c0 - v0 * s0
    vy0 = u0 * s0 + v0 * c0
    if np.hypot(vx0, vy0) < 0.05:
        vx0 = 0.1 * c0
        vy0 = 0.1 * s0
    v0_vec = (vx0, vy0)

    dir_f = waypoints[8] - waypoints[7]
    dir_f_norm = np.linalg.norm(dir_f)
    dir_f_unit = dir_f / dir_f_norm if dir_f_norm > 1e-6 else np.array([1.0, 0.0])
    vf_vec = tuple(0.01 * dir_f_unit)

    t_start = time.perf_counter()
    planner = MinJerkQP(waypoints, times, vel_start=v0_vec, vel_end=vf_vec,
                        psi_start=psi0, r_start=r0)
    t_sim, pos, vel, acc, jerk = planner.sample(dt_sim=DT_SIM)
    qp_time_ms = (time.perf_counter() - t_start) * 1000.0

    t_fl0 = time.perf_counter()
    flat_data = reconstruct_flatness_full(pos, vel, acc, jerk, t_sim)
    flatness_time_ms = (time.perf_counter() - t_fl0) * 1000.0
    total_time_ms = qp_time_ms + flatness_time_ms

    print("==================================================")
    print("CASE 3 REALTIME MPC - Python QP Trajectory Planning")
    print("--------------------------------------------------")
    print(f"Initial Pose: x0={x0:.3f} m, y0={y0:.3f} m, psi0={np.degrees(psi0):.2f} deg")
    print(f"Initial Vel:  u0={u0:.3f} m/s, v0={v0:.3f} m/s, r0={r0:.3f} rad/s")
    print(f"Samples:      {len(t_sim)} (dt={DT_SIM:.5f} s, T={t_sim[-1]-t_sim[0]:.3f} s)")
    print(f"QP Solve:     {qp_time_ms:.4f} ms")
    print(f"Flatness:     {flatness_time_ms:.4f} ms")
    print(f"Total Time:   {total_time_ms:.4f} ms")
    print("==================================================")

    eta_ref = flat_data['eta']
    nu_ref = flat_data['nu']
    tau_ref = flat_data['tau_plan']
    T_plan = flat_data['T_plan']
    cmds = flat_data['cmds']

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            't', 'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
            'tau_u_ref', 'tau_r_ref', 'T1_ref', 'T2_ref', 'cmd_left_ref', 'cmd_right_ref'
        ])
        for i in range(len(t_sim)):
            writer.writerow([
                f'{t_sim[i]:.6f}',
                f'{eta_ref[i, 0]:.6f}', f'{eta_ref[i, 1]:.6f}', f'{eta_ref[i, 2]:.6f}',
                f'{nu_ref[i, 0]:.6f}', f'{nu_ref[i, 1]:.6f}', f'{nu_ref[i, 2]:.6f}',
                f'{tau_ref[i, 0]:.6f}', f'{tau_ref[i, 1]:.6f}',
                f'{T_plan[i, 0]:.6f}', f'{T_plan[i, 1]:.6f}',
                f'{cmds[i, 0]:.6f}', f'{cmds[i, 1]:.6f}',
            ])
    print(f"[main_realtime] Trajectory reference saved to: {out_csv}")
    wp_csv = out_csv.parent / 'waypoints.csv'
    with open(wp_csv, 'w', newline='') as fwp:
        writer_wp = csv.writer(fwp)
        for wp in waypoints:
            writer_wp.writerow([f'{wp[0]:.6f}', f'{wp[1]:.6f}'])

    metrics = {
        'case': 'Case 3 Realtime (Python)',
        'solver_type': 'QP (6-Param Model in Python)',
        'QP Planning Time': f'{qp_time_ms:.5f} ms',
        'Flatness Reconstruction Time': f'{flatness_time_ms:.5f} ms',
        'Total Time': f'{total_time_ms:.5f} ms',
        'tiempo_qp_ms': float(qp_time_ms),
        'tiempo_planitud_ms': float(flatness_time_ms),
        'tiempo_total_ms': float(total_time_ms),
        'duration_s': float(t_sim[-1] - t_sim[0]),
        'num_samples': len(t_sim),
        'planning_time_ms': {
            'qp_solve_ms': qp_time_ms,
            'flatness_reconstruction_ms': flatness_time_ms,
            'total_planning_ms': total_time_ms
        }
    }
    with open(metrics_json, 'w') as jf:
        json.dump(metrics, jf, indent=4)
    print(f"[main_realtime] Planning metrics saved to: {metrics_json}")

if __name__ == '__main__':
    main()
