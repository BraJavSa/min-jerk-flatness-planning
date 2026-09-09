#!/usr/bin/env python3
# Technique: Case 3 REALTIME - Non-Linear Programming (NLP/IPOPT) Trajectory Planning
# + 6-Parameter Exact Flatness Reconstruction, seeded desde la pose/velocidad REAL
# del vehiculo al momento de arrancar la mision (recibida UNA sola vez desde caso3.py).
#
# La forma del camino (waypoints relativos) es identica a la del main.py offline;
# aqui simplemente se rota (por psi0) y se traslada (por x0,y0) de forma rigida para
# que el primer waypoint coincida con la pose actual del barco, en vez de arrancar
# siempre en (0,0) con rumbo 0.
#
# IMPORTANTE: x0,y0,psi0,u0,v0,r0 deben venir YA en el frame de planificacion
# (el mismo que usan usv_params/trajectory_nlp/flatness_reconstruct). La conversion
# desde la odometria ENU cruda de ROS la hace caso3.py ANTES de invocar este script
# (ver comentario de conversion de frame en caso3.py).
#
# Uso:
#   python3 main_realtime.py x0 y0 psi0 u0 v0 r0 output_csv_path [metrics_json_path]

import os
import sys
import csv
import time
import json
import numpy as np

# Resolucion de case_3/ (donde vive usv_params.py):
# 1) Si caso3.py nos invoco, ya nos paso la ruta correcta via PYTHONPATH -> "import usv_params"
#    funciona directo, sin tocar sys.path.
# 2) Si se corre este script suelto desde el arbol fuente (no instalado), case_3/
#    esta un nivel arriba de realtime/.
# 3) Fallback absoluto, por si acaso.
_CANDIDATES = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_3',
]
for _cand in _CANDIDATES:
    if _cand and os.path.isfile(os.path.join(_cand, 'usv_params.py')) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from usv_params import DT_SIM
from trajectory_nlp import FlatnessNLP
from flatness_reconstruct import reconstruct_flatness_full

# Misma forma de camino relativo que el main.py offline (arranca en (0,0), rumbo 0)
WAYPOINTS_LOCAL = np.array([
    [0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0],
    [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]
])
BASE_TIMES = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])
TIME_SCALE = 2.10
EPSILON_TV = 0.1


def build_world_waypoints(x0, y0, psi0):
    """Rota (psi0) y traslada (x0,y0) el camino local para que arranque en la pose actual."""
    c, s = np.cos(psi0), np.sin(psi0)
    R = np.array([[c, -s], [s, c]])
    return (R @ WAYPOINTS_LOCAL.T).T + np.array([x0, y0])


def main():
    if len(sys.argv) < 8:
        print("Uso: main_realtime.py x0 y0 psi0 u0 v0 r0 output_csv [metrics_json]")
        sys.exit(1)

    x0, y0, psi0, u0, v0, r0 = (float(a) for a in sys.argv[1:7])
    out_csv = sys.argv[7]
    metrics_json = sys.argv[8] if len(sys.argv) > 8 else os.path.join(
        os.path.dirname(os.path.abspath(out_csv)), 'planning_metrics.json')

    waypoints = build_world_waypoints(x0, y0, psi0)
    times = BASE_TIMES * TIME_SCALE

    # Velocidad inicial en frame mundo = R(psi0) @ (u0, v0)_body
    c0, s0 = np.cos(psi0), np.sin(psi0)
    vx0 = u0 * c0 - v0 * s0
    vy0 = u0 * s0 + v0 * c0
    v0_vec = (vx0, vy0)

    dir_f = waypoints[8] - waypoints[7]
    dir_f_unit = dir_f / np.linalg.norm(dir_f)
    vf_vec = tuple(0.01 * dir_f_unit)

    t_start = time.perf_counter()
    planner = FlatnessNLP(waypoints, times, vel_start=v0_vec, vel_end=vf_vec,
                           epsilon=EPSILON_TV, r_start=r0)
    t_sim, pos, vel, acc, jerk = planner.sample(dt_sim=DT_SIM)
    nlp_time_ms = (time.perf_counter() - t_start) * 1000.0

    t_fl0 = time.perf_counter()
    flat_data = reconstruct_flatness_full(pos, vel, acc, jerk, t_sim)
    flatness_time_ms = (time.perf_counter() - t_fl0) * 1000.0
    total_time_ms = nlp_time_ms + flatness_time_ms

    print("==================================================")
    print("CASE 3 REALTIME - NLP seeded desde pose/velocidad actual")
    print("--------------------------------------------------")
    print(f"Pose inicial (frame planif.): x0={x0:.3f} y0={y0:.3f} psi0={np.degrees(psi0):.2f} deg")
    print(f"Vel inicial body (frame planif.): u0={u0:.3f} v0={v0:.3f} r0={r0:.3f}")
    print(f"Muestras: {len(t_sim)} (dt={DT_SIM:.5f}s, T={t_sim[-1]-t_sim[0]:.3f}s)")
    print(f"Tiempo NLP:              {nlp_time_ms:.5f} ms")
    print(f"Tiempo reconstruccion:   {flatness_time_ms:.5f} ms")
    print(f"Tiempo total:            {total_time_ms:.5f} ms")
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
    print(f"Trayectoria completa guardada en: {out_csv}")

    metrics = {
        "case": "Case 3 Realtime",
        "solver_type": "CasADi NLP (IPOPT) seeded from live pose",
        "tiempo_qp_ms": float(nlp_time_ms),
        "tiempo_planitud_ms": float(flatness_time_ms),
        "tiempo_total_ms": float(total_time_ms),
        "solve_time_ms": float(total_time_ms),
        "total_sim_time_s": float(t_sim[-1] - t_sim[0]),
        "num_samples": int(len(t_sim)),
        "x0": x0, "y0": y0, "psi0": psi0, "u0": u0, "v0": v0, "r0": r0
    }
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metricas guardadas en: {metrics_json}")


if __name__ == '__main__':
    main()