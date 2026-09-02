# Technique: Case 3 - Non-Linear Programming (NLP/IPOPT) Trajectory Planning + 6-Parameter Exact Flatness Reconstruction

import numpy as np
from usv_params import (
    m11_real, m22_real, m33_real, Xu_real, Yv_real, Nr_real, 
    dP, T_MAX, T_MIN, cmd_from_thrust_array, thrust_from_cmd_poly
)

def reconstruct_flatness_full(pos, vel, acc, jerk, t):
    x, y, psi = pos[:, 0], pos[:, 1], pos[:, 2]
    dx, dy, dpsi = vel[:, 0], vel[:, 1], vel[:, 2]
    ddx, ddy, ddpsi = acc[:, 0], acc[:, 1], acc[:, 2]
    
    u = dx * np.cos(psi) + dy * np.sin(psi)
    v = -dx * np.sin(psi) + dy * np.cos(psi)
    r = dpsi
    
    du = ddx * np.cos(psi) + ddy * np.sin(psi) + v * r
    dv = -ddx * np.sin(psi) + ddy * np.cos(psi) - u * r
    dr = ddpsi
    
    tau_u_raw = m11_real * du - m22_real * v * r + Xu_real * u
    tau_v_raw = m22_real * dv + m11_real * u * r + Yv_real * v
    tau_r_raw = m33_real * dr - (m11_real - m22_real) * u * v + Nr_real * r
    
    T1_raw = 0.5 * (tau_u_raw + tau_r_raw / dP)
    T2_raw = 0.5 * (tau_u_raw - tau_r_raw / dP)
    
    tau_u_max = 2.0 * T_MAX
    tau_u_min = 2.0 * T_MIN
    tau_r_max = (T_MAX - T_MIN) * dP
    tau_r_min = -tau_r_max
    
    tau_u = np.clip(tau_u_raw, tau_u_min, tau_u_max)
    tau_r = np.clip(tau_r_raw, tau_r_min, tau_r_max)
    
    T1_dem = np.clip(T1_raw, T_MIN, T_MAX)
    T2_dem = np.clip(T2_raw, T_MIN, T_MAX)
    
    cmd_1 = cmd_from_thrust_array(T1_dem)
    cmd_2 = cmd_from_thrust_array(T2_dem)
    
    T1_act = thrust_from_cmd_poly(cmd_1)
    T2_act = thrust_from_cmd_poly(cmd_2)
    
    tau_u_act = T1_act + T2_act
    tau_r_act = (T1_act - T2_act) * dP
    
    eta = np.column_stack([x, y, psi])
    nu = np.column_stack([u, v, r])
    tau_plan = np.column_stack([tau_u, tau_r])
    tau_act = np.column_stack([tau_u_act, tau_r_act])
    cmds = np.column_stack([cmd_1, cmd_2])
    
    return {
        'eta': eta,
        'nu': nu,
        'tau_plan': tau_plan,
        'tau_act': tau_act,
        'tau_v_raw': tau_v_raw,
        'cmds': cmds,
        'T_plan': np.column_stack([T1_dem, T2_dem]),
        'T_act': np.column_stack([T1_act, T2_act])
    }