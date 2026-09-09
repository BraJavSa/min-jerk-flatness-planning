# Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning + 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
# Reconstructing psi, u, v, r, tau_u, tau_r using the complete 9-parameter nonlinear hydrodynamic model.

import numpy as np
from c2_usv_params import (
    m11_9, m22_9, m33_9, Xu_9, Xuu_9, Yv_9, Yvv_9, Nr_9, Nrr_9, dP_9,
    T_MAX, T_MIN, cmd_from_thrust_array, thrust_from_cmd_poly
)
from c2_sqp_optimizer import optimize_psi_sqp


def _psi_dot_ode(psi, x_d, y_d, x_dd, y_dd, lam_tikhonov=0.015, r_hard_limit=5.0):
    """
    Pseudo-Flat first-order ODE for psi(t) retaining the complete nonlinear damping model:
      alpha(psi) = ((m22 - m11) / m22) * u
      beta(psi)  = -x_dd * sin(psi) + y_dd * cos(psi) - ((Yv + Yvv*|v|) / m22) * v
      r = (alpha * beta) / (alpha^2 + lambda^2)
    """
    u = x_d * np.cos(psi) + y_d * np.sin(psi)
    v = -x_d * np.sin(psi) + y_d * np.cos(psi)

    alpha = ((m22_9 - m11_9) / m22_9) * u
    beta = -x_dd * np.sin(psi) + y_dd * np.cos(psi) + ((Yv_9 + Yvv_9 * abs(v)) / m22_9) * v

    r = (alpha * beta) / (alpha**2 + lam_tikhonov**2)
    return float(np.clip(r, -r_hard_limit, r_hard_limit))


def reconstruct_flatness_h2(pos, vel, acc, jerk, t, psi0=None):
    x, y = pos[:, 0], pos[:, 1]
    x_d, y_d = vel[:, 0], vel[:, 1]
    x_dd, y_dd = acc[:, 0], acc[:, 1]

    # Reemplazo por la resolución algebraica bidiagonal (SQP / Newton)
    psi, r = optimize_psi_sqp(t, x_d, y_d, x_dd, y_dd, psi0=psi0)

    u = x_d * np.cos(psi) + y_d * np.sin(psi)
    v = -x_d * np.sin(psi) + y_d * np.cos(psi)

    u_dot = np.gradient(u, t)
    r_dot = np.gradient(r, t)

    # 9-parameter complete nonlinear dynamics recovery
    tau_u_raw = m11_9 * u_dot - m22_9 * v * r + Xu_9 * u + Xuu_9 * abs(u) * u
    tau_r_raw = m33_9 * r_dot - (m11_9 - m22_9) * u * v + Nr_9 * r + Nrr_9 * abs(r) * r

    T1_raw = 0.5 * (tau_u_raw + tau_r_raw / dP_9)
    T2_raw = 0.5 * (tau_u_raw - tau_r_raw / dP_9)

    tau_u_max = 2.0 * T_MAX
    tau_u_min = 2.0 * T_MIN
    tau_r_max = (T_MAX - T_MIN) * dP_9
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
    tau_r_act = (T1_act - T2_act) * dP_9

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
        'tau_u_raw': tau_u_raw,
        'tau_r_raw': tau_r_raw,
        'cmds': cmds,
        'T1_raw': T1_raw,
        'T2_raw': T2_raw,
        'T_plan': np.column_stack([T1_dem, T2_dem]),
        'T_act': np.column_stack([T1_act, T2_act])
    }