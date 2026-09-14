#!/usr/bin/env python3
"""
min_jerk_qp.py

Python implementation of the Minimum-Jerk Quadratic Program (QP) Trajectory Planner
for the Fictitious-Input Full Actuation differential-flatness method (6 parameters).

Formulates the trajectory generation as an analytical equality-constrained QP
using 5th-order (quintic) polynomial splines between waypoints, solved via KKT system.
"""

import numpy as np

N_COEF = 6  # 5th-order quintic polynomial (c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5)


def poly_basis(tau: float, order: int) -> np.ndarray:
    """Evaluate basis vector for derivative of given order at local time tau."""
    basis = np.zeros(N_COEF)
    for i in range(order, N_COEF):
        coeff = 1.0
        for k in range(order):
            coeff *= (i - k)
        basis[i] = coeff * (tau ** (i - order))
    return basis


def segment_cost_matrix(T: float) -> np.ndarray:
    """Compute the 6x6 jerk cost matrix for a segment of duration T.

    Integral of (d^3 p / dt^3)^2 from 0 to T.
    """
    H = np.zeros((N_COEF, N_COEF))
    for i in range(3, N_COEF):
        for j in range(3, N_COEF):
            ci = float(i * (i - 1) * (i - 2))
            cj = float(j * (j - 1) * (j - 2))
            power = (i - 3) + (j - 3) + 1
            H[i, j] = ci * cj * (T ** power) / float(power)
    return H


def solve_min_jerk_1d(waypoints: np.ndarray, times: np.ndarray,
                       v0: float, vf: float,
                       a0: float = 0.0, af: float = 0.0) -> np.ndarray:
    """Solve 1D minimum-jerk equality-constrained QP via KKT linear system.

    Returns:
        coeffs: np.ndarray of shape (n_seg, 6)
    """
    n_wp = len(waypoints)
    n_seg = n_wp - 1
    n_vars = n_seg * N_COEF

    # Objective matrix H (block diagonal)
    H = np.zeros((n_vars, n_vars))
    for k in range(n_seg):
        T = times[k + 1] - times[k]
        H[k * N_COEF:(k + 1) * N_COEF, k * N_COEF:(k + 1) * N_COEF] = segment_cost_matrix(T)

    A_list = []
    b_list = []

    # 1. Waypoint interpolation constraints at segment boundaries
    for k in range(n_seg):
        T = times[k + 1] - times[k]
        # p_k(0) = wp_k
        row0 = np.zeros(n_vars)
        row0[k * N_COEF:(k + 1) * N_COEF] = poly_basis(0.0, 0)
        A_list.append(row0)
        b_list.append(float(waypoints[k]))

        # p_k(T_k) = wp_{k+1}
        row1 = np.zeros(n_vars)
        row1[k * N_COEF:(k + 1) * N_COEF] = poly_basis(T, 0)
        A_list.append(row1)
        b_list.append(float(waypoints[k + 1]))

    # 2. Initial velocity and final velocity
    row_v0 = np.zeros(n_vars)
    row_v0[0:N_COEF] = poly_basis(0.0, 1)
    A_list.append(row_v0)
    b_list.append(float(v0))

    T_last = times[-1] - times[-2]
    row_vf = np.zeros(n_vars)
    row_vf[(n_seg - 1) * N_COEF:n_seg * N_COEF] = poly_basis(T_last, 1)
    A_list.append(row_vf)
    b_list.append(float(vf))

    # 3. Initial acceleration and final acceleration
    row_a0 = np.zeros(n_vars)
    row_a0[0:N_COEF] = poly_basis(0.0, 2)
    A_list.append(row_a0)
    b_list.append(float(a0))

    row_af = np.zeros(n_vars)
    row_af[(n_seg - 1) * N_COEF:n_seg * N_COEF] = poly_basis(T_last, 2)
    A_list.append(row_af)
    b_list.append(float(af))

    # 4. C1 and C2 continuity at internal knots
    for k in range(n_seg - 1):
        T = times[k + 1] - times[k]
        for order in (1, 2):
            row_c = np.zeros(n_vars)
            row_c[k * N_COEF:(k + 1) * N_COEF] = poly_basis(T, order)
            row_c[(k + 1) * N_COEF:(k + 2) * N_COEF] = -poly_basis(0.0, order)
            A_list.append(row_c)
            b_list.append(0.0)

    A = np.array(A_list)
    b = np.array(b_list)
    m = len(b)

    # Solve KKT linear system [H  A^T; A  0] [c; lambda] = [0; b]
    KKT = np.zeros((n_vars + m, n_vars + m))
    KKT[:n_vars, :n_vars] = H + 1e-8 * np.eye(n_vars)
    KKT[:n_vars, n_vars:] = A.T
    KKT[n_vars:, :n_vars] = A

    rhs = np.zeros(n_vars + m)
    rhs[n_vars:] = b

    sol = np.linalg.solve(KKT, rhs)
    return sol[:n_vars].reshape(n_seg, N_COEF)


class MinJerkQP:
    """Minimum-Jerk Quadratic Program Trajectory Planner in Python.

    Drop-in replacement for CasADi FlatnessNLP, solving the 2D quintic spline
    via linear KKT system in < 1 ms, then reconstructing flat outputs
    (x, y, psi) and derivatives for the 6-parameter Fictitious-Input method.
    """

    def __init__(self, waypoints, times,
                 vel_start=(0.1, 0.0), vel_end=(0.01, 0.0),
                 psi_start=None, r_start=0.0,
                 include_sway_dynamics: bool = True, **kwargs):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.tf = float(self.times[-1])
        self.v0 = vel_start
        self.vf = vel_end
        self.psi_start = psi_start
        self.r_start = r_start

        self.include_sway_dynamics = include_sway_dynamics

        # Solve 1D QP for X and Y
        self.cx = solve_min_jerk_1d(self.waypoints[:, 0], self.times, self.v0[0], self.vf[0])
        self.cy = solve_min_jerk_1d(self.waypoints[:, 1], self.times, self.v0[1], self.vf[1])

    def sample(self, dt_sim: float):
        """Sample the continuous minimum-jerk trajectory at fixed dt_sim steps.

        Returns:
            t_sim: 1D array of timestamps
            pos:   (N, 3) array of [x, y, psi]
            vel:   (N, 3) array of [dx, dy, dpsi]
            acc:   (N, 3) array of [ddx, ddy, ddpsi]
            jerk:  (N, 3) array of [jx, jy, jpsi]
        """
        t_sim = np.arange(0.0, self.tf + 1e-8, dt_sim)
        n = len(t_sim)
        n_seg = len(self.times) - 1

        seg_idx = np.searchsorted(self.times, t_sim, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, n_seg - 1)
        tau = t_sim - self.times[seg_idx]

        coeffs = np.stack([self.cx, self.cy])  # shape (2, n_seg, 6)
        c = coeffs[:, seg_idx, :]              # shape (2, n, 6)

        # Vectorized polynomial evaluation
        tau_p0 = np.array([tau**p for p in range(6)])
        tau_p1 = np.array([np.zeros_like(tau), np.ones_like(tau), 2*tau, 3*tau**2, 4*tau**3, 5*tau**4])
        tau_p2 = np.array([np.zeros_like(tau), np.zeros_like(tau), 2*np.ones_like(tau), 6*tau, 12*tau**2, 20*tau**3])
        tau_p3 = np.array([np.zeros_like(tau), np.zeros_like(tau), np.zeros_like(tau), 6*np.ones_like(tau), 24*tau, 60*tau**2])

        pos_xy = np.sum(c * tau_p0.T[None, :, :], axis=2).T
        vel_xy = np.sum(c * tau_p1.T[None, :, :], axis=2).T
        acc_xy = np.sum(c * tau_p2.T[None, :, :], axis=2).T
        jerk_xy = np.sum(c * tau_p3.T[None, :, :], axis=2).T

        dx, dy = vel_xy[:, 0], vel_xy[:, 1]
        speed = np.hypot(dx, dy)

        # Course angle (path velocity direction)
        chi_raw = np.zeros(n)
        for i in range(n):
            if speed[i] > 0.05:
                chi_raw[i] = np.arctan2(dy[i], dx[i])
            elif i > 0:
                chi_raw[i] = chi_raw[i - 1]
            elif self.psi_start is not None:
                chi_raw[i] = self.psi_start
            else:
                chi_raw[i] = np.arctan2(self.waypoints[1, 1] - self.waypoints[0, 1],
                                        self.waypoints[1, 0] - self.waypoints[0, 0])

        chi = np.unwrap(chi_raw)

        if self.include_sway_dynamics:
            # Underactuated sway dynamics: m22 * dv/dt + Yv * v = -m11 * U * r
            # Allows natural sideslip so that fictitious sway force tau_v ≈ 0
            from usv_params import m11_real, m22_real, Yv_real
            from scipy.signal import savgol_filter

            r_chi = (dx * acc_xy[:, 1] - dy * acc_xy[:, 0]) / np.maximum(speed**2, 1e-4)

            v_sway = np.zeros(n)
            for i in range(n - 1):
                rhs = -(Yv_real / m22_real) * v_sway[i] - (m11_real / m22_real) * speed[i] * r_chi[i]
                v_sway[i + 1] = v_sway[i] + rhs * dt_sim

            beta = np.arcsin(np.clip(v_sway / np.maximum(speed, 0.05), -0.35, 0.35))
            win_len = min(51, (n // 2) * 2 - 1)
            beta_smooth = savgol_filter(beta, window_length=win_len, polyorder=3)
            psi = chi - beta_smooth

            dpsi = savgol_filter(np.gradient(psi, dt_sim), window_length=win_len, polyorder=3)
            ddpsi = savgol_filter(np.gradient(dpsi, dt_sim), window_length=win_len, polyorder=3)
            jpsi = np.gradient(ddpsi, dt_sim)
        else:
            psi = chi.copy()
            dpsi = np.gradient(psi, t_sim)
            ddpsi = np.gradient(dpsi, t_sim)
            jpsi = np.gradient(ddpsi, t_sim)

        # Ensure exact match with initial psi0 if provided
        if self.psi_start is not None:
            diff0 = (self.psi_start - psi[0] + np.pi) % (2.0 * np.pi) - np.pi
            if abs(diff0) > 1e-4:
                T_blend = min(2.5, self.tf * 0.1)
                tau_b = np.clip(t_sim / T_blend, 0.0, 1.0)
                blend = 1.0 - (10.0 * tau_b**3 - 15.0 * tau_b**4 + 6.0 * tau_b**5)
                psi += diff0 * blend
                dpsi += np.gradient(diff0 * blend, dt_sim)

        if self.r_start is not None and abs(self.r_start) > 1e-4:
            dpsi[0] = self.r_start

        ddpsi[0] = 0.0
        ddpsi[-1] = 0.0

        pos = np.column_stack([pos_xy, psi])
        vel = np.column_stack([vel_xy, dpsi])
        acc = np.column_stack([acc_xy, ddpsi])
        jerk = np.column_stack([jerk_xy, jpsi])

        return t_sim, pos, vel, acc, jerk
