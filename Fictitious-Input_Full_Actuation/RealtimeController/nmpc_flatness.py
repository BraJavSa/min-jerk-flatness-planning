#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import casadi as ca
from scipy.interpolate import BSpline

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

from usv_params import (
    m11_real, m22_real, m33_real, Xu_real, Yv_real, Nr_real,
    dP, SURGE_GAIN, YAW_ARM, T_MAX, T_MIN, cmd_from_thrust_richards, thrust_from_cmd_richards
)

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

@dataclass
class NmpcWeights:
    q_pos: float = 50.0
    q_yaw: float = 20.0
    q_u: float = 1.0
    q_r: float = 3.0
    w_jerk: float = 0.0
    r_tau: float = 1e-3

class NmpcFlatness:
    NX = 6
    NU = 2
    N = 30

    def __init__(self, dt: float = 1.0 / 30.0,
                 n_ctrl: int = 12,
                 n_colloc: int = 20,
                 epsilon: float = 0.15,
                 weights: NmpcWeights = None):
        self.dt = dt
        self.tf = self.N * self.dt
        self.degree = 4
        self.n_ctrl = n_ctrl
        self.n_colloc = n_colloc
        self.epsilon = epsilon
        self.w = weights if weights is not None else NmpcWeights()

        n_internal = self.n_ctrl - self.degree - 1
        internal = np.linspace(0.0, self.tf, n_internal + 2)[1:-1]
        self.knots = np.concatenate((
            np.zeros(self.degree + 1),
            internal,
            np.full(self.degree + 1, self.tf)
        ))
        self.t_colloc = np.linspace(0.0, self.tf, self.n_colloc)

        b0 = self._basis_matrix(self.t_colloc, 0)
        b1 = self._basis_matrix(self.t_colloc, 1)
        b2 = self._basis_matrix(self.t_colloc, 2)
        b3 = self._basis_matrix(self.t_colloc, 3)
        b_bnd0 = self._basis_matrix(np.array([0.0]), 0)
        b_bnd1 = self._basis_matrix(np.array([0.0]), 1)

        self.B0_ca = ca.DM(b0)
        self.B1_ca = ca.DM(b1)
        self.B2_ca = ca.DM(b2)
        self.B3_ca = ca.DM(b3)
        self.B_bnd0_ca = ca.DM(b_bnd0)
        self.B_bnd1_ca = ca.DM(b_bnd1)

        self._build_opti()
        self.P_prev = None

        self._warmup()

    def _basis_matrix(self, ts: np.ndarray, der: int) -> np.ndarray:
        ts_eval = np.clip(ts, 0.0, self.tf - 1e-9)
        ts_eval[np.isclose(ts, 0.0)] = 0.0
        B = np.zeros((len(ts), self.n_ctrl))
        for i in range(self.n_ctrl):
            c = np.zeros(self.n_ctrl)
            c[i] = 1.0
            spl = BSpline(self.knots, c, self.degree, extrapolate=True)
            B[:, i] = spl(ts_eval, nu=der)
        return B

    def _build_opti(self):
        self.opti = ca.Opti()

        self.P = self.opti.variable(self.n_ctrl, 3)
        Px = self.P[:, 0]
        Py = self.P[:, 1]
        Ppsi = self.P[:, 2]

        self.P_x0 = self.opti.parameter(6)
        self.P_xref = self.opti.parameter(self.n_colloc)
        self.P_yref = self.opti.parameter(self.n_colloc)
        self.P_psiref = self.opti.parameter(self.n_colloc)
        self.P_uref = self.opti.parameter(self.n_colloc)
        self.P_rref = self.opti.parameter(self.n_colloc)
        self.P_turef = self.opti.parameter(self.n_colloc)
        self.P_trref = self.opti.parameter(self.n_colloc)

        x = ca.mtimes(self.B0_ca, Px)
        y = ca.mtimes(self.B0_ca, Py)
        psi = ca.mtimes(self.B0_ca, Ppsi)

        dx = ca.mtimes(self.B1_ca, Px)
        dy = ca.mtimes(self.B1_ca, Py)
        dpsi = ca.mtimes(self.B1_ca, Ppsi)

        ddx = ca.mtimes(self.B2_ca, Px)
        ddy = ca.mtimes(self.B2_ca, Py)
        ddpsi = ca.mtimes(self.B2_ca, Ppsi)

        u = dx * ca.cos(psi) + dy * ca.sin(psi)
        v = -dx * ca.sin(psi) + dy * ca.cos(psi)
        r = dpsi

        du = ddx * ca.cos(psi) + ddy * ca.sin(psi) + v * r
        dv = -ddx * ca.sin(psi) + ddy * ca.cos(psi) - u * r
        dr = ddpsi

        tau_u = m11_real * du - m22_real * v * r + Xu_real * u
        tau_v = m22_real * dv + m11_real * u * r + Yv_real * v
        tau_r = m33_real * dr - (m11_real - m22_real) * u * v + Nr_real * r

        self.T1 = 0.5 * (tau_u / SURGE_GAIN + tau_r / (2.0 * YAW_ARM))
        self.T2 = 0.5 * (tau_u / SURGE_GAIN - tau_r / (2.0 * YAW_ARM))
        self.tau_u = tau_u
        self.tau_v = tau_v
        self.tau_r = tau_r

        x0 = self.P_x0[0]; y0 = self.P_x0[1]; psi0 = self.P_x0[2]
        u0 = self.P_x0[3]; v0 = self.P_x0[4]; r0 = self.P_x0[5]

        self.opti.subject_to(ca.mtimes(self.B_bnd0_ca, Px) == x0)
        self.opti.subject_to(ca.mtimes(self.B_bnd0_ca, Py) == y0)
        self.opti.subject_to(ca.mtimes(self.B_bnd0_ca, Ppsi) == psi0)

        self.opti.subject_to(ca.mtimes(self.B_bnd1_ca, Px) == u0 * ca.cos(psi0) - v0 * ca.sin(psi0))
        self.opti.subject_to(ca.mtimes(self.B_bnd1_ca, Py) == u0 * ca.sin(psi0) + v0 * ca.cos(psi0))
        self.opti.subject_to(ca.mtimes(self.B_bnd1_ca, Ppsi) == r0)

        self.opti.subject_to(self.opti.bounded(-self.epsilon, self.tau_v, self.epsilon))

        self.opti.subject_to(self.opti.bounded(T_MIN, self.T1, T_MAX))
        self.opti.subject_to(self.opti.bounded(T_MIN, self.T2, T_MAX))

        err_pos = ca.sumsqr(x - self.P_xref) + ca.sumsqr(y - self.P_yref)
        err_yaw = ca.sumsqr(psi - self.P_psiref)
        err_u   = ca.sumsqr(u - self.P_uref)
        err_r   = ca.sumsqr(r - self.P_rref)
        err_tau = ca.sumsqr(self.tau_u - self.P_turef) + ca.sumsqr(self.tau_r - self.P_trref)

        cost = (self.w.q_pos * err_pos +
                self.w.q_yaw * err_yaw +
                self.w.q_u * err_u +
                self.w.q_r * err_r)
        if self.w.r_tau > 0.0:
            cost = cost + self.w.r_tau * err_tau
        if self.w.w_jerk > 0.0:
            jx = ca.mtimes(self.B3_ca, Px)
            jy = ca.mtimes(self.B3_ca, Py)
            jpsi = ca.mtimes(self.B3_ca, Ppsi)
            cost = cost + self.w.w_jerk * (ca.sumsqr(jx) + ca.sumsqr(jy) + 5.0 * ca.sumsqr(jpsi))
        self.opti.minimize(cost)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 30,
            "print_level": 0,
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "warm_start_init_point": "yes"
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    def _warmup(self):
        dummy_x0 = np.zeros(6)
        dummy_ref = np.zeros((self.N, 3))
        dummy_nu = np.zeros((self.N, 3))
        self.solve(dummy_x0, dummy_ref, dummy_nu)

    def reset(self):
        self.P_prev = None

    def solve(self, x0: np.ndarray,
              eta_ref: np.ndarray,
              nu_ref: np.ndarray,
              u_prev: np.ndarray = None,
              tau_ref: np.ndarray = None):
        t0 = time.perf_counter()

        n_avail = len(eta_ref)
        t_ref = np.linspace(0.0, self.tf, n_avail)

        xref_col = np.interp(self.t_colloc, t_ref, eta_ref[:, 0])
        yref_col = np.interp(self.t_colloc, t_ref, eta_ref[:, 1])
        uref_col = np.interp(self.t_colloc, t_ref, nu_ref[:, 0])
        rref_col = np.interp(self.t_colloc, t_ref, nu_ref[:, 2])

        if tau_ref is not None:
            turef_col = np.interp(self.t_colloc, t_ref, tau_ref[:, 0])
            trref_col = np.interp(self.t_colloc, t_ref, tau_ref[:, 1])
        else:
            turef_col = np.zeros(self.n_colloc)
            trref_col = np.zeros(self.n_colloc)

        psi_r0 = float(eta_ref[0, 2])
        dpsi_curr = wrap_to_pi(float(x0[2]) - psi_r0)
        psi0_unwrapped = psi_r0 + dpsi_curr

        psiref_unwrapped = np.unwrap(eta_ref[:, 2])
        psiref_col = np.interp(self.t_colloc, t_ref, psiref_unwrapped)

        x0_param = np.array([
            float(x0[0]),
            float(x0[1]),
            psi0_unwrapped,
            float(x0[3]),
            float(x0[4]),
            float(x0[5])
        ])

        self.opti.set_value(self.P_x0, x0_param)
        self.opti.set_value(self.P_xref, xref_col)
        self.opti.set_value(self.P_yref, yref_col)
        self.opti.set_value(self.P_psiref, psiref_col)
        self.opti.set_value(self.P_uref, uref_col)
        self.opti.set_value(self.P_rref, rref_col)
        self.opti.set_value(self.P_turef, turef_col)
        self.opti.set_value(self.P_trref, trref_col)

        if self.P_prev is not None:
            self.opti.set_initial(self.P, self.P_prev)
        else:
            self.opti.set_initial(self.P[:, 0], np.linspace(x0_param[0], xref_col[-1], self.n_ctrl))
            self.opti.set_initial(self.P[:, 1], np.linspace(x0_param[1], yref_col[-1], self.n_ctrl))
            self.opti.set_initial(self.P[:, 2], np.linspace(psi0_unwrapped, psiref_col[-1], self.n_ctrl))

        try:
            sol = self.opti.solve()
            P_val = sol.value(self.P)
            T1_raw = float(sol.value(self.T1)[0])
            T2_raw = float(sol.value(self.T2)[0])
            tau_u_val = float(sol.value(self.tau_u)[0])
            tau_v_val = float(sol.value(self.tau_v)[0])
            tau_r_val = float(sol.value(self.tau_r)[0])
        except Exception:

            P_val = self.opti.debug.value(self.P)
            T1_raw = float(self.opti.debug.value(self.T1)[0])
            T2_raw = float(self.opti.debug.value(self.T2)[0])
            tau_u_val = float(self.opti.debug.value(self.tau_u)[0])
            tau_v_val = float(self.opti.debug.value(self.tau_v)[0])
            tau_r_val = float(self.opti.debug.value(self.tau_r)[0])

        self.P_prev = P_val

        T1_sat = max(T_MIN, min(T_MAX, T1_raw))
        T2_sat = max(T_MIN, min(T_MAX, T2_raw))

        cmd_l = cmd_from_thrust_richards(T1_sat)
        cmd_r = cmd_from_thrust_richards(T2_sat)

        solve_ms = (time.perf_counter() - t0) * 1000.0

        return (T1_sat, T2_sat), (cmd_l, cmd_r), (tau_u_val, tau_v_val, tau_r_val), solve_ms
