# Technique: Case 3 - Non-Linear Programming (NLP/IPOPT) Trajectory Planning
# via B-spline-parameterized FLAT OUTPUTS (Helling, Lutz, Meurer 2021, Sec. 3.3 + Appendix A).
#
# Key difference vs. a "generic" direct-collocation NLP:
#   The flat output z(t) = [x(t), y(t), psi(t)] is parameterized DIRECTLY with a
#   B-spline. Its control points P are the ONLY decision variables. Position,
#   velocity, acceleration and jerk are then just fixed LINEAR combinations of P
#   (basis matrices), so the vessel dynamics (7a) are satisfied *automatically*
#   -- there is no ODE/integrator equality constraint to impose (that's the whole
#   point of exploiting differential flatness).
#
#   The only thing that still needs to be enforced is the constraint that makes the
#   fully-actuated flat model behave like the real UNDERACTUATED vessel: the
#   fictitious sway force theta_tau_v (paper eq. 9) must be (approximately) zero,
#   eq. (29): -epsilon <= theta_tau_v <= epsilon. That is the ONLY nonlinear
#   inequality constraint in the NLP.

import numpy as np
import casadi as ca
from scipy.interpolate import BSpline
from usv_params import m11_real, m22_real, Yv_real


class FlatnessNLP:
    def __init__(self, waypoints, times, vel_start=(0.1, 0.0), vel_end=(0.01, 0.0),
                 epsilon=0.1, degree=4, n_ctrl_pts=None, n_colloc=400):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.tf = float(self.times[-1])
        self.v0 = vel_start
        self.vf = vel_end
        self.epsilon = epsilon
        self.degree = degree  # D_j in the paper ( = beta_j + 2 = 4 )

        n_wp = len(self.waypoints)
        self.n_ctrl = n_ctrl_pts if n_ctrl_pts is not None else max(4 * n_wp, self.degree + 6)

        # Clamped, open, uniform knot vector on [0, tf]  (paper eq. 19)
        self.knots = self._clamped_knot_vector(self.n_ctrl, self.degree, self.tf)

        # ---- Precompute basis matrices (constants, NOT decision variables) ----
        self.t_colloc = np.linspace(0.0, self.tf, n_colloc)
        self.B0 = self._basis_matrix(self.t_colloc, der=0)
        self.B1 = self._basis_matrix(self.t_colloc, der=1)
        self.B2 = self._basis_matrix(self.t_colloc, der=2)
        self.B3 = self._basis_matrix(self.t_colloc, der=3)

        self.B_wp = self._basis_matrix(self.times, der=0)          # waypoints, pos
        self.B_bnd0 = self._basis_matrix(np.array([0.0, self.tf]), der=0)
        self.B_bnd1 = self._basis_matrix(np.array([0.0, self.tf]), der=1)
        self.B_bnd2 = self._basis_matrix(np.array([0.0, self.tf]), der=2)

        self._build_nlp()

    # ------------------------------------------------------------------ #
    # B-spline machinery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamped_knot_vector(n_ctrl, degree, tf):
        n_internal = n_ctrl - degree - 1
        if n_internal < 0:
            raise ValueError("n_ctrl_pts must be >= degree + 1")
        internal = np.linspace(0.0, tf, n_internal + 2)[1:-1]
        knots = np.concatenate((np.zeros(degree + 1), internal, np.full(degree + 1, tf)))
        return knots

    def _basis_matrix(self, ts, der):
        """B[i,j] = d^der/dt^der  N_j(ts[i])  -- purely numeric, precomputed once."""
        ts_eval = np.clip(ts, 0.0, self.tf - 1e-9)  # avoid right-open-interval NaN at t=tf
        ts_eval[np.isclose(ts, 0.0)] = 0.0
        B = np.zeros((len(ts), self.n_ctrl))
        for i in range(self.n_ctrl):
            c = np.zeros(self.n_ctrl)
            c[i] = 1.0
            spl = BSpline(self.knots, c, self.degree, extrapolate=True)
            B[:, i] = spl(ts_eval, nu=der)
        return B

    def _greville_abscissae(self):
        g = np.zeros(self.n_ctrl)
        for i in range(self.n_ctrl):
            g[i] = np.mean(self.knots[i + 1: i + self.degree + 1])
        return g

    def _waypoint_interp(self, t_query):
        """Piecewise-linear guess for (x, y, psi) used only to seed the NLP."""
        idx = np.searchsorted(self.times, t_query)
        idx = max(1, min(idx, len(self.times) - 1))
        p0, p1 = self.waypoints[idx - 1], self.waypoints[idx]
        t0, t1 = self.times[idx - 1], self.times[idx]
        alpha = (t_query - t0) / max(t1 - t0, 1e-6)
        x_g = p0[0] + alpha * (p1[0] - p0[0])
        y_g = p0[1] + alpha * (p1[1] - p0[1])
        psi_g = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        return x_g, y_g, psi_g

    # ------------------------------------------------------------------ #
    # NLP setup
    # ------------------------------------------------------------------ #
    def _build_nlp(self):
        B0, B1, B2, B3 = (ca.DM(M) for M in (self.B0, self.B1, self.B2, self.B3))
        B_wp = ca.DM(self.B_wp)
        B_bnd0, B_bnd1, B_bnd2 = (ca.DM(M) for M in (self.B_bnd0, self.B_bnd1, self.B_bnd2))

        self.opti = ca.Opti()
        # ONLY decision variable: the flat-output B-spline control points.
        self.P = self.opti.variable(self.n_ctrl, 3)  # columns: x, y, psi

        Px, Py, Ppsi = self.P[:, 0], self.P[:, 1], self.P[:, 2]

        # --- Cost: minimize jerk energy of the flat outputs (yaw weighted more) ---
        jx = ca.mtimes(B3, Px)
        jy = ca.mtimes(B3, Py)
        jpsi = ca.mtimes(B3, Ppsi)
        dt_c = self.t_colloc[1] - self.t_colloc[0]
        cost = dt_c * (ca.sumsqr(jx) + ca.sumsqr(jy) + 10.0 * ca.sumsqr(jpsi))
        self.opti.minimize(cost)

        # --- Waypoint constraints: z(t_wp) = waypoint  (equality, positions only) ---
        x_wp = ca.mtimes(B_wp, Px)
        y_wp = ca.mtimes(B_wp, Py)
        self.opti.subject_to(x_wp == self.waypoints[:, 0])
        self.opti.subject_to(y_wp == self.waypoints[:, 1])

        # --- Boundary conditions on velocity / acceleration (rows 0 = t0, 1 = tf) ---
        vx_b = ca.mtimes(B_bnd1, Px)
        vy_b = ca.mtimes(B_bnd1, Py)
        vpsi_b = ca.mtimes(B_bnd1, Ppsi)
        ax_b = ca.mtimes(B_bnd2, Px)
        ay_b = ca.mtimes(B_bnd2, Py)
        apsi_b = ca.mtimes(B_bnd2, Ppsi)

        self.opti.subject_to(vx_b[0] == self.v0[0])
        self.opti.subject_to(vy_b[0] == self.v0[1])
        self.opti.subject_to(vpsi_b[0] == 0.0)          # r(0) = 0
        self.opti.subject_to(vx_b[1] == self.vf[0])
        self.opti.subject_to(vy_b[1] == self.vf[1])

        self.opti.subject_to(ax_b[0] == 0.0)
        self.opti.subject_to(ay_b[0] == 0.0)
        self.opti.subject_to(apsi_b[0] == 0.0)
        self.opti.subject_to(ax_b[1] == 0.0)
        self.opti.subject_to(ay_b[1] == 0.0)
        self.opti.subject_to(apsi_b[1] == 0.0)

        # --- THE constraint from the paper: fictitious sway force theta_tau_v ---
        # (eq. 9 / 29). This is the ONLY thing that ties the fully-actuated flat
        # model back to the real underactuated vessel. Nothing else is imposed.
        psi = ca.mtimes(B0, Ppsi)
        dx = ca.mtimes(B1, Px)
        dy = ca.mtimes(B1, Py)
        dpsi = ca.mtimes(B1, Ppsi)
        ddx = ca.mtimes(B2, Px)
        ddy = ca.mtimes(B2, Py)

        u = dx * ca.cos(psi) + dy * ca.sin(psi)
        r = dpsi
        v = -dx * ca.sin(psi) + dy * ca.cos(psi)
        dv = -ddx * ca.sin(psi) + ddy * ca.cos(psi) - u * r

        tau_v = m22_real * dv + m11_real * u * r + Yv_real * v
        self.opti.subject_to(self.opti.bounded(-self.epsilon, tau_v, self.epsilon))

        # --- Initial guess: greville-time interpolation of the waypoints ---
        g = self._greville_abscissae()
        x0 = np.zeros(self.n_ctrl)
        y0 = np.zeros(self.n_ctrl)
        psi0 = np.zeros(self.n_ctrl)
        for i, tg in enumerate(g):
            x0[i], y0[i], psi0[i] = self._waypoint_interp(tg)
        self.opti.set_initial(Px, x0)
        self.opti.set_initial(Py, y0)
        self.opti.set_initial(Ppsi, psi0)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, dt_sim):
        print("[NLP] Solving CasADi optimization for Flatness Trajectory (B-spline flat outputs)...")
        sol = self.opti.solve()
        P_sol = sol.value(self.P)  # (n_ctrl, 3)

        t_sim = np.arange(0.0, self.tf + 1e-8, dt_sim)
        t_eval = np.clip(t_sim, 0.0, self.tf - 1e-9)
        t_eval[np.isclose(t_sim, 0.0)] = 0.0

        pos = np.zeros((len(t_sim), 3))
        vel = np.zeros((len(t_sim), 3))
        acc = np.zeros((len(t_sim), 3))
        jerk = np.zeros((len(t_sim), 3))

        for j in range(3):
            spl = BSpline(self.knots, P_sol[:, j], self.degree, extrapolate=True)
            pos[:, j] = spl(t_eval, nu=0)
            vel[:, j] = spl(t_eval, nu=1)
            acc[:, j] = spl(t_eval, nu=2)
            jerk[:, j] = spl(t_eval, nu=3)

        print(f"[NLP] Solved successfully. Sampled {len(t_sim)} points.")
        return t_sim, pos, vel, acc, jerk