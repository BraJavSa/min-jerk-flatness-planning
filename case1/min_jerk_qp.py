# Technique: Case 1 - Minimum Jerk QP Trajectory Planning + 5-Parameter Exact Flatness Reconstruction (m11 = m22)

import numpy as np

class MinJerkTrajectory1D:
    POLY_ORDER = 5

    def __init__(self, waypoints, times, v0=0.0, vf=0.0):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.v0 = float(v0)
        self.vf = float(vf)
        self.n_seg = len(waypoints) - 1
        assert self.n_seg >= 1, "Requires at least 2 waypoints"
        assert len(times) == len(waypoints)
        self.coeffs = self._solve()

    def _poly_basis(self, tau, order=0):
        n = self.POLY_ORDER
        basis = np.zeros(n + 1)
        for i in range(n + 1):
            if i < order:
                continue
            coeff = 1.0
            for k in range(order):
                coeff *= (i - k)
            basis[i] = coeff * (tau ** (i - order))
        return basis

    def _solve(self):
        n_seg = self.n_seg
        n_coef = self.POLY_ORDER + 1
        n_vars = n_seg * n_coef
        
        H = np.zeros((n_vars, n_vars))
        for k in range(n_seg):
            T = self.times[k + 1] - self.times[k]
            H[k * n_coef:(k + 1) * n_coef, k * n_coef:(k + 1) * n_coef] = self._segment_cost_matrix(T)
            
        A_rows = []
        b_vals = []
        
        for k in range(n_seg):
            T = self.times[k + 1] - self.times[k]
            
            row = np.zeros(n_vars)
            row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(0.0, order=0)
            A_rows.append(row)
            b_vals.append(self.waypoints[k])
            
            row = np.zeros(n_vars)
            row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(T, order=0)
            A_rows.append(row)
            b_vals.append(self.waypoints[k + 1])
            
        row = np.zeros(n_vars)
        row[0:n_coef] = self._poly_basis(0.0, order=1)
        A_rows.append(row)
        b_vals.append(self.v0)
        
        T_last = self.times[-1] - self.times[-2]
        row = np.zeros(n_vars)
        row[(n_seg - 1) * n_coef:n_seg * n_coef] = self._poly_basis(T_last, order=1)
        A_rows.append(row)
        b_vals.append(self.vf)
        
        row = np.zeros(n_vars)
        row[0:n_coef] = self._poly_basis(0.0, order=2)
        A_rows.append(row)
        b_vals.append(0.0)
        
        row = np.zeros(n_vars)
        row[(n_seg - 1) * n_coef:n_seg * n_coef] = self._poly_basis(T_last, order=2)
        A_rows.append(row)
        b_vals.append(0.0)
            
        for k in range(n_seg - 1):
            T = self.times[k + 1] - self.times[k]
            for order in (1, 2):
                row = np.zeros(n_vars)
                row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(T, order=order)
                row[(k + 1) * n_coef:(k + 2) * n_coef] = -self._poly_basis(0.0, order=order)
                A_rows.append(row)
                b_vals.append(0.0)
                
        A = np.array(A_rows)
        b = np.array(b_vals)
        m = A.shape[0]
        
        KKT = np.zeros((n_vars + m, n_vars + m))
        KKT[:n_vars, :n_vars] = H + 1e-8 * np.eye(n_vars)
        KKT[:n_vars, n_vars:] = A.T
        KKT[n_vars:, :n_vars] = A
        rhs = np.concatenate([np.zeros(n_vars), b])
        
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        c = sol[:n_vars]
        return c.reshape(n_seg, n_coef)

    def _segment_cost_matrix(self, T):
        n = self.POLY_ORDER + 1
        H = np.zeros((n, n))
        for i in range(3, n):
            for j in range(3, n):
                ci = i * (i - 1) * (i - 2)
                cj = j * (j - 1) * (j - 2)
                power = i - 3 + (j - 3) + 1
                H[i, j] = ci * cj * (T ** power) / power
        return H

    def eval(self, t, order=0):
        k = np.searchsorted(self.times, t, side='right') - 1
        k = min(max(k, 0), self.n_seg - 1)
        tau = t - self.times[k]
        basis = self._poly_basis(tau, order=order)
        return float(self.coeffs[k] @ basis)

class MinJerkTrajectory2D:
    def __init__(self, waypoints_xy, times, vel_start=(0.1, 0.0), vel_end=(0.0, 0.0)):
        waypoints_xy = np.asarray(waypoints_xy, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.traj_x = MinJerkTrajectory1D(waypoints_xy[:, 0], times, v0=vel_start[0], vf=vel_end[0])
        self.traj_y = MinJerkTrajectory1D(waypoints_xy[:, 1], times, v0=vel_start[1], vf=vel_end[1])

    def eval(self, t, order=0):
        return np.array([self.traj_x.eval(t, order), self.traj_y.eval(t, order)])

    def sample(self, dt=1.0/30.0):
        t0, tf = self.times[0], self.times[-1]
        ts = np.arange(t0, tf + 1e-8, dt)
        pos = np.array([self.eval(t, 0) for t in ts])
        vel = np.array([self.eval(t, 1) for t in ts])
        acc = np.array([self.eval(t, 2) for t in ts])
        jerk = np.array([self.eval(t, 3) for t in ts])
        return ts, pos, vel, acc, jerk

