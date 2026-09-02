# Technique: Case 3 - Non-Linear Programming (NLP/IPOPT) Trajectory Planning + 6-Parameter Exact Flatness Reconstruction

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline
from usv_params import m11_real, m22_real, Yv_real

class FlatnessNLP:
    def __init__(self, waypoints, times, vel_start=(0.1, 0.0), vel_end=(0.01, 0.0), epsilon=0.5, dt_nlp=0.2):
        self.waypoints = np.array(waypoints)
        self.times = np.array(times)
        self.v0 = vel_start
        self.vf = vel_end
        self.dt = dt_nlp
        self.epsilon = epsilon
        
        self.N = int(np.ceil(self.times[-1] / self.dt))
        self.opti = ca.Opti()
        
        self.q = self.opti.variable(3, self.N + 1)
        self.v = self.opti.variable(3, self.N + 1)
        self.a = self.opti.variable(3, self.N + 1)
        self.j = self.opti.variable(3, self.N)
        
        cost = ca.sumsqr(self.j[0, :]) + ca.sumsqr(self.j[1, :]) + 10.0 * ca.sumsqr(self.j[2, :])
        self.opti.minimize(cost)
        
        for k in range(self.N):
            self.opti.subject_to(self.q[:, k+1] == self.q[:, k] + self.v[:, k] * self.dt + 0.5 * self.a[:, k] * self.dt**2)
            self.opti.subject_to(self.v[:, k+1] == self.v[:, k] + self.a[:, k] * self.dt + 0.5 * self.j[:, k] * self.dt**2)
            self.opti.subject_to(self.a[:, k+1] == self.a[:, k] + self.j[:, k] * self.dt)
            
            psi = self.q[2, k]
            vx, vy = self.v[0, k], self.v[1, k]
            ax, ay = self.a[0, k], self.a[1, k]
            
            u_body = vx * ca.cos(psi) + vy * ca.sin(psi)
            v_body = -vx * ca.sin(psi) + vy * ca.cos(psi)
            r_body = self.v[2, k]
            
            dv_body = -ax * ca.sin(psi) + ay * ca.cos(psi) - u_body * r_body
            
            tau_v = m22_real * dv_body + m11_real * u_body * r_body + Yv_real * v_body
            
            self.opti.subject_to(self.opti.bounded(-self.epsilon, tau_v, self.epsilon))
            
        for wp_idx, t_wp in enumerate(self.times):
            node = int(round(t_wp / self.dt))
            node = min(node, self.N)
            self.opti.subject_to(self.q[0, node] == self.waypoints[wp_idx, 0])
            self.opti.subject_to(self.q[1, node] == self.waypoints[wp_idx, 1])
            
        self.opti.subject_to(self.v[0, 0] == self.v0[0])
        self.opti.subject_to(self.v[1, 0] == self.v0[1])
        self.opti.subject_to(self.v[2, 0] == 0.0)
        self.opti.subject_to(self.a[:, 0] == [0, 0, 0])
        
        self.opti.subject_to(self.a[:, -1] == [0, 0, 0])
        
        self._set_initial_guess()
        
        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    def _set_initial_guess(self):
        for k in range(self.N + 1):
            t_k = k * self.dt
            idx = np.searchsorted(self.times, t_k)
            idx = max(1, min(idx, len(self.times) - 1))
            
            p0 = self.waypoints[idx-1]
            p1 = self.waypoints[idx]
            t0, t1 = self.times[idx-1], self.times[idx]
            
            alpha = (t_k - t0) / (max(t1 - t0, 1e-6))
            x_g = p0[0] + alpha * (p1[0] - p0[0])
            y_g = p0[1] + alpha * (p1[1] - p0[1])
            psi_g = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            
            self.opti.set_initial(self.q[0, k], x_g)
            self.opti.set_initial(self.q[1, k], y_g)
            self.opti.set_initial(self.q[2, k], psi_g)

    def sample(self, dt_sim):
        print("[NLP] Solving CasADi optimization for Flatness Trajectory...")
        sol = self.opti.solve()
        
        t_nlp = np.linspace(0, self.N * self.dt, self.N + 1)
        q_sol = sol.value(self.q)
        
        cs = CubicSpline(t_nlp, q_sol.T)
        t_sim = np.arange(0, self.times[-1] + 1e-8, dt_sim)
        
        pos = cs(t_sim)
        vel = cs(t_sim, 1)
        acc = cs(t_sim, 2)
        jerk = cs(t_sim, 3)
        
        print(f"[NLP] Solved successfully. Sampled {len(t_sim)} points.")
        return t_sim, pos, vel, acc, jerk