/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 * Reconstructing psi, u, v, r, tau_u, tau_r using the complete
 * 9-parameter nonlinear hydrodynamic model.
 */
#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

/*
 * Pseudo-Flat first-order ODE for psi(t) retaining the complete nonlinear
 * damping model:
 *   alpha(psi) = ((m22 - m11) / m22) * u
 *   beta(psi)  = -x_dd*sin(psi) + y_dd*cos(psi) + ((Yv + Yvv*|v|)/m22) * v
 *   r = (alpha * beta) / (alpha^2 + lambda^2)
 */
double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd,
                    double lam_tikhonov, double r_hard_limit);

/*
 * Full flatness reconstruction over N samples.
 *
 * Inputs (each length N, x/y interleaved for 2D arrays):
 *   pos, vel, acc, jerk : [2*N] interleaved (x0,y0,x1,y1,...)
 *   t                   : [N] time stamps
 *   psi0_ptr            : optional initial heading (NULL => auto)
 *
 * Outputs (all pre-allocated by caller, length N unless noted):
 *   eta      [3*N] interleaved (x, y, psi)
 *   nu       [3*N] interleaved (u, v, r)
 *   tau_plan [2*N] interleaved (tau_u, tau_r)   -- clipped/planned
 *   tau_act  [2*N] interleaved (tau_u_act, tau_r_act) -- from actuated thrust
 *   tau_u_raw, tau_r_raw [N]  -- unclipped dynamics-recovered torques
 *   cmds     [2*N] interleaved (cmd_1, cmd_2)
 *   T_plan   [2*N] interleaved (T1_dem, T2_dem)
 *   T_act    [2*N] interleaved (T1_act, T2_act)
 */
void reconstruct_flatness_h2(const double *pos, const double *vel,
                              const double *acc, const double *jerk,
                              const double *t, int N,
                              const double *psi0_ptr,
                              double *eta, double *nu,
                              double *tau_plan, double *tau_act,
                              double *tau_u_raw, double *tau_r_raw,
                              double *cmds, double *T_plan, double *T_act);

#endif /* FLATNESS_RECONSTRUCT_H */
