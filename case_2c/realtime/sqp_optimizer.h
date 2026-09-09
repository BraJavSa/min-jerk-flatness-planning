/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * SQP-style algebraic solver for psi(t), replacing the RK4/NLP heading
 * reconstruction with a bidiagonal linear solve per Newton iteration.
 */
#ifndef SQP_OPTIMIZER_H
#define SQP_OPTIMIZER_H

/*
 * Solve for psi(t) and r(t) = psi_dot(t) given the flat outputs
 * (x_d, y_d, x_dd, y_dd) sampled at N uniform time steps t[0..N-1].
 *
 * psi0_ptr: pointer to an initial heading value; pass NULL to auto-select
 *           psi0 = atan2(y_d[0], x_d[0]) (or 0 if speed is near zero).
 * max_iters: maximum number of Newton/SQP iterations (Python default: 5).
 *
 * psi_out, r_out: pre-allocated arrays of length N, filled with the
 *                 solved heading and yaw-rate profiles.
 */
void optimize_psi_sqp(const double *t, const double *x_d, const double *y_d,
                       const double *x_dd, const double *y_dd, int N,
                       const double *psi0_ptr, int max_iters,
                       double *psi_out, double *r_out);

#endif /* SQP_OPTIMIZER_H */
