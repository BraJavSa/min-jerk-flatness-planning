#ifndef C2_SQP_OPTIMIZER_H
#define C2_SQP_OPTIMIZER_H

/* Implicit (non-integrating) solve for psi(t) over the whole horizon.
 * Mirrors c2_sqp_optimizer.py::optimize_psi_sqp: backward-Euler
 * discretization of the algebraic ODE psi_dot = r(psi), relinearized
 * (numerical Jacobian) up to 5 times. The linear system is lower
 * bidiagonal, so each iteration is solved via O(N) forward substitution
 * (no sparse factorization needed). */
void optimize_psi_sqp(int N, double dt, const double *xd, const double *yd,
                       const double *xdd, const double *ydd,
                       double *psi_out, double *r_out);

#endif /* C2_SQP_OPTIMIZER_H */
