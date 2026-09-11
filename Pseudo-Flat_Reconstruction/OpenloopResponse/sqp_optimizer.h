#ifndef SQP_OPTIMIZER_H
#define SQP_OPTIMIZER_H

void optimize_psi_sqp(const double *t, const double *x_d, const double *y_d,
                       const double *x_dd, const double *y_dd, int N,
                       const double *psi0_ptr, int max_iters,
                       double *psi_out, double *r_out);

#endif
