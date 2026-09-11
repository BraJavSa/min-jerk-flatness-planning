#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd,
                    double lam_tikhonov, double r_hard_limit);

void reconstruct_flatness_h2(const double *pos, const double *vel,
                              const double *acc, const double *jerk,
                              const double *t, int N,
                              const double *psi0_ptr,
                              double *eta, double *nu,
                              double *tau_plan, double *tau_act,
                              double *tau_u_raw, double *tau_r_raw,
                              double *cmds, double *T_plan, double *T_act);

#endif
