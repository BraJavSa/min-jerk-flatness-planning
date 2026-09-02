#ifndef C2_FLATNESS_RECONSTRUCT_H
#define C2_FLATNESS_RECONSTRUCT_H

/* Algebraic yaw-rate ODE derived from the unactuated sway equation
 * (see README.md for the full derivation). Shared with c2_sqp_optimizer.c,
 * mirroring the local import in c2_flatness_reconstruct.py::optimize_psi_sqp. */
double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd);

typedef struct {
    int N;
    double *x, *y, *psi;        /* eta */
    double *u, *v, *r;          /* nu */
    double *tau_u, *tau_r;      /* tau_plan (clipped) */
    double *tau_u_raw, *tau_r_raw;
    double *T1_dem, *T2_dem;    /* T_plan */
    double *cmd1, *cmd2;
    double *T1_act, *T2_act;    /* T_act (after thruster inversion round-trip) */
    double *tau_u_act, *tau_r_act; /* tau_act */
} FlatnessResult;

void reconstruct_flatness_h2(int N, const double *t,
                              const double *x, const double *y,
                              const double *x_d, const double *y_d,
                              const double *x_dd, const double *y_dd,
                              FlatnessResult *out);
void flatness_result_free(FlatnessResult *r);

#endif /* C2_FLATNESS_RECONSTRUCT_H */
