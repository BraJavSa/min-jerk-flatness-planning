/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 * Reconstructing psi, u, v, r, tau_u, tau_r using the complete
 * 9-parameter nonlinear hydrodynamic model.
 */
#include "flatness_reconstruct.h"
#include "usv_params.h"
#include "sqp_optimizer.h"
#include <stdlib.h>
#include <math.h>

static double clip(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd,
                    double lam_tikhonov, double r_hard_limit)
{
    double u = x_d * cos(psi) + y_d * sin(psi);
    double v = -x_d * sin(psi) + y_d * cos(psi);

    double alpha = ((m22_9 - m11_9) / m22_9) * u;
    double beta = -x_dd * sin(psi) + y_dd * cos(psi) + ((Yv_9 + Yvv_9 * fabs(v)) / m22_9) * v;

    double r = (alpha * beta) / (alpha * alpha + lam_tikhonov * lam_tikhonov);
    return clip(r, -r_hard_limit, r_hard_limit);
}

/* np.gradient: central differences interior, one-sided at the endpoints,
 * for a non-uniform (or uniform) time vector t. */
static void gradient(const double *y, const double *t, int N, double *dy)
{
    int i;
    if (N < 2) {
        if (N == 1) dy[0] = 0.0;
        return;
    }
    dy[0] = (y[1] - y[0]) / (t[1] - t[0]);
    dy[N - 1] = (y[N - 1] - y[N - 2]) / (t[N - 1] - t[N - 2]);
    for (i = 1; i < N - 1; ++i) {
        double h_left = t[i] - t[i - 1];
        double h_right = t[i + 1] - t[i];
        /* general non-uniform central difference formula (matches
         * numpy.gradient's second-order accurate scheme) */
        dy[i] = (h_left * h_left * y[i + 1] + (h_right * h_right - h_left * h_left) * y[i]
                 - h_right * h_right * y[i - 1])
                / (h_left * h_right * (h_left + h_right));
    }
}

void reconstruct_flatness_h2(const double *pos, const double *vel,
                              const double *acc, const double *jerk,
                              const double *t, int N,
                              const double *psi0_ptr,
                              double *eta, double *nu,
                              double *tau_plan, double *tau_act,
                              double *tau_u_raw, double *tau_r_raw,
                              double *cmds, double *T_plan, double *T_act)
{
    int k;
    (void)jerk; /* jerk is not used in the reconstruction dynamics itself */

    double *x_d = (double *)malloc(sizeof(double) * N);
    double *y_d = (double *)malloc(sizeof(double) * N);
    double *x_dd = (double *)malloc(sizeof(double) * N);
    double *y_dd = (double *)malloc(sizeof(double) * N);
    double *psi = (double *)malloc(sizeof(double) * N);
    double *r = (double *)malloc(sizeof(double) * N);
    double *u = (double *)malloc(sizeof(double) * N);
    double *v = (double *)malloc(sizeof(double) * N);
    double *u_dot = (double *)malloc(sizeof(double) * N);
    double *r_dot = (double *)malloc(sizeof(double) * N);
    double *T1_raw = (double *)malloc(sizeof(double) * N);
    double *T2_raw = (double *)malloc(sizeof(double) * N);
    double *T1_dem = (double *)malloc(sizeof(double) * N);
    double *T2_dem = (double *)malloc(sizeof(double) * N);
    double *cmd_1 = (double *)malloc(sizeof(double) * N);
    double *cmd_2 = (double *)malloc(sizeof(double) * N);
    double *T1_act = (double *)malloc(sizeof(double) * N);
    double *T2_act = (double *)malloc(sizeof(double) * N);

    for (k = 0; k < N; ++k) {
        x_d[k] = vel[2 * k + 0];
        y_d[k] = vel[2 * k + 1];
        x_dd[k] = acc[2 * k + 0];
        y_dd[k] = acc[2 * k + 1];
    }

    /* Replace with the bidiagonal (SQP / Newton) algebraic solve */
    optimize_psi_sqp(t, x_d, y_d, x_dd, y_dd, N, psi0_ptr, 5, psi, r);

    for (k = 0; k < N; ++k) {
        u[k] = x_d[k] * cos(psi[k]) + y_d[k] * sin(psi[k]);
        v[k] = -x_d[k] * sin(psi[k]) + y_d[k] * cos(psi[k]);
    }

    gradient(u, t, N, u_dot);
    gradient(r, t, N, r_dot);

    double tau_u_max = 2.0 * T_MAX;
    double tau_u_min = 2.0 * T_MIN;
    double tau_r_max = (T_MAX - T_MIN) * dP_9;
    double tau_r_min = -tau_r_max;

    for (k = 0; k < N; ++k) {
        /* 9-parameter complete nonlinear dynamics recovery */
        double tau_u_r = m11_9 * u_dot[k] - m22_9 * v[k] * r[k] + Xu_9 * u[k] + Xuu_9 * fabs(u[k]) * u[k];
        double tau_r_r = m33_9 * r_dot[k] - (m11_9 - m22_9) * u[k] * v[k] + Nr_9 * r[k] + Nrr_9 * fabs(r[k]) * r[k];

        tau_u_raw[k] = tau_u_r;
        tau_r_raw[k] = tau_r_r;

        T1_raw[k] = 0.5 * (tau_u_r + tau_r_r / dP_9);
        T2_raw[k] = 0.5 * (tau_u_r - tau_r_r / dP_9);

        double tau_u_v = clip(tau_u_r, tau_u_min, tau_u_max);
        double tau_r_v = clip(tau_r_r, tau_r_min, tau_r_max);

        T1_dem[k] = clip(T1_raw[k], T_MIN, T_MAX);
        T2_dem[k] = clip(T2_raw[k], T_MIN, T_MAX);

        cmd_1[k] = cmd_from_thrust_richards(T1_dem[k]);
        cmd_2[k] = cmd_from_thrust_richards(T2_dem[k]);

        T1_act[k] = thrust_from_cmd_richards(cmd_1[k]);
        T2_act[k] = thrust_from_cmd_richards(cmd_2[k]);

        double tau_u_a = T1_act[k] + T2_act[k];
        double tau_r_a = (T1_act[k] - T2_act[k]) * dP_9;

        eta[3 * k + 0] = pos[2 * k + 0];
        eta[3 * k + 1] = pos[2 * k + 1];
        eta[3 * k + 2] = psi[k];

        nu[3 * k + 0] = u[k];
        nu[3 * k + 1] = v[k];
        nu[3 * k + 2] = r[k];

        tau_plan[2 * k + 0] = tau_u_v;
        tau_plan[2 * k + 1] = tau_r_v;

        tau_act[2 * k + 0] = tau_u_a;
        tau_act[2 * k + 1] = tau_r_a;

        cmds[2 * k + 0] = cmd_1[k];
        cmds[2 * k + 1] = cmd_2[k];

        T_plan[2 * k + 0] = T1_dem[k];
        T_plan[2 * k + 1] = T2_dem[k];

        T_act[2 * k + 0] = T1_act[k];
        T_act[2 * k + 1] = T2_act[k];
    }

    free(x_d); free(y_d); free(x_dd); free(y_dd);
    free(psi); free(r); free(u); free(v);
    free(u_dot); free(r_dot);
    free(T1_raw); free(T2_raw); free(T1_dem); free(T2_dem);
    free(cmd_1); free(cmd_2); free(T1_act); free(T2_act);
}
