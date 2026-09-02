#include "c2_flatness_reconstruct.h"
#include "c2_sqp_optimizer.h"
#include "c2_usv_params.h"
#include <stdlib.h>
#include <math.h>

double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd) {
    const double lam_tikhonov = 0.015, r_hard_limit = 5.0;
    double u = x_d * cos(psi) + y_d * sin(psi);
    double v = -x_d * sin(psi) + y_d * cos(psi);

    double alpha = ((m22_p - m11_p) / m22_p) * u;
    double beta = -x_dd * sin(psi) + y_dd * cos(psi) + (Yv_p / m22_p) * v;

    double r = (alpha * beta) / (alpha * alpha + lam_tikhonov * lam_tikhonov);
    if (r > r_hard_limit) r = r_hard_limit;
    if (r < -r_hard_limit) r = -r_hard_limit;
    return r;
}

static void np_gradient(int N, double dt, const double *y, double *dy) {
    if (N == 1) { dy[0] = 0.0; return; }
    dy[0] = (y[1] - y[0]) / dt;
    dy[N - 1] = (y[N - 1] - y[N - 2]) / dt;
    for (int k = 1; k < N - 1; k++) dy[k] = (y[k + 1] - y[k - 1]) / (2.0 * dt);
}

void reconstruct_flatness_h2(int N, const double *t,
                              const double *x, const double *y,
                              const double *x_d, const double *y_d,
                              const double *x_dd, const double *y_dd,
                              FlatnessResult *out) {
    out->N = N;
    out->x = malloc(sizeof(double) * (size_t)N);
    out->y = malloc(sizeof(double) * (size_t)N);
    out->psi = malloc(sizeof(double) * (size_t)N);
    out->u = malloc(sizeof(double) * (size_t)N);
    out->v = malloc(sizeof(double) * (size_t)N);
    out->r = malloc(sizeof(double) * (size_t)N);
    out->tau_u = malloc(sizeof(double) * (size_t)N);
    out->tau_r = malloc(sizeof(double) * (size_t)N);
    out->tau_u_raw = malloc(sizeof(double) * (size_t)N);
    out->tau_r_raw = malloc(sizeof(double) * (size_t)N);
    out->T1_dem = malloc(sizeof(double) * (size_t)N);
    out->T2_dem = malloc(sizeof(double) * (size_t)N);
    out->cmd1 = malloc(sizeof(double) * (size_t)N);
    out->cmd2 = malloc(sizeof(double) * (size_t)N);
    out->T1_act = malloc(sizeof(double) * (size_t)N);
    out->T2_act = malloc(sizeof(double) * (size_t)N);
    out->tau_u_act = malloc(sizeof(double) * (size_t)N);
    out->tau_r_act = malloc(sizeof(double) * (size_t)N);

    for (int i = 0; i < N; i++) { out->x[i] = x[i]; out->y[i] = y[i]; }

    /* Reemplazo por la resolucion algebraica (SQP) -- no time integration. */
    optimize_psi_sqp(N, t[1] - t[0], x_d, y_d, x_dd, y_dd, out->psi, out->r);

    for (int i = 0; i < N; i++) {
        out->u[i] = x_d[i] * cos(out->psi[i]) + y_d[i] * sin(out->psi[i]);
        out->v[i] = -x_d[i] * sin(out->psi[i]) + y_d[i] * cos(out->psi[i]);
    }

    double *u_dot = malloc(sizeof(double) * (size_t)N);
    double *r_dot = malloc(sizeof(double) * (size_t)N);
    double dt = t[1] - t[0];
    np_gradient(N, dt, out->u, u_dot);
    np_gradient(N, dt, out->r, r_dot);

    double tau_u_max = 2.0 * T_MAX, tau_u_min = 2.0 * T_MIN;
    double tau_r_max = (T_MAX - T_MIN) * dP_p, tau_r_min = -tau_r_max;

    for (int i = 0; i < N; i++) {
        double tau_u_raw = m11_p * u_dot[i] - m22_p * out->v[i] * out->r[i] + Xu_p * out->u[i];
        double tau_r_raw = m33_p * r_dot[i] - (m11_p - m22_p) * out->u[i] * out->v[i] + Nr_p * out->r[i];
        out->tau_u_raw[i] = tau_u_raw;
        out->tau_r_raw[i] = tau_r_raw;

        double tu = tau_u_raw, tr = tau_r_raw;
        if (tu > tau_u_max) tu = tau_u_max;
        if (tu < tau_u_min) tu = tau_u_min;
        if (tr > tau_r_max) tr = tau_r_max;
        if (tr < tau_r_min) tr = tau_r_min;
        out->tau_u[i] = tu;
        out->tau_r[i] = tr;

        double T1_raw = 0.5 * (tau_u_raw + tau_r_raw / dP_p);
        double T2_raw = 0.5 * (tau_u_raw - tau_r_raw / dP_p);
        double T1d = T1_raw, T2d = T2_raw;
        if (T1d > T_MAX) T1d = T_MAX;
        if (T1d < T_MIN) T1d = T_MIN;
        if (T2d > T_MAX) T2d = T_MAX;
        if (T2d < T_MIN) T2d = T_MIN;
        out->T1_dem[i] = T1d;
        out->T2_dem[i] = T2d;

        double cmd1 = cmd_from_thrust_richards(T1d);
        double cmd2 = cmd_from_thrust_richards(T2d);
        out->cmd1[i] = cmd1;
        out->cmd2[i] = cmd2;

        double T1_act = thrust_from_cmd_richards(cmd1);
        double T2_act = thrust_from_cmd_richards(cmd2);
        out->T1_act[i] = T1_act;
        out->T2_act[i] = T2_act;

        out->tau_u_act[i] = T1_act + T2_act;
        out->tau_r_act[i] = (T1_act - T2_act) * dP_p;
    }

    free(u_dot); free(r_dot);
}

void flatness_result_free(FlatnessResult *r) {
    free(r->x); free(r->y); free(r->psi);
    free(r->u); free(r->v); free(r->r);
    free(r->tau_u); free(r->tau_r);
    free(r->tau_u_raw); free(r->tau_r_raw);
    free(r->T1_dem); free(r->T2_dem);
    free(r->cmd1); free(r->cmd2);
    free(r->T1_act); free(r->T2_act);
    free(r->tau_u_act); free(r->tau_r_act);
}
