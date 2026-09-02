/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Direct port of c2_flatness_reconstruct.py.
 */
#include "flatness_reconstruct.h"
#include "usv_params.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static double clip(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* np.interp equivalent: linear interpolation of arr(t) sampled at grid
 * points xt[0..n-1] (assumed non-decreasing), clamping outside range. */
static double interp1(const double *xt, const double *arr, int n, double x) {
    if (x <= xt[0]) return arr[0];
    if (x >= xt[n - 1]) return arr[n - 1];
    /* linear scan is fine here (n is modest, called O(n) times per sample
     * in the RK4 stepper, matching the reference's own O(n) np.interp cost) */
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xt[mid] <= x) lo = mid; else hi = mid;
    }
    double x0 = xt[lo], x1 = xt[hi];
    double y0 = arr[lo], y1 = arr[hi];
    if (x1 == x0) return y0;
    double frac = (x - x0) / (x1 - x0);
    return y0 + frac * (y1 - y0);
}

static double psi_dot_ode(double psi, double x_d, double y_d, double x_dd, double y_dd) {
    const double lam_tikhonov = 0.015;
    const double r_hard_limit = 5.0;

    double u = x_d * cos(psi) + y_d * sin(psi);
    double v = -x_d * sin(psi) + y_d * cos(psi);

    double alpha = ((M22_6 - M11_6) / M22_6) * u;
    double beta = -x_dd * sin(psi) + y_dd * cos(psi) + (YV_6 / M22_6) * v;

    double r = (alpha * beta) / (alpha * alpha + lam_tikhonov * lam_tikhonov);
    return clip(r, -r_hard_limit, r_hard_limit);
}

/* Integrate psi via RK4 over the (possibly non-uniform) time grid t,
 * matching Python's integrate_psi. Fills psi_out[n] and r_out[n]. */
static void integrate_psi(const double *t, const double *x_d, const double *y_d,
                           const double *x_dd, const double *y_dd, int n,
                           int psi0_present, double psi0_value,
                           double *psi_out, double *r_out) {
    double psi0;
    if (psi0_present) {
        psi0 = psi0_value;
    } else {
        double speed0 = hypot(x_d[0], y_d[0]);
        psi0 = (speed0 > 0.001) ? atan2(y_d[0], x_d[0]) : 0.0;
    }
    psi_out[0] = psi0;

    for (int k = 0; k < n - 1; ++k) {
        double dt = t[k + 1] - t[k];
        double tk = t[k];

        double xk_d = x_d[k], yk_d = y_d[k];
        double xk_dd = x_dd[k], yk_dd = y_dd[k];

        double th = tk + 0.5 * dt;
        double xk_d_h = interp1(t, x_d, n, th);
        double yk_d_h = interp1(t, y_d, n, th);
        double xk_dd_h = interp1(t, x_dd, n, th);
        double yk_dd_h = interp1(t, y_dd, n, th);

        double t1 = tk + dt;
        double xk1_d = interp1(t, x_d, n, t1);
        double yk1_d = interp1(t, y_d, n, t1);
        double xk1_dd = interp1(t, x_dd, n, t1);
        double yk1_dd = interp1(t, y_dd, n, t1);

        double k1 = psi_dot_ode(psi_out[k], xk_d, yk_d, xk_dd, yk_dd);
        double k2 = psi_dot_ode(psi_out[k] + 0.5 * dt * k1, xk_d_h, yk_d_h, xk_dd_h, yk_dd_h);
        double k3 = psi_dot_ode(psi_out[k] + 0.5 * dt * k2, xk_d_h, yk_d_h, xk_dd_h, yk_dd_h);
        double k4 = psi_dot_ode(psi_out[k] + dt * k3, xk1_d, yk1_d, xk1_dd, yk1_dd);

        psi_out[k + 1] = psi_out[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    for (int k = 0; k < n; ++k) {
        r_out[k] = psi_dot_ode(psi_out[k], x_d[k], y_d[k], x_dd[k], y_dd[k]);
    }
}

/* np.gradient equivalent: central differences interior, one-sided
 * (2nd order accurate, same formula NumPy uses) at the edges, over a
 * possibly non-uniform grid t. */
static void gradient(const double *arr, const double *t, int n, double *out) {
    if (n == 1) {
        out[0] = 0.0;
        return;
    }
    /* left edge: forward difference using first two points */
    out[0] = (arr[1] - arr[0]) / (t[1] - t[0]);
    /* right edge: backward difference using last two points */
    out[n - 1] = (arr[n - 1] - arr[n - 2]) / (t[n - 1] - t[n - 2]);
    /* interior: numpy's non-uniform central difference formula */
    for (int i = 1; i < n - 1; ++i) {
        double hs = t[i] - t[i - 1];
        double hd = t[i + 1] - t[i];
        double a = -hd / (hs * (hs + hd));
        double b = (hd - hs) / (hs * hd);
        double c = hs / (hd * (hs + hd));
        out[i] = a * arr[i - 1] + b * arr[i] + c * arr[i + 1];
    }
}

void reconstruct_flatness_h2(const double *pos, const double *vel, const double *acc,
                              const double *jerk, const double *t, int n,
                              int psi0_present, double psi0_value,
                              FlatnessData *out) {
    (void)jerk; /* unused, mirrors Python signature (jerk unused there too) */

    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    double *x_d = (double *)malloc(sizeof(double) * n);
    double *y_d = (double *)malloc(sizeof(double) * n);
    double *x_dd = (double *)malloc(sizeof(double) * n);
    double *y_dd = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; ++i) {
        x[i] = pos[i * 2 + 0];
        y[i] = pos[i * 2 + 1];
        x_d[i] = vel[i * 2 + 0];
        y_d[i] = vel[i * 2 + 1];
        x_dd[i] = acc[i * 2 + 0];
        y_dd[i] = acc[i * 2 + 1];
    }

    double *psi = (double *)malloc(sizeof(double) * n);
    double *r = (double *)malloc(sizeof(double) * n);
    integrate_psi(t, x_d, y_d, x_dd, y_dd, n, psi0_present, psi0_value, psi, r);

    double *u = (double *)malloc(sizeof(double) * n);
    double *v = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        u[i] = x_d[i] * cos(psi[i]) + y_d[i] * sin(psi[i]);
        v[i] = -x_d[i] * sin(psi[i]) + y_d[i] * cos(psi[i]);
    }

    double *u_dot = (double *)malloc(sizeof(double) * n);
    double *r_dot = (double *)malloc(sizeof(double) * n);
    gradient(u, t, n, u_dot);
    gradient(r, t, n, r_dot);

    double *tau_u_raw = (double *)malloc(sizeof(double) * n);
    double *tau_r_raw = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        tau_u_raw[i] = M11_6 * u_dot[i] - M22_6 * v[i] * r[i] + XU_6 * u[i];
        tau_r_raw[i] = M33_6 * r_dot[i] - (M11_6 - M22_6) * u[i] * v[i] + NR_6 * r[i];
    }

    double *T1_raw = (double *)malloc(sizeof(double) * n);
    double *T2_raw = (double *)malloc(sizeof(double) * n);
    double *cmd1 = (double *)malloc(sizeof(double) * n);
    double *cmd2 = (double *)malloc(sizeof(double) * n);
    double *T1_act = (double *)malloc(sizeof(double) * n);
    double *T2_act = (double *)malloc(sizeof(double) * n);

    double tau_u_max = 2.0 * T_MAX;
    double tau_u_min = 2.0 * T_MIN;
    double tau_r_max = (T_MAX - T_MIN) * DP_6;
    double tau_r_min = -tau_r_max;

    double *tau_u = (double *)malloc(sizeof(double) * n);
    double *tau_r = (double *)malloc(sizeof(double) * n);
    double *tau_u_act = (double *)malloc(sizeof(double) * n);
    double *tau_r_act = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; ++i) {
        T1_raw[i] = 0.5 * (tau_u_raw[i] + tau_r_raw[i] / DP_6);
        T2_raw[i] = 0.5 * (tau_u_raw[i] - tau_r_raw[i] / DP_6);

        tau_u[i] = clip(tau_u_raw[i], tau_u_min, tau_u_max);
        tau_r[i] = clip(tau_r_raw[i], tau_r_min, tau_r_max);

        double T1_dem = clip(T1_raw[i], T_MIN, T_MAX);
        double T2_dem = clip(T2_raw[i], T_MIN, T_MAX);

        cmd1[i] = cmd_from_thrust_richards(T1_dem);
        cmd2[i] = cmd_from_thrust_richards(T2_dem);

        T1_act[i] = thrust_from_cmd_richards(cmd1[i]);
        T2_act[i] = thrust_from_cmd_richards(cmd2[i]);

        tau_u_act[i] = T1_act[i] + T2_act[i];
        tau_r_act[i] = (T1_act[i] - T2_act[i]) * DP_6;

        /* stash T1_dem/T2_dem for T_plan output */
        T1_raw[i] = T1_raw[i]; /* keep raw as-is (matches Python T1_raw semantics) */
        (void)T1_dem; (void)T2_dem;
    }

    /* Fill output struct */
    out->n = n;
    out->eta = (double *)malloc(sizeof(double) * n * 3);
    out->nu = (double *)malloc(sizeof(double) * n * 3);
    out->tau_plan = (double *)malloc(sizeof(double) * n * 2);
    out->tau_act = (double *)malloc(sizeof(double) * n * 2);
    out->tau_u_raw = (double *)malloc(sizeof(double) * n);
    out->tau_r_raw = (double *)malloc(sizeof(double) * n);
    out->cmds = (double *)malloc(sizeof(double) * n * 2);
    out->T1_raw = (double *)malloc(sizeof(double) * n);
    out->T2_raw = (double *)malloc(sizeof(double) * n);
    out->T_plan = (double *)malloc(sizeof(double) * n * 2);
    out->T_act = (double *)malloc(sizeof(double) * n * 2);

    for (int i = 0; i < n; ++i) {
        out->eta[i * 3 + 0] = x[i];
        out->eta[i * 3 + 1] = y[i];
        out->eta[i * 3 + 2] = psi[i];

        out->nu[i * 3 + 0] = u[i];
        out->nu[i * 3 + 1] = v[i];
        out->nu[i * 3 + 2] = r[i];

        out->tau_plan[i * 2 + 0] = tau_u[i];
        out->tau_plan[i * 2 + 1] = tau_r[i];

        out->tau_act[i * 2 + 0] = tau_u_act[i];
        out->tau_act[i * 2 + 1] = tau_r_act[i];

        out->tau_u_raw[i] = tau_u_raw[i];
        out->tau_r_raw[i] = tau_r_raw[i];

        out->cmds[i * 2 + 0] = cmd1[i];
        out->cmds[i * 2 + 1] = cmd2[i];

        out->T1_raw[i] = T1_raw[i];
        out->T2_raw[i] = T2_raw[i];

        double T1_dem = clip(T1_raw[i], T_MIN, T_MAX);
        double T2_dem = clip(T2_raw[i], T_MIN, T_MAX);
        out->T_plan[i * 2 + 0] = T1_dem;
        out->T_plan[i * 2 + 1] = T2_dem;

        out->T_act[i * 2 + 0] = T1_act[i];
        out->T_act[i * 2 + 1] = T2_act[i];
    }

    free(x); free(y); free(x_d); free(y_d); free(x_dd); free(y_dd);
    free(psi); free(r); free(u); free(v);
    free(u_dot); free(r_dot);
    free(tau_u_raw); free(tau_r_raw);
    free(T1_raw); free(T2_raw); free(cmd1); free(cmd2); free(T1_act); free(T2_act);
    free(tau_u); free(tau_r); free(tau_u_act); free(tau_r_act);
}

void flatness_data_free(FlatnessData *fd) {
    free(fd->eta); free(fd->nu); free(fd->tau_plan); free(fd->tau_act);
    free(fd->tau_u_raw); free(fd->tau_r_raw); free(fd->cmds);
    free(fd->T1_raw); free(fd->T2_raw); free(fd->T_plan); free(fd->T_act);
    memset(fd, 0, sizeof(*fd));
}
