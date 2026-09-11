#include <stdlib.h>
#include <math.h>
#include "usv_params.h"
#include "flatness_reconstruct.h"

FlatSample reconstruct_flatness_h2_point(double x, double y,
                                          double dx, double dy,
                                          double ddx, double ddy,
                                          double psi, double r, double dr) {
    FlatSample s;

    double u = dx * cos(psi) + dy * sin(psi);
    double v = -dx * sin(psi) + dy * cos(psi);

    double tau_u_raw = m_5 * (ddx * cos(psi) + ddy * sin(psi)) + Xu_5 * u;
    double tau_r_raw = m33_5 * dr + Nr_5 * r;

    double T1_raw = 0.5 * (tau_u_raw / SURGE_GAIN + tau_r_raw / (2.0 * YAW_ARM));
    double T2_raw = 0.5 * (tau_u_raw / SURGE_GAIN - tau_r_raw / (2.0 * YAW_ARM));

    double tau_u_max = SURGE_GAIN * 2.0 * T_MAX;
    double tau_u_min = SURGE_GAIN * 2.0 * T_MIN;
    double tau_r_max = 2.0 * YAW_ARM * (T_MAX - T_MIN);
    double tau_r_min = -tau_r_max;

    double tau_u = clip(tau_u_raw, tau_u_min, tau_u_max);
    double tau_r = clip(tau_r_raw, tau_r_min, tau_r_max);

    double T1_dem = clip(T1_raw, T_MIN, T_MAX);
    double T2_dem = clip(T2_raw, T_MIN, T_MAX);

    double cmd_1 = cmd_from_thrust(T1_dem);
    double cmd_2 = cmd_from_thrust(T2_dem);

    double T1_act = thrust_from_cmd(cmd_1);
    double T2_act = thrust_from_cmd(cmd_2);

    double tau_u_act = SURGE_GAIN * (T1_act + T2_act);
    double tau_r_act = 2.0 * YAW_ARM * (T1_act - T2_act);

    s.eta[0] = x; s.eta[1] = y; s.eta[2] = psi;
    s.nu[0] = u; s.nu[1] = v; s.nu[2] = r;
    s.tau_plan[0] = tau_u; s.tau_plan[1] = tau_r;
    s.tau_act[0] = tau_u_act; s.tau_act[1] = tau_r_act;
    s.T_plan[0] = T1_dem; s.T_plan[1] = T2_dem;
    s.T_act[0] = T1_act; s.T_act[1] = T2_act;
    return s;
}

void reconstruct_flatness_h2(const double *x, const double *y,
                              const double *dx, const double *dy,
                              const double *ddx, const double *ddy,
                              int n, const double *t,
                              FlatSample *out) {
    double *psi_raw = (double *)malloc(sizeof(double) * n);
    double *psi = (double *)malloc(sizeof(double) * n);
    double *r = (double *)malloc(sizeof(double) * n);
    double *dr = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        double A = ddy[i] + (Yv_5 / m_5) * dy[i];
        double B = ddx[i] + (Yv_5 / m_5) * dx[i];
        double mag = hypot(A, B);
        if (mag > 0.05) {
            psi_raw[i] = atan2(A, B);
        } else {
            double speed = hypot(dx[i], dy[i]);
            if (speed > 0.05) psi_raw[i] = atan2(dy[i], dx[i]);
            else if (i > 0) psi_raw[i] = psi_raw[i - 1];
            else psi_raw[i] = 0.0;
        }
    }

    psi[0] = psi_raw[0];
    double cum = 0.0;
    for (int i = 1; i < n; i++) {
        double d = psi_raw[i] - psi_raw[i - 1];
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        cum += d;
        psi[i] = psi_raw[0] + cum;
    }

    double h = (n > 1) ? (t[1] - t[0]) : 1.0;
    for (int i = 0; i < n; i++) {
        if (n == 1) { r[i] = 0.0; continue; }
        if (i == 0) r[i] = (-3.0 * psi[0] + 4.0 * psi[1] - psi[2]) / (2.0 * h);
        else if (i == n - 1) r[i] = (3.0 * psi[n - 1] - 4.0 * psi[n - 2] + psi[n - 3]) / (2.0 * h);
        else r[i] = (psi[i + 1] - psi[i - 1]) / (t[i + 1] - t[i - 1]);
    }
    for (int i = 0; i < n; i++) {
        if (n == 1) { dr[i] = 0.0; continue; }
        if (i == 0) dr[i] = (-3.0 * r[0] + 4.0 * r[1] - r[2]) / (2.0 * h);
        else if (i == n - 1) dr[i] = (3.0 * r[n - 1] - 4.0 * r[n - 2] + r[n - 3]) / (2.0 * h);
        else dr[i] = (r[i + 1] - r[i - 1]) / (t[i + 1] - t[i - 1]);
    }

    for (int i = 0; i < n; i++) {
        out[i] = reconstruct_flatness_h2_point(x[i], y[i], dx[i], dy[i], ddx[i], ddy[i],
                                                psi[i], r[i], dr[i]);
    }

    free(psi_raw); free(psi); free(r); free(dr);
}
