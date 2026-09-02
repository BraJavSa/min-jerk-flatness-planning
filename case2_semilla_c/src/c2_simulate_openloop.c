/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 * C port of c2_simulate_openloop.py: plans with the algebraic (non-integrating)
 * psi solve, then feeds tau_act through an RK4 truth-model simulation to
 * measure open-loop tracking error. */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "c2_usv_params.h"
#include "c2_min_jerk_qp.h"
#include "c2_flatness_reconstruct.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_WP 9

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* state = [x, y, psi, u, v, r] */
static void deriv(const double s[6], double Tu, double Tr, double out[6]) {
    double psi = s[2], u = s[3], v = s[4], r = s[5];
    out[0] = u * cos(psi) - v * sin(psi);
    out[1] = u * sin(psi) + v * cos(psi);
    out[2] = r;
    out[3] = (Tu + m22_p * v * r - Xu_p * u) / m11_p;
    out[4] = (-m11_p * u * r - Yv_p * v) / m22_p;
    out[5] = (Tr + (m11_p - m22_p) * u * v - Nr_p * r) / m33_p;
}

static void rk4_step(double s[6], double Tu, double Tr, double dt) {
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    deriv(s, Tu, Tr, k1);
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5 * dt * k1[i];
    deriv(tmp, Tu, Tr, k2);
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5 * dt * k2[i];
    deriv(tmp, Tu, Tr, k3);
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + dt * k3[i];
    deriv(tmp, Tu, Tr, k4);
    for (int i = 0; i < 6; i++)
        s[i] += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

int main(void) {
    double waypoints_x[N_WP] = {0.0, 8.0, 14.0, 14.0, 20.0, 28.0, 32.0, 26.0, 20.0};
    double waypoints_y[N_WP] = {0.0, 0.0, 5.0, 13.0, 17.0, 17.0, 10.0, 4.0, 1.5};
    double base_times[N_WP]  = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
    double time_scale = 2.10;
    double times[N_WP];
    for (int i = 0; i < N_WP; i++) times[i] = base_times[i] * time_scale;

    double v0x = 0.1, v0y = 0.0;
    double dfx = waypoints_x[N_WP - 1] - waypoints_x[N_WP - 2];
    double dfy = waypoints_y[N_WP - 1] - waypoints_y[N_WP - 2];
    double dn = hypot(dfx, dfy);
    double vfx = 0.01 * dfx / dn, vfy = 0.01 * dfy / dn;

    double t_start_ms = now_ms();

    MinJerk2D planner;
    minjerk2d_solve(&planner, waypoints_x, waypoints_y, times, N_WP, v0x, v0y, vfx, vfy);

    double *t_sim, *px, *py, *vx, *vy, *ax, *ay, *jx, *jy;
    int N;
    minjerk2d_sample(&planner, DT_SIM, &t_sim, &px, &py, &vx, &vy, &ax, &ay, &jx, &jy, &N);

    FlatnessResult flat;
    reconstruct_flatness_h2(N, t_sim, px, py, vx, vy, ax, ay, &flat);

    double solve_time_ms = now_ms() - t_start_ms;

    /* --- open-loop RK4 truth-model rollout, driven by tau_act --- */
    double *real_x = malloc(sizeof(double) * (size_t)N);
    double *real_y = malloc(sizeof(double) * (size_t)N);
    double *real_psi = malloc(sizeof(double) * (size_t)N);
    double *real_u = malloc(sizeof(double) * (size_t)N);
    double *real_v = malloc(sizeof(double) * (size_t)N);
    double *real_r = malloc(sizeof(double) * (size_t)N);

    double state[6] = { flat.x[0], flat.y[0], flat.psi[0], flat.u[0], flat.v[0], flat.r[0] };
    real_x[0] = state[0]; real_y[0] = state[1]; real_psi[0] = state[2];
    real_u[0] = state[3]; real_v[0] = state[4]; real_r[0] = state[5];

    for (int i = 0; i < N - 1; i++) {
        double Tu = flat.tau_u_act[i], Tr = flat.tau_r_act[i];
        rk4_step(state, Tu, Tr, DT_SIM);
        real_x[i + 1] = state[0]; real_y[i + 1] = state[1]; real_psi[i + 1] = state[2];
        real_u[i + 1] = state[3]; real_v[i + 1] = state[4]; real_r[i + 1] = state[5];
    }

    /* --- tracking error metrics --- */
    double sum_pos2 = 0.0, max_pos = 0.0;
    double sum_psi2 = 0.0;
    double sum_u2 = 0.0, sum_v2 = 0.0, sum_r2 = 0.0;
    for (int i = 0; i < N; i++) {
        double dx = real_x[i] - flat.x[i], dy = real_y[i] - flat.y[i];
        double dpos = hypot(dx, dy);
        sum_pos2 += dpos * dpos;
        if (dpos > max_pos) max_pos = dpos;

        double dpsi = atan2(sin(real_psi[i] - flat.psi[i]), cos(real_psi[i] - flat.psi[i]));
        sum_psi2 += dpsi * dpsi;

        double du = real_u[i] - flat.u[i], dv = real_v[i] - flat.v[i], dr = real_r[i] - flat.r[i];
        sum_u2 += du * du; sum_v2 += dv * dv; sum_r2 += dr * dr;
    }
    double rmse_pos = sqrt(sum_pos2 / N);
    double rmse_psi_rad = sqrt(sum_psi2 / N);
    double rmse_psi_deg = rmse_psi_rad * 180.0 / M_PI;
    double rmse_u = sqrt(sum_u2 / N);
    double rmse_v = sqrt(sum_v2 / N);
    double rmse_r = sqrt(sum_r2 / N);

    printf("\n==================================================\n");
    printf("CASE 2 OPEN-LOOP TRAJECTORY METRICS (C port):\n");
    printf("  - Trajectory Solver Time (QP): %.4f ms\n", solve_time_ms);
    printf("  - Position RMSE:               %.4f m\n", rmse_pos);
    printf("  - Max Position Error:          %.4f m\n", max_pos);
    printf("  - Heading RMSE:                %.4f rad (%.2f deg)\n", rmse_psi_rad, rmse_psi_deg);
    printf("  - Surge Velocity (u) RMSE:     %.4f m/s\n", rmse_u);
    printf("  - Sway Velocity (v) RMSE:      %.4f m/s\n", rmse_v);
    printf("  - Yaw Rate (r) RMSE:           %.4f rad/s\n", rmse_r);
    printf("==================================================\n\n");

    /* --- c2_openloop_results.csv (planned reference) --- */
    FILE *fr = fopen("c2_openloop_results.csv", "w");
    if (fr) {
        fprintf(fr, "t,x,y,psi,u,v,r,tau_u,tau_r,T1,T2\n");
        for (int i = 0; i < N; i++)
            fprintf(fr, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i], flat.x[i], flat.y[i], flat.psi[i],
                    flat.u[i], flat.v[i], flat.r[i], flat.tau_u[i], flat.tau_r[i],
                    flat.T1_dem[i], flat.T2_dem[i]);
        fclose(fr);
    }

    /* --- c2_openloop_applied.csv (applied controls + resulting real state) --- */
    FILE *fa = fopen("c2_openloop_applied.csv", "w");
    if (fa) {
        fprintf(fa, "t,tau_u_applied,tau_r_applied,T1_applied,T2_applied,real_x,real_y,real_psi,real_u,real_v,real_r\n");
        for (int i = 0; i < N; i++) {
            double tu = (i < N - 1) ? flat.tau_u_act[i] : flat.tau_u_act[N - 2];
            double tr = (i < N - 1) ? flat.tau_r_act[i] : flat.tau_r_act[N - 2];
            double T1a = (i < N - 1) ? flat.T1_act[i] : flat.T1_act[N - 2];
            double T2a = (i < N - 1) ? flat.T2_act[i] : flat.T2_act[N - 2];
            fprintf(fa, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i], tu, tr, T1a, T2a,
                    real_x[i], real_y[i], real_psi[i], real_u[i], real_v[i], real_r[i]);
        }
        fclose(fa);
    }

    /* --- openloop_metrics.json --- */
    FILE *fm = fopen("openloop_metrics.json", "w");
    if (fm) {
        fprintf(fm,
            "{\n"
            "    \"case\": \"Case 2 Semilla (C)\",\n"
            "    \"solver_type\": \"QP (6-Param Pseudo-Flatness) - C port\",\n"
            "    \"trajectory_solver_time_ms\": %.6f,\n"
            "    \"tracking_error\": {\n"
            "        \"rmse_position_m\": %.6f,\n"
            "        \"max_position_error_m\": %.6f,\n"
            "        \"rmse_heading_rad\": %.6f,\n"
            "        \"rmse_heading_deg\": %.6f,\n"
            "        \"rmse_surge_u_mps\": %.6f,\n"
            "        \"rmse_sway_v_mps\": %.6f,\n"
            "        \"rmse_yaw_rate_r_radps\": %.6f\n"
            "    }\n"
            "}\n",
            solve_time_ms, rmse_pos, max_pos, rmse_psi_rad, rmse_psi_deg, rmse_u, rmse_v, rmse_r);
        fclose(fm);
    }

    printf("c2_openloop_results.csv, c2_openloop_applied.csv, openloop_metrics.json written.\n");
    printf("Run 'python3 plot_results.py --openloop' to regenerate the diagnostic PNG.\n");

    free(real_x); free(real_y); free(real_psi); free(real_u); free(real_v); free(real_r);
    flatness_result_free(&flat);
    free(t_sim); free(px); free(py); free(vx); free(vy); free(ax); free(ay); free(jx); free(jy);
    minjerk2d_free(&planner);
    return 0;
}
