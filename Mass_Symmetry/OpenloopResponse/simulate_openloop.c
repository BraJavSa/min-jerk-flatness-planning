#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "usv_params.h"
#include "min_jerk_qp.h"
#include "flatness_reconstruct.h"

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void full_dynamics_deriv(const double s[6], double Tu, double Tr, double out[6]) {
    double psi_i = s[2], u_i = s[3], v_i = s[4], r_i = s[5];
    out[0] = u_i * cos(psi_i) - v_i * sin(psi_i);
    out[1] = u_i * sin(psi_i) + v_i * cos(psi_i);
    out[2] = r_i;
    out[3] = (Tu + m22_full * v_i * r_i - Xu_full * u_i - Xuu_full * fabs(u_i) * u_i) / m11_full;
    out[4] = (-m11_full * u_i * r_i - Yv_full * v_i - Yvv_full * fabs(v_i) * v_i) / m22_full;
    out[5] = (Tr - (m22_full - m11_full) * u_i * v_i - Nr_full * r_i - Nrr_full * fabs(r_i) * r_i) / m33_full;
}

static void rk4_step(double state[6], double Tu, double Tr, double dt) {
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    full_dynamics_deriv(state, Tu, Tr, k1);
    for (int i = 0; i < 6; i++) tmp[i] = state[i] + 0.5 * dt * k1[i];
    full_dynamics_deriv(tmp, Tu, Tr, k2);
    for (int i = 0; i < 6; i++) tmp[i] = state[i] + 0.5 * dt * k2[i];
    full_dynamics_deriv(tmp, Tu, Tr, k3);
    for (int i = 0; i < 6; i++) tmp[i] = state[i] + dt * k3[i];
    full_dynamics_deriv(tmp, Tu, Tr, k4);
    for (int i = 0; i < 6; i++)
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

int main(void) {
    double wp_x[] = {0.0, 8.0, 14.0, 14.0, 20.0, 28.0, 32.0, 26.0, 20.0};
    double wp_y[] = {0.0, 0.0, 5.0,  13.0, 17.0, 17.0, 10.0, 4.0,  1.5};
    int n_wp = 9;

    double base_times[] = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
    double time_scale = 2.10;
    double times[9];
    for (int i = 0; i < n_wp; i++) times[i] = base_times[i] * time_scale;

    double v0x = 0.1, v0y = 0.0;
    double dir_fx = wp_x[8] - wp_x[7];
    double dir_fy = wp_y[8] - wp_y[7];
    double dir_norm = hypot(dir_fx, dir_fy);
    double vfx = 0.01 * dir_fx / dir_norm;
    double vfy = 0.01 * dir_fy / dir_norm;

    double t_qp0 = now_ms();
    MinJerk2D planner;
    minjerk2d_solve(&planner, wp_x, wp_y, times, n_wp, v0x, v0y, vfx, vfy);
    double qp_time_ms = now_ms() - t_qp0;

    double t0 = times[0], tf = times[n_wp - 1];
    int n_steps = (int)floor((tf - t0) / DT_SIM + 1e-8) + 1;

    double *t_sim = malloc(sizeof(double) * n_steps);
    double *x = malloc(sizeof(double) * n_steps);
    double *y = malloc(sizeof(double) * n_steps);
    double *dx = malloc(sizeof(double) * n_steps);
    double *dy = malloc(sizeof(double) * n_steps);
    double *ddx = malloc(sizeof(double) * n_steps);
    double *ddy = malloc(sizeof(double) * n_steps);

    for (int i = 0; i < n_steps; i++) {
        double t = t0 + i * DT_SIM;
        t_sim[i] = t;
        x[i] = minjerk1d_eval(&planner.tx, t, 0);
        y[i] = minjerk1d_eval(&planner.ty, t, 0);
        dx[i] = minjerk1d_eval(&planner.tx, t, 1);
        dy[i] = minjerk1d_eval(&planner.ty, t, 1);
        ddx[i] = minjerk1d_eval(&planner.tx, t, 2);
        ddy[i] = minjerk1d_eval(&planner.ty, t, 2);
    }

    FlatSample *flat = malloc(sizeof(FlatSample) * n_steps);
    reconstruct_flatness_h2(x, y, dx, dy, ddx, ddy, n_steps, t_sim, flat);

    double state_real[6] = { flat[0].eta[0], flat[0].eta[1], flat[0].eta[2],
                              flat[0].nu[0],  flat[0].nu[1],  flat[0].nu[2] };

    double sum_pos2 = 0.0, max_pos = 0.0, sum_psi2 = 0.0, sum_u2 = 0.0, sum_v2 = 0.0, sum_r2 = 0.0;

    double *real_x = malloc(sizeof(double) * n_steps);
    double *real_y = malloc(sizeof(double) * n_steps);
    double *real_psi = malloc(sizeof(double) * n_steps);
    double *real_u = malloc(sizeof(double) * n_steps);
    double *real_v = malloc(sizeof(double) * n_steps);
    double *real_r = malloc(sizeof(double) * n_steps);

    real_x[0] = state_real[0]; real_y[0] = state_real[1]; real_psi[0] = state_real[2];
    real_u[0] = state_real[3]; real_v[0] = state_real[4]; real_r[0] = state_real[5];

    double t_sim0 = now_ms();
    {
        double ex = state_real[0] - flat[0].eta[0];
        double ey = state_real[1] - flat[0].eta[1];
        double pos_err = hypot(ex, ey);
        sum_pos2 += pos_err * pos_err;
        if (pos_err > max_pos) max_pos = pos_err;
    }
    for (int i = 0; i < n_steps - 1; i++) {
        double Tu = flat[i].tau_act[0];
        double Tr = flat[i].tau_act[1];
        rk4_step(state_real, Tu, Tr, DT_SIM);

        real_x[i + 1] = state_real[0]; real_y[i + 1] = state_real[1]; real_psi[i + 1] = state_real[2];
        real_u[i + 1] = state_real[3]; real_v[i + 1] = state_real[4]; real_r[i + 1] = state_real[5];

        double ex = state_real[0] - flat[i + 1].eta[0];
        double ey = state_real[1] - flat[i + 1].eta[1];
        double pos_err = hypot(ex, ey);
        sum_pos2 += pos_err * pos_err;
        if (pos_err > max_pos) max_pos = pos_err;

        double dpsi = state_real[2] - flat[i + 1].eta[2];
        double psi_err = atan2(sin(dpsi), cos(dpsi));
        sum_psi2 += psi_err * psi_err;

        double eu = state_real[3] - flat[i + 1].nu[0];
        double ev = state_real[4] - flat[i + 1].nu[1];
        double er = state_real[5] - flat[i + 1].nu[2];
        sum_u2 += eu * eu; sum_v2 += ev * ev; sum_r2 += er * er;
    }
    double sim_time_ms = now_ms() - t_sim0;

    double rmse_pos = sqrt(sum_pos2 / n_steps);
    double rmse_psi_rad = sqrt(sum_psi2 / n_steps);
    double rmse_psi_deg = rmse_psi_rad * 180.0 / M_PI;
    double rmse_u = sqrt(sum_u2 / n_steps);
    double rmse_v = sqrt(sum_v2 / n_steps);
    double rmse_r = sqrt(sum_r2 / n_steps);

    printf("==================================================\n");
    printf("CASE 1 (C) - Open-loop validation: 5-param QP vs 9-param plant\n");
    printf("--------------------------------------------------\n");
    printf("Samples:                         %d (dt = %.5f s, T = %.3f s)\n",
           n_steps, DT_SIM, t_sim[n_steps - 1] - t_sim[0]);
    printf("QP Planning Time:                %.5f ms\n", qp_time_ms);
    printf("Simulation Time (RK4):           %.5f ms\n", sim_time_ms);
    printf("--------------------------------------------------\n");
    printf("Position RMSE:                    %.4f m\n", rmse_pos);
    printf("Max position error:               %.4f m\n", max_pos);
    printf("Heading RMSE:                      %.4f rad (%.2f deg)\n", rmse_psi_rad, rmse_psi_deg);
    printf("Surge (u) RMSE:                    %.4f m/s\n", rmse_u);
    printf("Sway  (v) RMSE:                    %.4f m/s\n", rmse_v);
    printf("Yaw rate (r) RMSE:                  %.4f rad/s\n", rmse_r);
    printf("==================================================\n");

    const char *json_path = "openloop_metrics.json";
    FILE *fj = fopen(json_path, "w");
    if (fj) {
        fprintf(fj, "{\n");
        fprintf(fj, "    \"case\": \"Case 1 Open-Loop (C)\",\n");
        fprintf(fj, "    \"solver_type\": \"QP (5-Param Model) -> 9-Param Plant\",\n");
        fprintf(fj, "    \"position_rmse_m\": %.6f,\n", rmse_pos);
        fprintf(fj, "    \"max_position_error_m\": %.6f,\n", max_pos);
        fprintf(fj, "    \"heading_rmse_rad\": %.6f,\n", rmse_psi_rad);
        fprintf(fj, "    \"heading_rmse_deg\": %.4f,\n", rmse_psi_deg);
        fprintf(fj, "    \"surge_rmse_mps\": %.6f,\n", rmse_u);
        fprintf(fj, "    \"sway_rmse_mps\": %.6f,\n", rmse_v);
        fprintf(fj, "    \"yaw_rate_rmse_radps\": %.6f,\n", rmse_r);
        fprintf(fj, "    \"qp_time_ms\": %.5f,\n", qp_time_ms);
        fprintf(fj, "    \"simulation_time_ms\": %.5f,\n", sim_time_ms);
        fprintf(fj, "    \"tiempo_qp_ms\": %.5f,\n", qp_time_ms);
        fprintf(fj, "    \"tiempo_simulacion_ms\": %.5f,\n", sim_time_ms);
        fprintf(fj, "    \"total_sim_time_s\": %.5f,\n", t_sim[n_steps - 1] - t_sim[0]);
        fprintf(fj, "    \"num_samples\": %d\n", n_steps);
        fprintf(fj, "}\n");
        fclose(fj);
        printf("Metrics saved to: %s\n", json_path);
    }

    const char *csv_path = "case1_openloop_results.csv";
    FILE *fp = fopen(csv_path, "w");
    if (fp) {
        fprintf(fp, "t,x_plan,y_plan,psi_plan,u_plan,v_plan,r_plan,"
                     "tau_u_plan,tau_r_plan,T1_plan,T2_plan,"
                     "x_real,y_real,psi_real,u_real,v_real,r_real,"
                     "tau_u_applied,tau_r_applied,T1_applied,T2_applied\n");
        for (int i = 0; i < n_steps; i++) {
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i],
                    flat[i].eta[0], flat[i].eta[1], flat[i].eta[2],
                    flat[i].nu[0], flat[i].nu[1], flat[i].nu[2],
                    flat[i].tau_plan[0], flat[i].tau_plan[1],
                    flat[i].T_plan[0], flat[i].T_plan[1],
                    real_x[i], real_y[i], real_psi[i], real_u[i], real_v[i], real_r[i],
                    flat[i].tau_act[0], flat[i].tau_act[1],
                    flat[i].T_act[0], flat[i].T_act[1]);
        }
        fclose(fp);
        printf("CSV saved to: %s\n", csv_path);
    } else {
        fprintf(stderr, "Could not write to %s\n", csv_path);
    }

    minjerk2d_free(&planner);
    free(t_sim); free(x); free(y); free(dx); free(dy); free(ddx); free(ddy);
    free(real_x); free(real_y); free(real_psi); free(real_u); free(real_v); free(real_r);
    free(flat);
    return 0;
}
