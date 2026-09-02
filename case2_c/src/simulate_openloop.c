/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Direct port of c2_simulate_openloop.py.
 */
#include "min_jerk_qp.h"
#include "flatness_reconstruct.h"
#include "usv_params.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* state = [x, y, psi, u, v, r] */
static void plant_deriv(const double state[6], double Tu, double Tr, double d[6]) {
    double p_i = state[2], u_i = state[3], v_i = state[4], r_i = state[5];
    d[0] = u_i * cos(p_i) - v_i * sin(p_i);
    d[1] = u_i * sin(p_i) + v_i * cos(p_i);
    d[2] = r_i;
    d[3] = (Tu + M22_REAL * v_i * r_i - XU_REAL * u_i) / M11_REAL;
    d[4] = (-M11_REAL * u_i * r_i - YV_REAL * v_i) / M22_REAL;
    d[5] = (Tr + (M11_REAL - M22_REAL) * u_i * v_i - NR_REAL * r_i) / M33_REAL;
}

static void real_6param_rk4_step(const double state[6], double Tu, double Tr, double dt, double out[6]) {
    double k1[6], k2[6], k3[6], k4[6], tmp[6];

    plant_deriv(state, Tu, Tr, k1);
    for (int i = 0; i < 6; ++i) tmp[i] = state[i] + 0.5 * dt * k1[i];
    plant_deriv(tmp, Tu, Tr, k2);
    for (int i = 0; i < 6; ++i) tmp[i] = state[i] + 0.5 * dt * k2[i];
    plant_deriv(tmp, Tu, Tr, k3);
    for (int i = 0; i < 6; ++i) tmp[i] = state[i] + dt * k3[i];
    plant_deriv(tmp, Tu, Tr, k4);

    for (int i = 0; i < 6; ++i) {
        out[i] = state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
}

int main(void) {
    double waypoints[9][2] = {
        {0.0, 0.0}, {8.0, 0.0}, {14.0, 5.0}, {14.0, 13.0}, {20.0, 17.0},
        {28.0, 17.0}, {32.0, 10.0}, {26.0, 4.0}, {20.0, 1.5}
    };
    int n_wp = 9;

    double base_times[9] = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
    double time_scale = 2.10;
    double times[9];
    for (int i = 0; i < n_wp; ++i) times[i] = base_times[i] * time_scale;

    double v0_vec[2] = {0.1, 0.0};
    double dir_f[2] = {waypoints[8][0] - waypoints[7][0], waypoints[8][1] - waypoints[7][1]};
    double dir_norm = hypot(dir_f[0], dir_f[1]);
    double vf_vec[2] = {0.01 * dir_f[0] / dir_norm, 0.01 * dir_f[1] / dir_norm};

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    double wp_flat[18];
    for (int i = 0; i < n_wp; ++i) {
        wp_flat[i * 2 + 0] = waypoints[i][0];
        wp_flat[i * 2 + 1] = waypoints[i][1];
    }
    MinJerkTrajectory2D planner;
    int rc = mj2d_init(&planner, wp_flat, times, n_wp, v0_vec, vf_vec);
    if (rc != 0) {
        fprintf(stderr, "Trajectory solve failed (LAPACK info=%d)\n", rc);
        return 1;
    }

    double *t_sim, *pos, *vel, *acc, *jerk;
    int n = mj2d_sample(&planner, DT_SIM, &t_sim, &pos, &vel, &acc, &jerk);

    FlatnessData fd;
    reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim, n, 0, 0.0, &fd);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double solve_time_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 +
                            (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e6;

    /* --- Open-loop RK4 rollout of the "real" plant driven by tau_act --- */
    double *state_real = (double *)malloc(sizeof(double) * n * 6);
    double *tau_applied = (double *)malloc(sizeof(double) * (n - 1) * 2);
    double *T_applied = (double *)malloc(sizeof(double) * (n - 1) * 2);

    /* initial state = [eta_ref[0], nu_ref[0]] */
    state_real[0] = fd.eta[0]; state_real[1] = fd.eta[1]; state_real[2] = fd.eta[2];
    state_real[3] = fd.nu[0]; state_real[4] = fd.nu[1]; state_real[5] = fd.nu[2];

    for (int i = 0; i < n - 1; ++i) {
        double Tu = fd.tau_act[i * 2 + 0];
        double Tr = fd.tau_act[i * 2 + 1];
        tau_applied[i * 2 + 0] = Tu;
        tau_applied[i * 2 + 1] = Tr;
        T_applied[i * 2 + 0] = fd.T_act[i * 2 + 0];
        T_applied[i * 2 + 1] = fd.T_act[i * 2 + 1];

        double next[6];
        real_6param_rk4_step(&state_real[i * 6], Tu, Tr, DT_SIM, next);
        memcpy(&state_real[(i + 1) * 6], next, sizeof(double) * 6);
    }

    /* --- Metrics --- */
    double sum_pos_err2 = 0.0, max_pos_err = 0.0;
    double sum_psi_err2 = 0.0;
    double sum_u_err2 = 0.0, sum_v_err2 = 0.0, sum_r_err2 = 0.0;

    for (int i = 0; i < n; ++i) {
        double dx = state_real[i * 6 + 0] - fd.eta[i * 3 + 0];
        double dy = state_real[i * 6 + 1] - fd.eta[i * 3 + 1];
        double pos_err = hypot(dx, dy);
        sum_pos_err2 += pos_err * pos_err;
        if (pos_err > max_pos_err) max_pos_err = pos_err;

        double dpsi_raw = state_real[i * 6 + 2] - fd.eta[i * 3 + 2];
        double psi_err = atan2(sin(dpsi_raw), cos(dpsi_raw));
        sum_psi_err2 += psi_err * psi_err;

        double du = state_real[i * 6 + 3] - fd.nu[i * 3 + 0];
        double dv = state_real[i * 6 + 4] - fd.nu[i * 3 + 1];
        double dr = state_real[i * 6 + 5] - fd.nu[i * 3 + 2];
        sum_u_err2 += du * du;
        sum_v_err2 += dv * dv;
        sum_r_err2 += dr * dr;
    }

    double rmse_pos = sqrt(sum_pos_err2 / n);
    double rmse_psi_rad = sqrt(sum_psi_err2 / n);
    double rmse_psi_deg = rmse_psi_rad * 180.0 / M_PI;
    double rmse_u = sqrt(sum_u_err2 / n);
    double rmse_v = sqrt(sum_v_err2 / n);
    double rmse_r = sqrt(sum_r_err2 / n);

    FILE *fjson = fopen("openloop_metrics.json", "w");
    if (fjson) {
        fprintf(fjson,
            "{\n"
            "    \"case\": \"Case 2 (C)\",\n"
            "    \"solver_type\": \"QP (6-Param Pseudo-Flatness)\",\n"
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
            solve_time_ms, rmse_pos, max_pos_err, rmse_psi_rad, rmse_psi_deg,
            rmse_u, rmse_v, rmse_r);
        fclose(fjson);
    }

    printf("\n==================================================\n");
    printf("CASE 2 OPEN-LOOP TRAJECTORY METRICS:\n");
    printf("  - Trajectory Solver Time (QP): %.4f ms\n", solve_time_ms);
    printf("  - Position RMSE:               %.4f m\n", rmse_pos);
    printf("  - Max Position Error:          %.4f m\n", max_pos_err);
    printf("  - Heading RMSE:                %.4f rad (%.2f deg)\n", rmse_psi_rad, rmse_psi_deg);
    printf("  - Surge Velocity (u) RMSE:     %.4f m/s\n", rmse_u);
    printf("  - Sway Velocity (v) RMSE:      %.4f m/s\n", rmse_v);
    printf("  - Yaw Rate (r) RMSE:           %.4f rad/s\n", rmse_r);
    printf("Metrics saved to: openloop_metrics.json\n");
    printf("==================================================\n\n");

    /* --- CSV output --- */
    FILE *fcsv = fopen("c2_openloop_results.csv", "w");
    if (fcsv) {
        fprintf(fcsv,
            "t,x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref,"
            "x_real,y_real,psi_real,u_real,v_real,r_real,"
            "tau_u_plan,tau_r_plan,T1_plan,T2_plan\n");
        for (int i = 0; i < n; ++i) {
            fprintf(fcsv,
                "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                "%.9f,%.9f,%.9f,%.9f\n",
                t_sim[i],
                fd.eta[i*3+0], fd.eta[i*3+1], fd.eta[i*3+2],
                fd.nu[i*3+0], fd.nu[i*3+1], fd.nu[i*3+2],
                state_real[i*6+0], state_real[i*6+1], state_real[i*6+2],
                state_real[i*6+3], state_real[i*6+4], state_real[i*6+5],
                fd.tau_plan[i*2+0], fd.tau_plan[i*2+1],
                fd.T_plan[i*2+0], fd.T_plan[i*2+1]);
        }
        fclose(fcsv);
    }

    FILE *fapplied = fopen("c2_openloop_applied.csv", "w");
    if (fapplied) {
        fprintf(fapplied, "t,tau_u_applied,tau_r_applied,T1_applied,T2_applied\n");
        for (int i = 0; i < n - 1; ++i) {
            fprintf(fapplied, "%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    t_sim[i], tau_applied[i*2+0], tau_applied[i*2+1],
                    T_applied[i*2+0], T_applied[i*2+1]);
        }
        fclose(fapplied);
    }

    FILE *fwp = fopen("c2_waypoints.csv", "w");
    if (fwp) {
        fprintf(fwp, "idx,x,y\n");
        for (int i = 0; i < n_wp; ++i) {
            fprintf(fwp, "%d,%.9f,%.9f\n", i, waypoints[i][0], waypoints[i][1]);
        }
        fclose(fwp);
    }

    printf("Open-loop simulation completed. CSV written: c2_openloop_results.csv, c2_openloop_applied.csv\n");

    flatness_data_free(&fd);
    free(t_sim); free(pos); free(vel); free(acc); free(jerk);
    free(state_real); free(tau_applied); free(T_applied);
    mj2d_free(&planner);

    return 0;
}
