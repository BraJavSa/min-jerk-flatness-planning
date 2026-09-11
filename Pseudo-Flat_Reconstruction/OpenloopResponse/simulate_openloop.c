#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "usv_params.h"
#include "min_jerk_qp.h"
#include "flatness_reconstruct.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void deriv_9param(const double *s, double Tu, double Tr, double *ds)
{
    double x_, y_, psi_i, u_i, v_i, r_i;
    (void)x_; (void)y_;
    psi_i = s[2];
    u_i = s[3];
    v_i = s[4];
    r_i = s[5];

    double dx = u_i * cos(psi_i) - v_i * sin(psi_i);
    double dy = u_i * sin(psi_i) + v_i * cos(psi_i);
    double dpsi = r_i;

    double du = (Tu + m22_9 * v_i * r_i - Xu_9 * u_i - Xuu_9 * u_i * fabs(u_i)) / m11_9;
    double dv = (-m11_9 * u_i * r_i - Yv_9 * v_i - Yvv_9 * v_i * fabs(v_i)) / m22_9;
    double dr = (Tr + (m11_9 - m22_9) * u_i * v_i - Nr_9 * r_i - Nrr_9 * r_i * fabs(r_i)) / m33_9;

    ds[0] = dx; ds[1] = dy; ds[2] = dpsi;
    ds[3] = du; ds[4] = dv; ds[5] = dr;
}

static void real_9param_rk4_step(const double *state, double Tu, double Tr, double dt, double *state_next)
{
    int i;
    double k1[6], k2[6], k3[6], k4[6], tmp[6];

    deriv_9param(state, Tu, Tr, k1);

    for (i = 0; i < 6; ++i) tmp[i] = state[i] + 0.5 * dt * k1[i];
    deriv_9param(tmp, Tu, Tr, k2);

    for (i = 0; i < 6; ++i) tmp[i] = state[i] + 0.5 * dt * k2[i];
    deriv_9param(tmp, Tu, Tr, k3);

    for (i = 0; i < 6; ++i) tmp[i] = state[i] + dt * k3[i];
    deriv_9param(tmp, Tu, Tr, k4);

    for (i = 0; i < 6; ++i) {
        state_next[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

int main(void)
{
    const double waypoints[9][2] = {
        {0.0, 0.0},
        {8.0, 0.0},
        {14.0, 5.0},
        {14.0, 13.0},
        {20.0, 17.0},
        {28.0, 17.0},
        {32.0, 10.0},
        {26.0, 4.0},
        {20.0, 1.5}
    };
    const int n_wp = 9;

    double base_times[9] = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
    double time_scale = 2.10;
    double times[9];
    int i;
    for (i = 0; i < n_wp; ++i) {
        times[i] = base_times[i] * time_scale;
    }

    double v0_vec[2] = {0.1, 0.0};
    double dir_f[2] = { waypoints[8][0] - waypoints[7][0], waypoints[8][1] - waypoints[7][1] };
    double dir_f_norm = sqrt(dir_f[0] * dir_f[0] + dir_f[1] * dir_f[1]);
    double dir_f_unit[2] = { dir_f[0] / dir_f_norm, dir_f[1] / dir_f_norm };
    double vf_vec[2] = { 0.01 * dir_f_unit[0], 0.01 * dir_f_unit[1] };

    double waypoints_flat[18];
    for (i = 0; i < n_wp; ++i) {
        waypoints_flat[2 * i + 0] = waypoints[i][0];
        waypoints_flat[2 * i + 1] = waypoints[i][1];
    }

    double t_start = now_ms();

    MinJerkTrajectory2D planner;
    mjt2d_init(&planner, waypoints_flat, times, n_wp,
               v0_vec[0], v0_vec[1], vf_vec[0], vf_vec[1]);

    double *t_sim, *pos, *vel, *acc, *jerk;
    int N = mjt2d_sample(&planner, DT_SIM, &t_sim, &pos, &vel, &acc, &jerk);

    double *eta = (double *)malloc(sizeof(double) * 3 * N);
    double *nu = (double *)malloc(sizeof(double) * 3 * N);
    double *tau_plan = (double *)malloc(sizeof(double) * 2 * N);
    double *tau_act = (double *)malloc(sizeof(double) * 2 * N);
    double *tau_u_raw = (double *)malloc(sizeof(double) * N);
    double *tau_r_raw = (double *)malloc(sizeof(double) * N);
    double *cmds = (double *)malloc(sizeof(double) * 2 * N);
    double *T_plan = (double *)malloc(sizeof(double) * 2 * N);
    double *T_act = (double *)malloc(sizeof(double) * 2 * N);

    reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim, N, NULL,
                             eta, nu, tau_plan, tau_act,
                             tau_u_raw, tau_r_raw, cmds, T_plan, T_act);

    double solve_time_ms = now_ms() - t_start;
    int n_steps = N;

    double state_real[6];
    state_real[0] = eta[0]; state_real[1] = eta[1]; state_real[2] = eta[2];
    state_real[3] = nu[0]; state_real[4] = nu[1]; state_real[5] = nu[2];

    double *hist_state_real = (double *)malloc(sizeof(double) * 6 * n_steps);
    double *hist_tau_applied = (double *)malloc(sizeof(double) * 2 * (n_steps - 1));
    double *hist_T_applied = (double *)malloc(sizeof(double) * 2 * (n_steps - 1));

    for (i = 0; i < 6; ++i) hist_state_real[i] = state_real[i];

    for (i = 0; i < n_steps - 1; ++i) {
        double Tu_apply = tau_act[2 * i + 0];
        double Tr_apply = tau_act[2 * i + 1];
        hist_tau_applied[2 * i + 0] = Tu_apply;
        hist_tau_applied[2 * i + 1] = Tr_apply;
        hist_T_applied[2 * i + 0] = T_act[2 * i + 0];
        hist_T_applied[2 * i + 1] = T_act[2 * i + 1];

        double state_next[6];
        real_9param_rk4_step(state_real, Tu_apply, Tr_apply, DT_SIM, state_next);
        for (int j = 0; j < 6; ++j) state_real[j] = state_next[j];

        for (int j = 0; j < 6; ++j) hist_state_real[(i + 1) * 6 + j] = state_real[j];
    }

    double sum_pos_err2 = 0.0, max_err_pos = 0.0;
    double sum_psi_err2 = 0.0;
    double sum_u_err2 = 0.0, sum_v_err2 = 0.0, sum_r_err2 = 0.0;

    for (i = 0; i < n_steps; ++i) {
        double dx = hist_state_real[6 * i + 0] - eta[3 * i + 0];
        double dy = hist_state_real[6 * i + 1] - eta[3 * i + 1];
        double pos_err = sqrt(dx * dx + dy * dy);
        sum_pos_err2 += pos_err * pos_err;
        if (pos_err > max_err_pos) max_err_pos = pos_err;

        double dpsi = hist_state_real[6 * i + 2] - eta[3 * i + 2];
        double psi_err = atan2(sin(dpsi), cos(dpsi));
        sum_psi_err2 += psi_err * psi_err;

        double du = hist_state_real[6 * i + 3] - nu[3 * i + 0];
        double dv = hist_state_real[6 * i + 4] - nu[3 * i + 1];
        double dr = hist_state_real[6 * i + 5] - nu[3 * i + 2];
        sum_u_err2 += du * du;
        sum_v_err2 += dv * dv;
        sum_r_err2 += dr * dr;
    }

    double rmse_pos = sqrt(sum_pos_err2 / n_steps);
    double rmse_psi_rad = sqrt(sum_psi_err2 / n_steps);
    double rmse_psi_deg = rmse_psi_rad * 180.0 / M_PI;
    double rmse_u = sqrt(sum_u_err2 / n_steps);
    double rmse_v = sqrt(sum_v_err2 / n_steps);
    double rmse_r = sqrt(sum_r_err2 / n_steps);

    FILE *fm = fopen("openloop_metrics.json", "w");
    if (fm) {
        fprintf(fm,
            "{\n"
            "    \"case\": \"Case 2 Pseudo-Flat\",\n"
            "    \"solver_type\": \"QP (9-Param Pseudo-Flatness with 9-Param Plant Simulation)\",\n"
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
            solve_time_ms, rmse_pos, max_err_pos, rmse_psi_rad, rmse_psi_deg,
            rmse_u, rmse_v, rmse_r);
        fclose(fm);
    }

    printf("\n==================================================\n");
    printf("CASE 2 OPEN-LOOP TRAJECTORY METRICS (9-PARAM):\n");
    printf("  - Trajectory Solver Time (QP): %.4f ms\n", solve_time_ms);
    printf("  - Position RMSE:               %.4f m\n", rmse_pos);
    printf("  - Max Position Error:          %.4f m\n", max_err_pos);
    printf("  - Heading RMSE:                %.4f rad (%.2f deg)\n", rmse_psi_rad, rmse_psi_deg);
    printf("  - Surge Velocity (u) RMSE:     %.4f m/s\n", rmse_u);
    printf("  - Sway Velocity (v) RMSE:      %.4f m/s\n", rmse_v);
    printf("  - Yaw Rate (r) RMSE:           %.4f rad/s\n", rmse_r);
    printf("Metrics saved to: openloop_metrics.json\n");
    printf("==================================================\n\n");

    FILE *fc = fopen("case2_openloop_results.csv", "w");
    if (fc) {
        fprintf(fc, "t,x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref,tau_u_ref,tau_r_ref,T1_ref,T2_ref,"
                    "x_real,y_real,psi_real,u_real,v_real,r_real\n");
        for (i = 0; i < n_steps; ++i) {
            fprintf(fc, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                        "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    t_sim[i],
                    eta[3 * i + 0], eta[3 * i + 1], eta[3 * i + 2],
                    nu[3 * i + 0], nu[3 * i + 1], nu[3 * i + 2],
                    tau_plan[2 * i + 0], tau_plan[2 * i + 1],
                    T_plan[2 * i + 0], T_plan[2 * i + 1],
                    hist_state_real[6 * i + 0], hist_state_real[6 * i + 1], hist_state_real[6 * i + 2],
                    hist_state_real[6 * i + 3], hist_state_real[6 * i + 4], hist_state_real[6 * i + 5]);
        }
        fclose(fc);
    }

    FILE *fa = fopen("case2_openloop_applied.csv", "w");
    if (fa) {
        fprintf(fa, "t,tau_u_applied,tau_r_applied,T1_applied,T2_applied\n");
        for (i = 0; i < n_steps - 1; ++i) {
            fprintf(fa, "%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    t_sim[i],
                    hist_tau_applied[2 * i + 0], hist_tau_applied[2 * i + 1],
                    hist_T_applied[2 * i + 0], hist_T_applied[2 * i + 1]);
        }
        fclose(fa);
    }

    FILE *fw = fopen("case2_waypoints.csv", "w");
    if (fw) {
        fprintf(fw, "x,y\n");
        for (i = 0; i < n_wp; ++i) {
            fprintf(fw, "%.9f,%.9f\n", waypoints[i][0], waypoints[i][1]);
        }
        fclose(fw);
    }

    printf("Open-loop simulation completed. Results saved to: case2_openloop_results.csv\n");

    free(t_sim); free(pos); free(vel); free(acc); free(jerk);
    free(eta); free(nu); free(tau_plan); free(tau_act);
    free(tau_u_raw); free(tau_r_raw); free(cmds); free(T_plan); free(T_act);
    free(hist_state_real); free(hist_tau_applied); free(hist_T_applied);
    mjt2d_free(&planner);

    return 0;
}
