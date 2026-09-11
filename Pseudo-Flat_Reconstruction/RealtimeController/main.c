#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "usv_params.h"
#include "min_jerk_qp.h"
#include "flatness_reconstruct.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[])
{

    double x0 = 0.0, y0 = 0.0, psi0 = 0.0;
    double u0 = 0.0, v0 = 0.0, r0 = 0.0;
    const char *output_csv_path = "planned_trajectory_reference.csv";
    const char *metrics_json_path = "planning_metrics.json";

    if (argc >= 4) {
        x0 = atof(argv[1]);
        y0 = atof(argv[2]);
        psi0 = atof(argv[3]);
    }
    if (argc >= 7) {
        u0 = atof(argv[4]);
        v0 = atof(argv[5]);
        r0 = atof(argv[6]);
    }
    if (argc >= 8) {
        output_csv_path = argv[7];
    }
    if (argc >= 9) {
        metrics_json_path = argv[8];
    }

    printf("[Case 2 C Planner] Init state: (x=%.3f, y=%.3f, psi=%.3f rad, u=%.3f, v=%.3f, r=%.3f)\n",
           x0, y0, psi0, u0, v0, r0);

    const double base_waypoints[9][2] = {
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
    for (int i = 0; i < n_wp; ++i) {
        times[i] = base_times[i] * time_scale;
    }

    double c_psi = cos(psi0), s_psi = sin(psi0);
    double waypoints_flat[18];
    double wp_x[9], wp_y[9];
    for (int i = 0; i < n_wp; ++i) {
        double dx_rel = base_waypoints[i][0] - base_waypoints[0][0];
        double dy_rel = base_waypoints[i][1] - base_waypoints[0][1];
        wp_x[i] = x0 + c_psi * dx_rel - s_psi * dy_rel;
        wp_y[i] = y0 + s_psi * dx_rel + c_psi * dy_rel;
        waypoints_flat[2 * i + 0] = wp_x[i];
        waypoints_flat[2 * i + 1] = wp_y[i];
    }

    double v0_vec[2] = {
        u0 * c_psi - v0 * s_psi,
        u0 * s_psi + v0 * c_psi
    };
    if (hypot(v0_vec[0], v0_vec[1]) < 0.05) {
        v0_vec[0] = 0.1 * c_psi;
        v0_vec[1] = 0.1 * s_psi;
    }

    double dir_f[2] = { wp_x[8] - wp_x[7], wp_y[8] - wp_y[7] };
    double dir_f_norm = sqrt(dir_f[0] * dir_f[0] + dir_f[1] * dir_f[1]);
    double dir_f_unit[2] = { dir_f[0] / dir_f_norm, dir_f[1] / dir_f_norm };
    double vf_vec[2] = { 0.01 * dir_f_unit[0], 0.01 * dir_f_unit[1] };

    double t_qp0 = now_ms();
    MinJerkTrajectory2D planner;
    mjt2d_init(&planner, waypoints_flat, times, n_wp,
               v0_vec[0], v0_vec[1], vf_vec[0], vf_vec[1]);
    double qp_time_ms = now_ms() - t_qp0;

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

    double t_fl0 = now_ms();
    reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim, N, NULL,
                             eta, nu, tau_plan, tau_act,
                             tau_u_raw, tau_r_raw, cmds, T_plan, T_act);
    double flatness_time_ms = now_ms() - t_fl0;
    double total_time_ms = qp_time_ms + flatness_time_ms;

    printf("[Case 2 C Planner] Steps: %d (T = %.2f s) | QP: %.4f ms | Flatness: %.4f ms | Total: %.4f ms\n",
           N, t_sim[N - 1] - t_sim[0], qp_time_ms, flatness_time_ms, total_time_ms);

    FILE *fc = fopen(output_csv_path, "w");
    if (fc) {
        fprintf(fc, "t,x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref,tau_u_ref,tau_r_ref,T1_ref,T2_ref,cmd_left_ref,cmd_right_ref\n");
        for (int i = 0; i < N; ++i) {
            fprintf(fc, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i],
                    eta[3 * i + 0], eta[3 * i + 1], eta[3 * i + 2],
                    nu[3 * i + 0], nu[3 * i + 1], nu[3 * i + 2],
                    tau_plan[2 * i + 0], tau_plan[2 * i + 1],
                    T_plan[2 * i + 0], T_plan[2 * i + 1],
                    cmds[2 * i + 0], cmds[2 * i + 1]);
        }
        fclose(fc);
        printf("[Case 2 C Planner] Reference trajectory exported to: %s\n", output_csv_path);
    } else {
        fprintf(stderr, "[Case 2 C Planner] Error opening CSV file: %s\n", output_csv_path);
    }

    FILE *fm = fopen(metrics_json_path, "w");
    if (fm) {
        fprintf(fm,
            "{\n"
            "    \"case\": \"Case 2 Pseudo-Flat Realtime (C)\",\n"
            "    \"solver_type\": \"QP (6-Param Pseudo-Flatness)\",\n"
            "    \"QP Planning Time\": \"%.5f ms\",\n"
            "    \"Flatness Reconstruction Time\": \"%.5f ms\",\n"
            "    \"Total Time\": \"%.5f ms\",\n"
            "    \"tiempo_qp_ms\": %.5f,\n"
            "    \"tiempo_planitud_ms\": %.5f,\n"
            "    \"tiempo_total_ms\": %.5f,\n"
            "    \"total_sim_time_s\": %.5f,\n"
            "    \"num_samples\": %d\n"
            "}\n",
            qp_time_ms, flatness_time_ms, total_time_ms,
            qp_time_ms, flatness_time_ms, total_time_ms,
            t_sim[N - 1] - t_sim[0], N);
        fclose(fm);
    }

    free(t_sim); free(pos); free(vel); free(acc); free(jerk);
    free(eta); free(nu); free(tau_plan); free(tau_act);
    free(tau_u_raw); free(tau_r_raw); free(cmds); free(T_plan); free(T_act);
    mjt2d_free(&planner);

    return 0;
}
