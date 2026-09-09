/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
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

    printf("[Main] Trajectory Solver (QP) Compute Time: %.4f ms\n", solve_time_ms);

    /* Write metrics JSON */
    FILE *fm = fopen("planning_metrics.json", "w");
    if (fm) {
        fprintf(fm,
            "{\n"
            "    \"case\": \"Case 2 Semilla\",\n"
            "    \"solver_type\": \"QP (9-Param Pseudo-Flatness)\",\n"
            "    \"solve_time_ms\": %.6f,\n"
            "    \"total_sim_time_s\": %.6f,\n"
            "    \"num_samples\": %d\n"
            "}\n",
            solve_time_ms, t_sim[N - 1] - t_sim[0], N);
        fclose(fm);
    }

    /* Write CSV for plotting (waypoints saved separately) */
    FILE *fc = fopen("case2_planning_results.csv", "w");
    if (fc) {
        fprintf(fc, "t,x,y,psi,u,v,r,tau_u,tau_r,T1,T2,jerk_x,jerk_y\n");
        for (i = 0; i < N; ++i) {
            fprintf(fc, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    t_sim[i],
                    eta[3 * i + 0], eta[3 * i + 1], eta[3 * i + 2],
                    nu[3 * i + 0], nu[3 * i + 1], nu[3 * i + 2],
                    tau_plan[2 * i + 0], tau_plan[2 * i + 1],
                    T_plan[2 * i + 0], T_plan[2 * i + 1],
                    jerk[2 * i + 0], jerk[2 * i + 1]);
        }
        fclose(fc);
    }

    FILE *fw = fopen("case2_waypoints.csv", "w");
    if (fw) {
        fprintf(fw, "x,y\n");
        for (i = 0; i < n_wp; ++i) {
            fprintf(fw, "%.9f,%.9f\n", waypoints[i][0], waypoints[i][1]);
        }
        fclose(fw);
    }

    printf("[Main] Planning completed. Results saved to: case2_planning_results.csv\n");

    free(t_sim); free(pos); free(vel); free(acc); free(jerk);
    free(eta); free(nu); free(tau_plan); free(tau_act);
    free(tau_u_raw); free(tau_r_raw); free(cmds); free(T_plan); free(T_act);
    mjt2d_free(&planner);

    return 0;
}
