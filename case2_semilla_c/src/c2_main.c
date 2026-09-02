/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 * C port of c2_main.py (planning only, no closed/open-loop simulation). */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "c2_usv_params.h"
#include "c2_min_jerk_qp.h"
#include "c2_flatness_reconstruct.h"

#define N_WP 9

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
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

    printf("[Main] Trajectory Solver (QP) Compute Time: %.4f ms\n", solve_time_ms);

    /* --- c2_waypoints.csv --- */
    FILE *fwp = fopen("c2_waypoints.csv", "w");
    if (fwp) {
        fprintf(fwp, "idx,x,y,t\n");
        for (int i = 0; i < N_WP; i++)
            fprintf(fwp, "%d,%.6f,%.6f,%.6f\n", i, waypoints_x[i], waypoints_y[i], times[i]);
        fclose(fwp);
    }

    /* --- c2_main_results.csv --- */
    FILE *fr = fopen("c2_main_results.csv", "w");
    if (fr) {
        fprintf(fr, "t,x,y,psi,u,v,r,tau_u,tau_r,T1,T2,cmd1,cmd2,jerk_x,jerk_y\n");
        for (int i = 0; i < N; i++)
            fprintf(fr, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i], flat.x[i], flat.y[i], flat.psi[i],
                    flat.u[i], flat.v[i], flat.r[i],
                    flat.tau_u[i], flat.tau_r[i],
                    flat.T1_dem[i], flat.T2_dem[i],
                    flat.cmd1[i], flat.cmd2[i],
                    jx[i], jy[i]);
        fclose(fr);
    }

    /* --- planning_metrics.json --- */
    FILE *fm = fopen("planning_metrics.json", "w");
    if (fm) {
        fprintf(fm,
            "{\n"
            "    \"case\": \"Case 2\",\n"
            "    \"solver_type\": \"QP (6-Param Pseudo-Flatness) - C port\",\n"
            "    \"solve_time_ms\": %.6f,\n"
            "    \"total_sim_time_s\": %.6f,\n"
            "    \"num_samples\": %d\n"
            "}\n",
            solve_time_ms, t_sim[N - 1] - t_sim[0], N);
        fclose(fm);
    }

    printf("[Main] Planning completed.\n");
    printf("  c2_waypoints.csv, c2_main_results.csv, planning_metrics.json written.\n");
    printf("  Run 'python3 plot_results.py' to regenerate the diagnostic PNGs.\n");

    flatness_result_free(&flat);
    free(t_sim); free(px); free(py); free(vx); free(vy); free(ax); free(ay); free(jx); free(jy);
    minjerk2d_free(&planner);
    return 0;
}
