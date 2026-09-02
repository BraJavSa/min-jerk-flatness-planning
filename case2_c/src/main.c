/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Direct port of c2_main.py. Instead of matplotlib figures, writes a
 * CSV with all sampled trajectory/control channels plus a JSON of
 * planning metrics; a companion Python script (plot_results.py)
 * reproduces the original figures from the CSV.
 */
#include "min_jerk_qp.h"
#include "flatness_reconstruct.h"
#include "usv_params.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void) {
    double waypoints[9][2] = {
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

    MinJerkTrajectory2D planner;
    double wp_flat[18];
    for (int i = 0; i < n_wp; ++i) {
        wp_flat[i * 2 + 0] = waypoints[i][0];
        wp_flat[i * 2 + 1] = waypoints[i][1];
    }
    int rc = mj2d_init(&planner, wp_flat, times, n_wp, v0_vec, vf_vec);
    if (rc != 0) {
        fprintf(stderr, "[Main] Trajectory solve failed (LAPACK info=%d)\n", rc);
        return 1;
    }

    double *t_sim, *pos, *vel, *acc, *jerk;
    int n = mj2d_sample(&planner, DT_SIM, &t_sim, &pos, &vel, &acc, &jerk);

    FlatnessData fd;
    reconstruct_flatness_h2(pos, vel, acc, jerk, t_sim, n, 0, 0.0, &fd);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double solve_time_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 +
                            (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e6;

    printf("[Main] Trajectory Solver (QP) Compute Time: %.4f ms\n", solve_time_ms);

    /* --- Write metrics JSON --- */
    FILE *fjson = fopen("planning_metrics.json", "w");
    if (fjson) {
        fprintf(fjson,
            "{\n"
            "    \"case\": \"Case 2\",\n"
            "    \"solver_type\": \"QP (6-Param Pseudo-Flatness)\",\n"
            "    \"solve_time_ms\": %.6f,\n"
            "    \"total_sim_time_s\": %.6f,\n"
            "    \"num_samples\": %d\n"
            "}\n",
            solve_time_ms, t_sim[n - 1] - t_sim[0], n);
        fclose(fjson);
    }

    /* --- Write full CSV for plotting --- */
    FILE *fcsv = fopen("c2_main_results.csv", "w");
    if (fcsv) {
        fprintf(fcsv, "t,x,y,psi,u,v,r,tau_u,tau_r,T1_plan,T2_plan,jx,jy\n");
        for (int i = 0; i < n; ++i) {
            fprintf(fcsv, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    t_sim[i],
                    fd.eta[i * 3 + 0], fd.eta[i * 3 + 1], fd.eta[i * 3 + 2],
                    fd.nu[i * 3 + 0], fd.nu[i * 3 + 1], fd.nu[i * 3 + 2],
                    fd.tau_plan[i * 2 + 0], fd.tau_plan[i * 2 + 1],
                    fd.T_plan[i * 2 + 0], fd.T_plan[i * 2 + 1],
                    jerk[i * 2 + 0], jerk[i * 2 + 1]);
        }
        fclose(fcsv);
    }

    /* --- Write waypoints CSV (for plotting) --- */
    FILE *fwp = fopen("c2_waypoints.csv", "w");
    if (fwp) {
        fprintf(fwp, "idx,x,y\n");
        for (int i = 0; i < n_wp; ++i) {
            fprintf(fwp, "%d,%.9f,%.9f\n", i, waypoints[i][0], waypoints[i][1]);
        }
        fclose(fwp);
    }

    printf("[Main] Planning completed. CSV written: c2_main_results.csv, c2_waypoints.csv\n");

    flatness_data_free(&fd);
    free(t_sim); free(pos); free(vel); free(acc); free(jerk);
    mj2d_free(&planner);

    return 0;
}
