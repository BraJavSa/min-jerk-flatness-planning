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
    double *jx_arr = malloc(sizeof(double) * n_steps);
    double *jy_arr = malloc(sizeof(double) * n_steps);

    for (int i = 0; i < n_steps; i++) {
        double t = t0 + i * DT_SIM;
        t_sim[i] = t;
        x[i] = minjerk1d_eval(&planner.tx, t, 0);
        y[i] = minjerk1d_eval(&planner.ty, t, 0);
        dx[i] = minjerk1d_eval(&planner.tx, t, 1);
        dy[i] = minjerk1d_eval(&planner.ty, t, 1);
        ddx[i] = minjerk1d_eval(&planner.tx, t, 2);
        ddy[i] = minjerk1d_eval(&planner.ty, t, 2);
        jx_arr[i] = minjerk1d_eval(&planner.tx, t, 3);
        jy_arr[i] = minjerk1d_eval(&planner.ty, t, 3);
    }

    FlatSample *flat = malloc(sizeof(FlatSample) * n_steps);
    double t_fl0 = now_ms();
    reconstruct_flatness_h2(x, y, dx, dy, ddx, ddy, n_steps, t_sim, flat);
    double flatness_time_ms = now_ms() - t_fl0;

    printf("==================================================\n");
    printf("CASE 1 (C) - QP Planning + Flatness Reconstruction\n");
    printf("--------------------------------------------------\n");
    printf("Samples:                        %d (dt = %.5f s, T = %.3f s)\n",
           n_steps, DT_SIM, t_sim[n_steps - 1] - t_sim[0]);
    printf("QP Planning Time:               %.5f ms\n", qp_time_ms);
    printf("Flatness Reconstruction Time:   %.5f ms\n", flatness_time_ms);
    printf("Total Time:                     %.5f ms\n", qp_time_ms + flatness_time_ms);
    printf("--------------------------------------------------\n");
    printf("Final tau_u: %.4f N   Final tau_r: %.4f N.m\n",
           flat[n_steps - 1].tau_plan[0], flat[n_steps - 1].tau_plan[1]);
    printf("==================================================\n");

    const char *json_path = "planning_metrics.json";
    FILE *fj = fopen(json_path, "w");
    if (fj) {
        fprintf(fj, "{\n");
        fprintf(fj, "    \"case\": \"Case 1 (C)\",\n");
        fprintf(fj, "    \"solver_type\": \"QP (5-Param Model)\",\n");
        fprintf(fj, "    \"qp_planning_time_ms\": %.5f,\n", qp_time_ms);
        fprintf(fj, "    \"flatness_reconstruction_time_ms\": %.5f,\n", flatness_time_ms);
        fprintf(fj, "    \"total_time_ms\": %.5f,\n", qp_time_ms + flatness_time_ms);
        fprintf(fj, "    \"tiempo_qp_ms\": %.5f,\n", qp_time_ms);
        fprintf(fj, "    \"tiempo_planitud_ms\": %.5f,\n", flatness_time_ms);
        fprintf(fj, "    \"tiempo_total_ms\": %.5f,\n", qp_time_ms + flatness_time_ms);
        fprintf(fj, "    \"solve_time_ms\": %.5f,\n", qp_time_ms + flatness_time_ms);
        fprintf(fj, "    \"total_sim_time_s\": %.5f,\n", t_sim[n_steps - 1] - t_sim[0]);
        fprintf(fj, "    \"num_samples\": %d\n", n_steps);
        fprintf(fj, "}\n");
        fclose(fj);
        printf("Metrics saved to: %s\n", json_path);
    }

    const char *csv_path = "case1_planning_results.csv";
    FILE *fp = fopen(csv_path, "w");
    if (fp) {
        fprintf(fp, "t,x,y,psi,u,v,r,jerk_x,jerk_y,tau_u,tau_r,T1,T2\n");
        for (int i = 0; i < n_steps; i++) {
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i],
                    flat[i].eta[0], flat[i].eta[1], flat[i].eta[2],
                    flat[i].nu[0], flat[i].nu[1], flat[i].nu[2],
                    jx_arr[i], jy_arr[i],
                    flat[i].tau_plan[0], flat[i].tau_plan[1],
                    flat[i].T_plan[0], flat[i].T_plan[1]);
        }
        fclose(fp);
        FILE *fw = fopen("case1_waypoints.csv", "w");
        if (fw) {
            for (int i = 0; i < n_wp; i++) {
                fprintf(fw, "%.6f,%.6f\n", wp_x[i], wp_y[i]);
            }
            fclose(fw);
        }
        printf("CSV saved to: %s\n", csv_path);
    } else {
        fprintf(stderr, "Could not write to %s\n", csv_path);
    }

    minjerk2d_free(&planner);
    free(t_sim); free(x); free(y); free(dx); free(dy); free(ddx); free(ddy);
    free(jx_arr); free(jy_arr);
    free(flat);
    return 0;
}
