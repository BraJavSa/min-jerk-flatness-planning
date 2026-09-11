#include "flatness_reconstruct.h"
#include "min_jerk_qp.h"
#include "usv_params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[]) {

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

  printf("[Case 1 C Planner] Init state: (x=%.3f, y=%.3f, psi=%.3f rad, "
         "u=%.3f, v=%.3f, r=%.3f)\n",
         x0, y0, psi0, u0, v0, r0);

  double base_wp_x[] = {0.0, 8.0, 14.0, 14.0, 20.0, 28.0, 32.0, 26.0, 20.0};
  double base_wp_y[] = {0.0, 0.0, 5.0, 13.0, 17.0, 17.0, 10.0, 4.0, 1.5};
  int n_wp = 9;

  double base_times[] = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
  double time_scale = 2.10;
  double times[9];
  for (int i = 0; i < n_wp; i++) {
    times[i] = base_times[i] * time_scale;
  }

  double wp_x[9], wp_y[9];
  double c_psi = cos(psi0), s_psi = sin(psi0);
  for (int i = 0; i < n_wp; i++) {
    double dx_rel = base_wp_x[i] - base_wp_x[0];
    double dy_rel = base_wp_y[i] - base_wp_y[0];
    wp_x[i] = x0 + c_psi * dx_rel - s_psi * dy_rel;
    wp_y[i] = y0 + s_psi * dx_rel + c_psi * dy_rel;
  }

  double v0x = u0 * c_psi - v0 * s_psi;
  double v0y = u0 * s_psi + v0 * c_psi;
  if (hypot(v0x, v0y) < 0.05) {
    v0x = 0.1 * c_psi;
    v0y = 0.1 * s_psi;
  }

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
  double total_time_ms = qp_time_ms + flatness_time_ms;

  printf("[Case 1 C Planner] Steps: %d (T = %.2f s) | QP: %.4f ms | Flatness: "
         "%.4f ms | Total: %.4f ms\n",
         n_steps, t_sim[n_steps - 1] - t_sim[0], qp_time_ms, flatness_time_ms,
         total_time_ms);

  FILE *fp = fopen(output_csv_path, "w");
  if (fp) {
    fprintf(fp, "t,x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref,tau_u_ref,tau_r_ref,"
                "T1_ref,T2_ref,cmd_left_ref,cmd_right_ref\n");
    for (int i = 0; i < n_steps; i++) {
      double T1 = flat[i].T_plan[0];
      double T2 = flat[i].T_plan[1];
      double cmd_l = cmd_from_thrust(T1);
      double cmd_r = cmd_from_thrust(T2);

      fprintf(
          fp,
          "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
          t_sim[i], flat[i].eta[0], flat[i].eta[1], flat[i].eta[2],
          flat[i].nu[0], flat[i].nu[1], flat[i].nu[2], flat[i].tau_plan[0],
          flat[i].tau_plan[1], T1, T2, cmd_l, cmd_r);
    }
    fclose(fp);
    char wp_csv_path[512];
    char *last_slash = strrchr(output_csv_path, '/');
    if (last_slash) {
      int dlen = (int)(last_slash - output_csv_path);
      snprintf(wp_csv_path, sizeof(wp_csv_path), "%.*s/waypoints.csv", dlen, output_csv_path);
    } else {
      snprintf(wp_csv_path, sizeof(wp_csv_path), "waypoints.csv");
    }
    FILE *fwp = fopen(wp_csv_path, "w");
    if (fwp) {
      for (int i = 0; i < n_wp; i++) {
        fprintf(fwp, "%.6f,%.6f\n", wp_x[i], wp_y[i]);
      }
      fclose(fwp);
    }
    printf("[Case 1 C Planner] Reference trajectory exported to: %s\n",
           output_csv_path);
  } else {
    fprintf(stderr, "[Case 1 C Planner] Error opening CSV file: %s\n",
            output_csv_path);
  }

  FILE *fj = fopen(metrics_json_path, "w");
  if (fj) {
    fprintf(fj, "{\n");
    fprintf(fj, "    \"case\": \"Case 1 Realtime (C)\",\n");
    fprintf(fj, "    \"solver_type\": \"QP (5-Param Model)\",\n");
    fprintf(fj, "    \"QP Planning Time\": \"%.5f ms\",\n",
            qp_time_ms);
    fprintf(fj, "    \"Flatness Reconstruction Time\": \"%.5f ms\",\n",
            flatness_time_ms);
    fprintf(fj, "    \"Total Time\": \"%.5f ms\",\n", total_time_ms);
    fprintf(fj, "    \"tiempo_qp_ms\": %.5f,\n", qp_time_ms);
    fprintf(fj, "    \"tiempo_planitud_ms\": %.5f,\n", flatness_time_ms);
    fprintf(fj, "    \"tiempo_total_ms\": %.5f,\n", total_time_ms);
    fprintf(fj, "    \"num_samples\": %d\n", n_steps);
    fprintf(fj, "}\n");
    fclose(fj);
  }

  minjerk2d_free(&planner);
  free(t_sim);
  free(x);
  free(y);
  free(dx);
  free(dy);
  free(ddx);
  free(ddy);
  free(jx_arr);
  free(jy_arr);
  free(flat);
  return 0;
}
