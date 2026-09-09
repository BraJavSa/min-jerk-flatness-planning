/* Technique: Case 3 - Non-Linear Programming (NLP/IPOPT) Trajectory Planning +
 *            6-Parameter Exact Flatness Reconstruction
 *
 * Planificación y reconstrucción en C (equivalente a main.py).
 * Genera case3_planning_results.csv e imprime tiempos y métricas.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "usv_params.h"
#include "trajectory_nlp.h"
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

    double epsilon_tv = 0.1;

    /* --- Trajectory NLP Solver (timed) --- */
    printf("[NLP] Resolviendo optimizacion NLP con Ipopt (B-spline flat outputs)...\n");
    double t_nlp0 = now_ms();
    TrajectoryNLP planner;
    int ret = trajectory_nlp_solve(&planner, wp_x, wp_y, times, v0x, v0y, vfx, vfy, epsilon_tv);
    double nlp_time_ms = now_ms() - t_nlp0;

    if (ret != 0) {
        printf("[Aviso] El solver Ipopt finalizo con codigo %d\n", ret);
    }

    /* Muestreo de trayectoria */
    double *t_sim = NULL;
    double *pos = NULL;
    double *vel = NULL;
    double *acc = NULL;
    double *jerk = NULL;
    int n_steps = trajectory_nlp_sample(&planner, DT_SIM, &t_sim, &pos, &vel, &acc, &jerk);

    /* --- Reconstrucción por planitud exacta 6-param (timed) --- */
    FlatSample *flat = (FlatSample *)malloc(sizeof(FlatSample) * n_steps);
    double t_fl0 = now_ms();
    reconstruct_flatness_case3(pos, vel, acc, n_steps, flat);
    double flatness_time_ms = now_ms() - t_fl0;

    printf("==================================================\n");
    printf("CASE 3 (C) - Planificacion NLP (IPOPT) + Reconstruccion Exacta 6-Param\n");
    printf("--------------------------------------------------\n");
    printf("Muestras:                       %d (dt = %.5f s, T = %.3f s)\n",
           n_steps, DT_SIM, t_sim[n_steps - 1] - t_sim[0]);
    printf("Tiempo NLP (planificacion):     %.5f ms\n", nlp_time_ms);
    printf("Tiempo reconstruccion planitud: %.5f ms\n", flatness_time_ms);
    printf("Tiempo total:                   %.5f ms\n", nlp_time_ms + flatness_time_ms);
    printf("--------------------------------------------------\n");
    printf("tau_u final:  %.4f N   tau_r final: %.4f N.m\n",
           flat[n_steps - 1].tau_plan[0], flat[n_steps - 1].tau_plan[1]);
    printf("==================================================\n");

    /* Guardar planning_metrics.json */
    FILE *f_json = fopen("planning_metrics.json", "w");
    if (f_json) {
        fprintf(f_json, "{\n");
        fprintf(f_json, "    \"case\": \"Case 3 (C)\",\n");
        fprintf(f_json, "    \"solver_type\": \"Ipopt C API (NLP Flatness)\",\n");
        fprintf(f_json, "    \"solve_time_ms\": %.5f,\n", nlp_time_ms);
        fprintf(f_json, "    \"total_sim_time_s\": %.5f,\n", t_sim[n_steps - 1] - t_sim[0]);
        fprintf(f_json, "    \"num_samples\": %d\n", n_steps);
        fprintf(f_json, "}\n");
        fclose(f_json);
    }

    /* Guardar CSV para graficar con plot_results.py */
    const char *csv_path = "case3_planning_results.csv";
    FILE *fp = fopen(csv_path, "w");
    if (fp) {
        fprintf(fp, "t,x,y,psi,u,v,r,jerk_x,jerk_y,tau_u,tau_r,tau_v,T1,T2\n");
        for (int i = 0; i < n_steps; i++) {
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i],
                    flat[i].eta[0], flat[i].eta[1], flat[i].eta[2],
                    flat[i].nu[0], flat[i].nu[1], flat[i].nu[2],
                    jerk[i * 3 + 0], jerk[i * 3 + 1],
                    flat[i].tau_plan[0], flat[i].tau_plan[1], flat[i].tau_v_raw,
                    flat[i].T_plan[0], flat[i].T_plan[1]);
        }
        fclose(fp);
        printf("[Main] Resultados de planificacion guardados en: %s\n", csv_path);
    } else {
        fprintf(stderr, "Error al abrir %s para escritura.\n", csv_path);
    }

    free(t_sim);
    free(pos);
    free(vel);
    free(acc);
    free(jerk);
    free(flat);

    return 0;
}
