/* Technique: Case 3 - Validacion en lazo abierto de la planificacion NLP
 * (6-param flatness) contra la planta completa no lineal (9-param).
 * Imprime el tiempo de simulacion (RK4) y las metricas de error.
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

static void full_dynamics_deriv(const double s[6], double Tu, double Tr, double out[6]) {
    double psi_i = s[2], u_i = s[3], v_i = s[4], r_i = s[5];
    out[0] = u_i * cos(psi_i) - v_i * sin(psi_i);
    out[1] = u_i * sin(psi_i) + v_i * cos(psi_i);
    out[2] = r_i;

    /* 9-parameter nonlinear hydrodynamic damping terms */
    out[3] = (Tu + m22_full * v_i * r_i - Xu_full * u_i - Xuu_full * fabs(u_i) * u_i) / m11_full;
    out[4] = (-m11_full * u_i * r_i - Yv_full * v_i - Yvv_full * fabs(v_i) * v_i) / m22_full;
    out[5] = (Tr + (m11_full - m22_full) * u_i * v_i - Nr_full * r_i - Nrr_full * fabs(r_i) * r_i) / m33_full;
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
    double epsilon_tv = 0.1;

    printf("[NLP] Resolviendo trayectoria para simulacion en lazo abierto...\n");
    double t_nlp0 = now_ms();
    TrajectoryNLP planner;
    trajectory_nlp_solve(&planner, wp_x, wp_y, times, v0x, v0y, vfx, vfy, epsilon_tv);
    double nlp_time_ms = now_ms() - t_nlp0;

    double *t_sim = NULL;
    double *pos = NULL;
    double *vel = NULL;
    double *acc = NULL;
    double *jerk = NULL;
    int n_steps = trajectory_nlp_sample(&planner, DT_SIM, &t_sim, &pos, &vel, &acc, &jerk);

    FlatSample *flat = (FlatSample *)malloc(sizeof(FlatSample) * n_steps);
    reconstruct_flatness_case3(pos, vel, acc, n_steps, flat);

    /* Historiales para calculo de errores y guardado CSV */
    double *real_x   = (double *)malloc(sizeof(double) * n_steps);
    double *real_y   = (double *)malloc(sizeof(double) * n_steps);
    double *real_psi = (double *)malloc(sizeof(double) * n_steps);
    double *real_u   = (double *)malloc(sizeof(double) * n_steps);
    double *real_v   = (double *)malloc(sizeof(double) * n_steps);
    double *real_r   = (double *)malloc(sizeof(double) * n_steps);

    double *applied_tau_u = (double *)malloc(sizeof(double) * n_steps);
    double *applied_tau_r = (double *)malloc(sizeof(double) * n_steps);
    double *applied_T1    = (double *)malloc(sizeof(double) * n_steps);
    double *applied_T2    = (double *)malloc(sizeof(double) * n_steps);

    /* Estado inicial real de la embarcacion */
    double state_real[6] = {
        flat[0].eta[0], flat[0].eta[1], flat[0].eta[2],
        flat[0].nu[0],  flat[0].nu[1],  flat[0].nu[2]
    };

    real_x[0] = state_real[0]; real_y[0] = state_real[1]; real_psi[0] = state_real[2];
    real_u[0] = state_real[3]; real_v[0] = state_real[4]; real_r[0]   = state_real[5];

    double t_sim0 = now_ms();
    for (int i = 0; i < n_steps - 1; i++) {
        double Tu_apply = flat[i].tau_act[0];
        double Tr_apply = flat[i].tau_act[1];

        applied_tau_u[i] = Tu_apply;
        applied_tau_r[i] = Tr_apply;
        applied_T1[i]    = flat[i].T_act[0];
        applied_T2[i]    = flat[i].T_act[1];

        rk4_step(state_real, Tu_apply, Tr_apply, DT_SIM);

        real_x[i + 1]   = state_real[0];
        real_y[i + 1]   = state_real[1];
        real_psi[i + 1] = state_real[2];
        real_u[i + 1]   = state_real[3];
        real_v[i + 1]   = state_real[4];
        real_r[i + 1]   = state_real[5];
    }
    applied_tau_u[n_steps - 1] = applied_tau_u[n_steps - 2];
    applied_tau_r[n_steps - 1] = applied_tau_r[n_steps - 2];
    applied_T1[n_steps - 1]    = applied_T1[n_steps - 2];
    applied_T2[n_steps - 1]    = applied_T2[n_steps - 2];

    double rk4_time_ms = now_ms() - t_sim0;

    /* Metricas de seguimiento */
    double sum_pos2 = 0.0, max_pos = 0.0;
    double sum_psi2 = 0.0, sum_u2 = 0.0, sum_v2 = 0.0, sum_r2 = 0.0;

    for (int i = 0; i < n_steps; i++) {
        double dx = real_x[i] - flat[i].eta[0];
        double dy = real_y[i] - flat[i].eta[1];
        double err_pos = hypot(dx, dy);
        sum_pos2 += err_pos * err_pos;
        if (err_pos > max_pos) max_pos = err_pos;

        double dpsi = atan2(sin(real_psi[i] - flat[i].eta[2]), cos(real_psi[i] - flat[i].eta[2]));
        sum_psi2 += dpsi * dpsi;

        double du = real_u[i] - flat[i].nu[0];
        sum_u2 += du * du;

        double dv = real_v[i] - flat[i].nu[1];
        sum_v2 += dv * dv;

        double dr = real_r[i] - flat[i].nu[2];
        sum_r2 += dr * dr;
    }

    double rmse_pos = sqrt(sum_pos2 / (double)n_steps);
    double rmse_psi_rad = sqrt(sum_psi2 / (double)n_steps);
    double rmse_psi_deg = rmse_psi_rad * 180.0 / M_PI;
    double rmse_u = sqrt(sum_u2 / (double)n_steps);
    double rmse_v = sqrt(sum_v2 / (double)n_steps);
    double rmse_r = sqrt(sum_r2 / (double)n_steps);

    printf("\n==================================================\n");
    printf("CASE 3 (C) OPEN-LOOP TRAJECTORY METRICS (9-PARAM PLANT):\n");
    printf("  - Trajectory Solver Time (NLP): %.4f ms\n", nlp_time_ms);
    printf("  - Simulation Time (RK4):        %.4f ms\n", rk4_time_ms);
    printf("  - Position RMSE:                %.4f m\n", rmse_pos);
    printf("  - Max Position Error:           %.4f m\n", max_pos);
    printf("  - Heading RMSE:                 %.4f rad (%.2f deg)\n", rmse_psi_rad, rmse_psi_deg);
    printf("  - Surge Velocity (u) RMSE:      %.4f m/s\n", rmse_u);
    printf("  - Sway Velocity (v) RMSE:       %.4f m/s\n", rmse_v);
    printf("  - Yaw Rate (r) RMSE:            %.4f rad/s\n", rmse_r);
    printf("==================================================\n");

    /* Guardar openloop_metrics.json */
    FILE *f_json = fopen("openloop_metrics.json", "w");
    if (f_json) {
        fprintf(f_json, "{\n");
        fprintf(f_json, "    \"case\": \"Case 3 (C)\",\n");
        fprintf(f_json, "    \"solver_type\": \"Ipopt C API (6-Param Flatness on 9-Param Plant)\",\n");
        fprintf(f_json, "    \"trajectory_solver_time_ms\": %.4f,\n", nlp_time_ms);
        fprintf(f_json, "    \"simulation_time_ms\": %.4f,\n", rk4_time_ms);
        fprintf(f_json, "    \"tracking_error\": {\n");
        fprintf(f_json, "        \"rmse_position_m\": %.6f,\n", rmse_pos);
        fprintf(f_json, "        \"max_position_error_m\": %.6f,\n", max_pos);
        fprintf(f_json, "        \"rmse_heading_rad\": %.6f,\n", rmse_psi_rad);
        fprintf(f_json, "        \"rmse_heading_deg\": %.6f,\n", rmse_psi_deg);
        fprintf(f_json, "        \"rmse_surge_u_mps\": %.6f,\n", rmse_u);
        fprintf(f_json, "        \"rmse_sway_v_mps\": %.6f,\n", rmse_v);
        fprintf(f_json, "        \"rmse_yaw_rate_r_radps\": %.6f\n", rmse_r);
        fprintf(f_json, "    }\n");
        fprintf(f_json, "}\n");
        fclose(f_json);
    }

    /* Guardar CSV para graficar con plot_results.py */
    const char *csv_path = "case3_openloop_results.csv";
    FILE *fp = fopen(csv_path, "w");
    if (fp) {
        fprintf(fp, "t,x_plan,y_plan,psi_plan,u_plan,v_plan,r_plan,"
                    "x_real,y_real,psi_real,u_real,v_real,r_real,"
                    "tau_u_plan,tau_r_plan,tau_u_applied,tau_r_applied,"
                    "T1_plan,T2_plan,T1_applied,T2_applied\n");
        for (int i = 0; i < n_steps; i++) {
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i],
                    flat[i].eta[0], flat[i].eta[1], flat[i].eta[2],
                    flat[i].nu[0], flat[i].nu[1], flat[i].nu[2],
                    real_x[i], real_y[i], real_psi[i],
                    real_u[i], real_v[i], real_r[i],
                    flat[i].tau_plan[0], flat[i].tau_plan[1],
                    applied_tau_u[i], applied_tau_r[i],
                    flat[i].T_plan[0], flat[i].T_plan[1],
                    applied_T1[i], applied_T2[i]);
        }
        fclose(fp);
        printf("[Simulate] Resultados de lazo abierto guardados en: %s\n", csv_path);
    }

    free(t_sim); free(pos); free(vel); free(acc); free(jerk); free(flat);
    free(real_x); free(real_y); free(real_psi);
    free(real_u); free(real_v); free(real_r);
    free(applied_tau_u); free(applied_tau_r); free(applied_T1); free(applied_T2);

    return 0;
}
