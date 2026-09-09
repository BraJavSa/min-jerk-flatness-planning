#include <stdlib.h>
#include <math.h>
#include "usv_params.h"
#include "flatness_reconstruct.h"

FlatSample reconstruct_flatness_case3_point(
    double x, double y, double psi,
    double dx, double dy, double dpsi,
    double ddx, double ddy, double ddpsi) {
    FlatSample s;

    double c_psi = cos(psi);
    double s_psi = sin(psi);

    /* Body velocities */
    double u = dx * c_psi + dy * s_psi;
    double v = -dx * s_psi + dy * c_psi;
    double r = dpsi;

    /* Body accelerations */
    double du = ddx * c_psi + ddy * s_psi + v * r;
    double dv = -ddx * s_psi + ddy * c_psi - u * r;
    double dr = ddpsi;

    /* Required forces using 6-parameter identified dynamics */
    double tau_u_raw = m11_real * du - m22_real * v * r + Xu_real * u;
    double tau_v_raw = m22_real * dv + m11_real * u * r + Yv_real * v;
    double tau_r_raw = m33_real * dr - (m11_real - m22_real) * u * v + Nr_real * r;

    /* Allocation to left (T1) and right (T2) thrusters */
    double T1_raw = 0.5 * (tau_u_raw + tau_r_raw / dP);
    double T2_raw = 0.5 * (tau_u_raw - tau_r_raw / dP);

    /* Limits */
    double tau_u_max = 2.0 * T_MAX;
    double tau_u_min = 2.0 * T_MIN;
    double tau_r_max = (T_MAX - T_MIN) * dP;
    double tau_r_min = -tau_r_max;

    double tau_u = clip(tau_u_raw, tau_u_min, tau_u_max);
    double tau_r = clip(tau_r_raw, tau_r_min, tau_r_max);

    double T1_dem = clip(T1_raw, T_MIN, T_MAX);
    double T2_dem = clip(T2_raw, T_MIN, T_MAX);

    /* Actuator inversion via Richards curve */
    double cmd_1 = cmd_from_thrust_richards(T1_dem);
    double cmd_2 = cmd_from_thrust_richards(T2_dem);

    /* Actual thrust generated */
    double T1_act = thrust_from_cmd_richards(cmd_1);
    double T2_act = thrust_from_cmd_richards(cmd_2);

    double tau_u_act = T1_act + T2_act;
    double tau_r_act = (T1_act - T2_act) * dP;

    s.eta[0] = x; s.eta[1] = y; s.eta[2] = psi;
    s.nu[0] = u; s.nu[1] = v; s.nu[2] = r;
    s.nudot[0] = du; s.nudot[1] = dv; s.nudot[2] = dr;
    s.tau_plan[0] = tau_u; s.tau_plan[1] = tau_r;
    s.tau_act[0] = tau_u_act; s.tau_act[1] = tau_r_act;
    s.tau_v_raw = tau_v_raw;
    s.T_plan[0] = T1_dem; s.T_plan[1] = T2_dem;
    s.T_act[0] = T1_act; s.T_act[1] = T2_act;
    s.cmds[0] = cmd_1; s.cmds[1] = cmd_2;

    return s;
}

void reconstruct_flatness_case3(
    const double *pos,
    const double *vel,
    const double *acc,
    int n,
    FlatSample *out) {
    for (int i = 0; i < n; i++) {
        out[i] = reconstruct_flatness_case3_point(
            pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2],
            vel[i * 3 + 0], vel[i * 3 + 1], vel[i * 3 + 2],
            acc[i * 3 + 0], acc[i * 3 + 1], acc[i * 3 + 2]);
    }
}
