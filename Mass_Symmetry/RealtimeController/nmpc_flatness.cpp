#include "nmpc_flatness.hpp"
#include "usv_params.h"
#include "flatness_reconstruct.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

static inline double clamp_val(double val, double min_val, double max_val) {
    return std::max(min_val, std::min(val, max_val));
}

NmpcFlatness::NmpcFlatness(double dt, const Weights &w) : dt_(dt), w_(w) {

}

void NmpcFlatness::reset() {
}

std::array<double, NmpcFlatness::NU> NmpcFlatness::solve(
    const std::array<double, NX> &x0,
    const std::vector<std::array<double, 3>> &eta_ref,
    const std::vector<std::array<double, 3>> &nu_ref,
    const std::array<double, NU> &u_prev,
    double *solve_time_ms) {

    auto t0 = std::chrono::steady_clock::now();

    const int n_pts = N;
    int n_avail = static_cast<int>(eta_ref.size());

    double x_curr = x0[0], y_curr = x0[1], psi_curr = x0[2];
    double u_curr = x0[3], v_curr = x0[4], r_curr = x0[5];

    double x_r0   = eta_ref[0][0], y_r0   = eta_ref[0][1], psi_r0 = eta_ref[0][2];
    double u_r0   = nu_ref[0][0],  v_r0   = nu_ref[0][1],  r_r0   = nu_ref[0][2];

    double xr_seq[N], yr_seq[N], dxr_seq[N], dyr_seq[N], ddxr_seq[N], ddyr_seq[N], t_seq[N];
    for (int k = 0; k < n_pts; k++) {
        t_seq[k] = k * dt_;
        int idx = std::min(k, n_avail - 1);
        xr_seq[k] = eta_ref[idx][0];
        yr_seq[k] = eta_ref[idx][1];

        double psir = eta_ref[idx][2];
        double ur = nu_ref[idx][0], vr = nu_ref[idx][1];
        dxr_seq[k] = ur * std::cos(psir) - vr * std::sin(psir);
        dyr_seq[k] = ur * std::sin(psir) + vr * std::cos(psir);

        int idx1 = std::min(k + 1, n_avail - 1);
        double psir1 = eta_ref[idx1][2];
        double ur1 = nu_ref[idx1][0], vr1 = nu_ref[idx1][1];
        double dxr1 = ur1 * std::cos(psir1) - vr1 * std::sin(psir1);
        double dyr1 = ur1 * std::sin(psir1) + vr1 * std::cos(psir1);

        ddxr_seq[k] = (dxr1 - dxr_seq[k]) / dt_;
        ddyr_seq[k] = (dyr1 - dyr_seq[k]) / dt_;
    }

    FlatSample samples[N];
    reconstruct_flatness_h2(xr_seq, yr_seq, dxr_seq, dyr_seq, ddxr_seq, ddyr_seq, n_pts, t_seq, samples);

    double tau_u_ff = samples[0].tau_plan[0];
    double tau_r_ff = samples[0].tau_plan[1];

    double dx = x_curr - x_r0;
    double dy = y_curr - y_r0;

    double ex =  std::cos(psi_curr) * dx + std::sin(psi_curr) * dy;
    double ey = -std::sin(psi_curr) * dx + std::cos(psi_curr) * dy;
    double eu = u_curr - u_r0;
    double er = r_curr - r_r0;

    double psi_los = -std::atan2(ey, 1.5);
    double psi_cmd = psi_r0 + psi_los;
    double dpsi = psi_curr - psi_cmd;
    while (dpsi > M_PI) dpsi -= 2.0 * M_PI;
    while (dpsi < -M_PI) dpsi += 2.0 * M_PI;

    double K_px   = 25.0 * (w_.q_pos / 200.0);
    double K_du   = 15.0 * (w_.q_lin / 1.0);
    double K_ppsi = 25.0 * std::sqrt((w_.q_pos / 200.0) * (w_.q_yawr / 5.0));
    double K_dr   = 10.0 * (w_.q_yawr / 5.0);

    double delta_tau_u = -K_px * ex - K_du * eu;
    double delta_tau_r = -K_ppsi * dpsi - K_dr * er;

    double tau_u_cmd = tau_u_ff + delta_tau_u;
    double tau_r_cmd = tau_r_ff + delta_tau_r;

    double T1_cmd = 0.5 * (tau_u_cmd / SURGE_GAIN + tau_r_cmd / (2.0 * YAW_ARM));
    double T2_cmd = 0.5 * (tau_u_cmd / SURGE_GAIN - tau_r_cmd / (2.0 * YAW_ARM));

    std::array<double, NU> u_out;
    u_out[0] = clamp_val(T1_cmd, T_MIN, T_MAX);
    u_out[1] = clamp_val(T2_cmd, T_MIN, T_MAX);

    if (solve_time_ms) {
        auto t1 = std::chrono::steady_clock::now();
        *solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return u_out;
}
