#include "nmpc_flatness.hpp"
#include "usv_params.h"
#include <casadi/casadi.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

static inline double wrap_to_pi(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

static inline double clamp_val(double val, double min_val, double max_val) {
    return std::max(min_val, std::min(val, max_val));
}

static double bspline_eval(double t, int i, int p, int der, const std::vector<double> &knots) {
    if (der > 0) {
        if (p == 0) return 0.0;
        double c1 = 0.0, c2 = 0.0;
        double d1 = knots[i + p] - knots[i];
        if (d1 > 1e-12) {
            c1 = (static_cast<double>(p) / d1) * bspline_eval(t, i, p - 1, der - 1, knots);
        }
        double d2 = knots[i + p + 1] - knots[i + 1];
        if (d2 > 1e-12) {
            c2 = (static_cast<double>(p) / d2) * bspline_eval(t, i + 1, p - 1, der - 1, knots);
        }
        return c1 - c2;
    }
    if (p == 0) {
        if ((knots[i] <= t && t < knots[i + 1]) ||
            (t >= knots.back() - 1e-12 && knots[i] <= t && t <= knots[i + 1] && knots[i + 1] >= knots.back() - 1e-12)) {
            return 1.0;
        }
        return 0.0;
    }
    double c1 = 0.0, c2 = 0.0;
    double d1 = knots[i + p] - knots[i];
    if (d1 > 1e-12) {
        c1 = ((t - knots[i]) / d1) * bspline_eval(t, i, p - 1, 0, knots);
    }
    double d2 = knots[i + p + 1] - knots[i + 1];
    if (d2 > 1e-12) {
        c2 = ((knots[i + p + 1] - t) / d2) * bspline_eval(t, i + 1, p - 1, 0, knots);
    }
    return c1 + c2;
}

static casadi::DM build_basis_matrix(const std::vector<double> &ts, int der,
                                     int n_ctrl, int degree, double tf,
                                     const std::vector<double> &knots) {
    int n_ts = static_cast<int>(ts.size());
    casadi::DM B = casadi::DM::zeros(n_ts, n_ctrl);
    for (int r = 0; r < n_ts; ++r) {
        double t_eval = std::max(0.0, std::min(ts[r], tf - 1e-9));
        if (std::abs(ts[r]) < 1e-12) t_eval = 0.0;
        for (int c = 0; c < n_ctrl; ++c) {
            B(r, c) = bspline_eval(t_eval, c, degree, der, knots);
        }
    }
    return B;
}

struct NmpcFlatness::Impl {
    double dt;
    double tf;
    int degree = 4;
    int n_ctrl = 12;
    int n_colloc = 20;
    double epsilon = 0.15;
    Weights w;

    std::vector<double> knots;
    std::vector<double> t_colloc;

    casadi::DM B0, B1, B2, B3, B_bnd0, B_bnd1;
    casadi::Opti opti;
    casadi::MX P;
    casadi::MX P_x0, P_xref, P_yref, P_psiref, P_uref, P_rref, P_turef, P_trref;
    casadi::MX T1, T2, tau_u, tau_v, tau_r;

    casadi::DM P_prev;
    bool has_prev = false;

    Impl(double dt_in, const Weights &w_in)
        : dt(dt_in), tf(NmpcFlatness::N * dt_in), w(w_in) {
        init_splines();
        build_opti();
        warmup();
    }

    void init_splines() {
        int n_internal = n_ctrl - degree - 1;
        knots.clear();
        knots.reserve(degree + 1 + n_internal + degree + 1);
        for (int i = 0; i < degree + 1; ++i) knots.push_back(0.0);
        for (int i = 1; i <= n_internal; ++i) {
            knots.push_back((tf * static_cast<double>(i)) / (n_internal + 1));
        }
        for (int i = 0; i < degree + 1; ++i) knots.push_back(tf);

        t_colloc.resize(n_colloc);
        for (int i = 0; i < n_colloc; ++i) {
            t_colloc[i] = (tf * static_cast<double>(i)) / (n_colloc - 1);
        }

        B0 = build_basis_matrix(t_colloc, 0, n_ctrl, degree, tf, knots);
        B1 = build_basis_matrix(t_colloc, 1, n_ctrl, degree, tf, knots);
        B2 = build_basis_matrix(t_colloc, 2, n_ctrl, degree, tf, knots);
        B3 = build_basis_matrix(t_colloc, 3, n_ctrl, degree, tf, knots);
        B_bnd0 = build_basis_matrix({0.0}, 0, n_ctrl, degree, tf, knots);
        B_bnd1 = build_basis_matrix({0.0}, 1, n_ctrl, degree, tf, knots);
    }

    void build_opti() {
        using namespace casadi;
        opti = Opti();

        P = opti.variable(n_ctrl, 3);
        MX Px = P(Slice(), 0);
        MX Py = P(Slice(), 1);
        MX Ppsi = P(Slice(), 2);

        P_x0 = opti.parameter(6);
        P_xref = opti.parameter(n_colloc);
        P_yref = opti.parameter(n_colloc);
        P_psiref = opti.parameter(n_colloc);
        P_uref = opti.parameter(n_colloc);
        P_rref = opti.parameter(n_colloc);
        P_turef = opti.parameter(n_colloc);
        P_trref = opti.parameter(n_colloc);

        MX x = mtimes(B0, Px);
        MX y = mtimes(B0, Py);
        MX psi = mtimes(B0, Ppsi);

        MX dx = mtimes(B1, Px);
        MX dy = mtimes(B1, Py);
        MX dpsi = mtimes(B1, Ppsi);

        MX ddx = mtimes(B2, Px);
        MX ddy = mtimes(B2, Py);
        MX ddpsi = mtimes(B2, Ppsi);

        MX u = dx * cos(psi) + dy * sin(psi);
        MX v = -dx * sin(psi) + dy * cos(psi);
        MX r = dpsi;

        MX du = ddx * cos(psi) + ddy * sin(psi) + v * r;
        MX dv = -ddx * sin(psi) + ddy * cos(psi) - u * r;
        MX dr = ddpsi;

        // Modelo de 6 parámetros (Fictitious-Input Full Actuation)
        tau_u = m11_6 * du - m22_6 * v * r + Xu_6 * u;
        tau_v = m22_6 * dv + m11_6 * u * r + Yv_6 * v;
        tau_r = m33_6 * dr - (m11_6 - m22_6) * u * v + Nr_6 * r;

        T1 = 0.5 * (tau_u / SURGE_GAIN + tau_r / (2.0 * YAW_ARM));
        T2 = 0.5 * (tau_u / SURGE_GAIN - tau_r / (2.0 * YAW_ARM));

        MX x0 = P_x0(0), y0 = P_x0(1), psi0 = P_x0(2);
        MX u0 = P_x0(3), v0 = P_x0(4), r0 = P_x0(5);

        opti.subject_to(mtimes(B_bnd0, Px) == x0);
        opti.subject_to(mtimes(B_bnd0, Py) == y0);
        opti.subject_to(mtimes(B_bnd0, Ppsi) == psi0);

        opti.subject_to(mtimes(B_bnd1, Px) == u0 * cos(psi0) - v0 * sin(psi0));
        opti.subject_to(mtimes(B_bnd1, Py) == u0 * sin(psi0) + v0 * cos(psi0));
        opti.subject_to(mtimes(B_bnd1, Ppsi) == r0);

        // Restricción de subactuación (fuerza de deriva)
        opti.subject_to(opti.bounded(-epsilon, tau_v, epsilon));

        // Límites de actuadores
        opti.subject_to(opti.bounded(T_MIN, T1, T_MAX));
        opti.subject_to(opti.bounded(T_MIN, T2, T_MAX));

        MX err_pos = sumsqr(x - P_xref) + sumsqr(y - P_yref);
        MX err_yaw = sumsqr(psi - P_psiref);
        MX err_u   = sumsqr(u - P_uref);
        MX err_r   = sumsqr(r - P_rref);
        MX err_tau = sumsqr(tau_u - P_turef) + sumsqr(tau_r - P_trref);

        MX cost = w.q_pos * err_pos + w.q_yaw * err_yaw + w.q_u * err_u + w.q_r * err_r;
        if (w.r_tau > 0.0) {
            cost += w.r_tau * err_tau;
        }
        if (w.w_jerk > 0.0) {
            MX jx = mtimes(B3, Px);
            MX jy = mtimes(B3, Py);
            MX jpsi = mtimes(B3, Ppsi);
            cost += w.w_jerk * (sumsqr(jx) + sumsqr(jy) + 5.0 * sumsqr(jpsi));
        }
        opti.minimize(cost);

        Dict p_opts, s_opts;
        p_opts["expand"] = true;
        p_opts["print_time"] = false;
        s_opts["max_iter"] = 30;
        s_opts["print_level"] = 0;
        s_opts["tol"] = 1e-3;
        s_opts["acceptable_tol"] = 1e-2;
        s_opts["warm_start_init_point"] = "yes";
        opti.solver("ipopt", p_opts, s_opts);
    }

    void warmup() {
        std::array<double, 6> dummy_x0{0.0, 0.0, 0.0, 0.1, 0.0, 0.0};
        std::vector<std::array<double, 3>> dummy_eta(NmpcFlatness::N, {0.0, 0.0, 0.0});
        std::vector<std::array<double, 3>> dummy_nu(NmpcFlatness::N, {0.1, 0.0, 0.0});
        std::vector<std::array<double, 2>> dummy_tau(NmpcFlatness::N, {0.0, 0.0});
        std::array<double, 2> dummy_u{0.0, 0.0};
        solve_internal(dummy_x0, dummy_eta, dummy_nu, dummy_u, nullptr, dummy_tau);
        has_prev = false;
    }

    std::array<double, 2> solve_internal(const std::array<double, 6> &x0,
                                         const std::vector<std::array<double, 3>> &eta_ref,
                                         const std::vector<std::array<double, 3>> &nu_ref,
                                         const std::array<double, 2> &u_prev,
                                         double *solve_time_ms,
                                         const std::vector<std::array<double, 2>> &tau_ref) {
        auto t0 = std::chrono::steady_clock::now();

        int n_avail = static_cast<int>(eta_ref.size());
        std::vector<double> t_ref(n_avail);
        for (int i = 0; i < n_avail; ++i) {
            t_ref[i] = (tf * static_cast<double>(i)) / std::max(1, n_avail - 1);
        }

        std::vector<double> xref_col(n_colloc);
        std::vector<double> yref_col(n_colloc);
        std::vector<double> uref_col(n_colloc);
        std::vector<double> rref_col(n_colloc);
        std::vector<double> psiref_unwrapped(n_avail);

        psiref_unwrapped[0] = eta_ref[0][2];
        for (int i = 1; i < n_avail; ++i) {
            double d = eta_ref[i][2] - eta_ref[i - 1][2];
            while (d > M_PI) d -= 2.0 * M_PI;
            while (d < -M_PI) d += 2.0 * M_PI;
            psiref_unwrapped[i] = psiref_unwrapped[i - 1] + d;
        }

        double psi_r0 = eta_ref[0][2];
        double dpsi_curr = wrap_to_pi(x0[2] - psi_r0);
        double psi0_unwrapped = psi_r0 + dpsi_curr;

        std::vector<double> psiref_col(n_colloc);
        std::vector<double> turef_col(n_colloc, 0.0);
        std::vector<double> trref_col(n_colloc, 0.0);
        bool has_tau = (tau_ref.size() >= static_cast<size_t>(n_avail));

        auto interp1d = [&](const std::vector<double> &xp, const auto &extract_val, double x_query) -> double {
            if (x_query <= xp.front()) return extract_val(0);
            if (x_query >= xp.back()) return extract_val(n_avail - 1);
            int idx = 0;
            while (idx < n_avail - 1 && xp[idx + 1] < x_query) ++idx;
            double alpha = (x_query - xp[idx]) / (xp[idx + 1] - xp[idx]);
            return extract_val(idx) + alpha * (extract_val(idx + 1) - extract_val(idx));
        };

        for (int i = 0; i < n_colloc; ++i) {
            double tc = t_colloc[i];
            xref_col[i] = interp1d(t_ref, [&](int k) { return eta_ref[k][0]; }, tc);
            yref_col[i] = interp1d(t_ref, [&](int k) { return eta_ref[k][1]; }, tc);
            uref_col[i] = interp1d(t_ref, [&](int k) { return nu_ref[k][0]; }, tc);
            rref_col[i] = interp1d(t_ref, [&](int k) { return nu_ref[k][2]; }, tc);
            psiref_col[i] = interp1d(t_ref, [&](int k) { return psiref_unwrapped[k]; }, tc);
            if (has_tau) {
                turef_col[i] = interp1d(t_ref, [&](int k) { return tau_ref[k][0]; }, tc);
                trref_col[i] = interp1d(t_ref, [&](int k) { return tau_ref[k][1]; }, tc);
            }
        }

        std::vector<double> x0_param = {
            x0[0], x0[1], psi0_unwrapped, x0[3], x0[4], x0[5]
        };

        opti.set_value(P_x0, x0_param);
        opti.set_value(P_xref, xref_col);
        opti.set_value(P_yref, yref_col);
        opti.set_value(P_psiref, psiref_col);
        opti.set_value(P_uref, uref_col);
        opti.set_value(P_rref, rref_col);
        opti.set_value(P_turef, turef_col);
        opti.set_value(P_trref, trref_col);

        if (has_prev) {
            opti.set_initial(P, P_prev);
        } else {
            casadi::DM init_P = casadi::DM::zeros(n_ctrl, 3);
            for (int i = 0; i < n_ctrl; ++i) {
                double frac = static_cast<double>(i) / std::max(1, n_ctrl - 1);
                init_P(i, 0) = x0_param[0] + frac * (xref_col.back() - x0_param[0]);
                init_P(i, 1) = x0_param[1] + frac * (yref_col.back() - x0_param[1]);
                init_P(i, 2) = psi0_unwrapped + frac * (psiref_col.back() - psi0_unwrapped);
            }
            opti.set_initial(P, init_P);
        }

        double T1_raw = 0.0, T2_raw = 0.0;
        try {
            auto sol = opti.solve();
            P_prev = sol.value(P);
            has_prev = true;
            T1_raw = double(sol.value(T1)(0));
            T2_raw = double(sol.value(T2)(0));
        } catch (...) {
            P_prev = opti.debug().value(P);
            has_prev = true;
            T1_raw = double(opti.debug().value(T1)(0));
            T2_raw = double(opti.debug().value(T2)(0));
        }

        double T1_cmd = clamp_val(T1_raw, T_MIN, T_MAX);
        double T2_cmd = clamp_val(T2_raw, T_MIN, T_MAX);

        if (solve_time_ms) {
            auto t1 = std::chrono::steady_clock::now();
            *solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        return {T1_cmd, T2_cmd};
    }
};

NmpcFlatness::NmpcFlatness(double dt, const Weights &w)
    : impl_(std::make_unique<Impl>(dt, w)) {}

NmpcFlatness::~NmpcFlatness() = default;

NmpcFlatness::NmpcFlatness(NmpcFlatness &&) noexcept = default;
NmpcFlatness &NmpcFlatness::operator=(NmpcFlatness &&) noexcept = default;

void NmpcFlatness::reset() {
    if (impl_) impl_->has_prev = false;
}

std::array<double, NmpcFlatness::NU> NmpcFlatness::solve(
    const std::array<double, NX> &x0,
    const std::vector<std::array<double, 3>> &eta_ref,
    const std::vector<std::array<double, 3>> &nu_ref,
    const std::array<double, NU> &u_prev,
    double *solve_time_ms,
    const std::vector<std::array<double, 2>> &tau_ref) {

    if (!impl_) return {0.0, 0.0};
    return impl_->solve_internal(x0, eta_ref, nu_ref, u_prev, solve_time_ms, tau_ref);
}
