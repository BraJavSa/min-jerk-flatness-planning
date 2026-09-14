#ifndef NMPC_FLATNESS_HPP
#define NMPC_FLATNESS_HPP

#include <array>
#include <vector>
#include <memory>

struct NmpcWeights {
    double q_pos  = 50.0;
    double q_yaw  = 20.0;
    double q_u    = 1.0;
    double q_r    = 3.0;
    double w_jerk = 0.0;
    double r_tau  = 1.0e-3;
};

class NmpcFlatness {
public:
    static constexpr int NX = 6;
    static constexpr int NU = 2;
    static constexpr int N  = 30;

    using Weights = NmpcWeights;

    explicit NmpcFlatness(double dt, const Weights &w = Weights());
    ~NmpcFlatness();

    NmpcFlatness(const NmpcFlatness &) = delete;
    NmpcFlatness &operator=(const NmpcFlatness &) = delete;
    NmpcFlatness(NmpcFlatness &&) noexcept;
    NmpcFlatness &operator=(NmpcFlatness &&) noexcept;

    std::array<double, NU> solve(const std::array<double, NX> &x0,
                                 const std::vector<std::array<double, 3>> &eta_ref,
                                 const std::vector<std::array<double, 3>> &nu_ref,
                                 const std::array<double, NU> &u_prev,
                                 double *solve_time_ms = nullptr,
                                 const std::vector<std::array<double, 2>> &tau_ref = {});

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif

