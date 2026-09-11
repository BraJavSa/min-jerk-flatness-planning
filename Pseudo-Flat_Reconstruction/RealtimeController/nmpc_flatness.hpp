#ifndef NMPC_FLATNESS_HPP
#define NMPC_FLATNESS_HPP

#include <array>
#include <vector>

struct NmpcWeights {
    double q_pos   = 200.0;
    double q_lin   = 1.0;
    double q_yawr  = 5.0;
    double r_eff   = 1.0e-4;
    double rd_rate = 1.0e-3;
};

class NmpcFlatness {
public:
    static constexpr int NX = 6;
    static constexpr int NU = 2;
    static constexpr int N  = 30;

    using Weights = NmpcWeights;

    explicit NmpcFlatness(double dt, const Weights &w = Weights());

    std::array<double, NU> solve(const std::array<double, NX> &x0,
                                 const std::vector<std::array<double, 3>> &eta_ref,
                                 const std::vector<std::array<double, 3>> &nu_ref,
                                 const std::array<double, NU> &u_prev,
                                 double *solve_time_ms = nullptr);

    void reset();

private:
    double dt_;
    Weights w_;
};

#endif
