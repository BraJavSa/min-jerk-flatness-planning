/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
#include "usv_params.h"
#include <math.h>

static double clip(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

double thrust_from_cmd_richards(double cmd) {
    double T;
    if (cmd > 0.01) {
        T = A_POS + (K_POS - A_POS) / pow(C_POS + exp(-B_POS * (cmd - M_POS)), 1.0 / V_POS);
    } else if (cmd < -0.01) {
        T = A_NEG + (K_NEG - A_NEG) / pow(C_NEG + exp(-B_NEG * (cmd - M_NEG)), 1.0 / V_NEG);
    } else {
        T = 0.0;
    }
    return clip(T, T_MIN, T_MAX);
}

double cmd_from_thrust_richards(double T_target) {
    double T_val = clip(T_target, T_MIN, T_MAX);
    if (fabs(T_val) < 1e-3) {
        return 0.0;
    }
    if (T_val > 0.0) {
        double val = pow((K_POS - A_POS) / (T_val - A_POS), V_POS) - C_POS;
        if (val <= 0.0) return 1.0;
        double c = M_POS - (1.0 / B_POS) * log(val);
        return clip(c, 0.0, 1.0);
    } else {
        double val = pow((K_NEG - A_NEG) / (T_val - A_NEG), V_NEG) - C_NEG;
        if (val <= 0.0) return -1.0;
        double c = M_NEG - (1.0 / B_NEG) * log(val);
        return clip(c, -1.0, 0.0);
    }
}
