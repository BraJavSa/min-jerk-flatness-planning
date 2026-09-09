#include <math.h>
#include "usv_params.h"

static const double A_POS = 1e-06, K_POS = 40.0209, B_POS = 2.6249, V_POS = 0.1615, C_POS = 0.9432, M_POS = 1e-05;
static const double A_NEG = -31.499, K_NEG = -1e-05, B_NEG = 3.6986, V_NEG = 0.3264, C_NEG = 0.9713, M_NEG = -1.0;

double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

double thrust_from_cmd(double cmd) {
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

double cmd_from_thrust(double T_target) {
    double T_val = clip(T_target, T_MIN, T_MAX);
    if (fabs(T_val) < 1e-3) return 0.0;
    if (T_val > 0) {
        double val = pow((K_POS - A_POS) / (T_val - A_POS), V_POS) - C_POS;
        if (val <= 0) return 1.0;
        double c = M_POS - (1.0 / B_POS) * log(val);
        return clip(c, 0.0, 1.0);
    } else {
        double val = pow((K_NEG - A_NEG) / (T_val - A_NEG), V_NEG) - C_NEG;
        if (val <= 0) return -1.0;
        double c = M_NEG - (1.0 / B_NEG) * log(val);
        return clip(c, -1.0, 0.0);
    }
}
