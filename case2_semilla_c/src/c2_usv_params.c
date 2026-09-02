#include "c2_usv_params.h"
#include <math.h>

/* Fallback constants from c2_usv_params.py (used when the JSON dynamic-model
 * files are not present, which is the case for this standalone C port). */
const double m11_p = 50.53;
const double m22_p = 85.08;
const double m33_p = 17.25;
const double Xu_p  = 151.56;
const double Yv_p  = 133.77;
const double Nr_p  = 34.57;
const double dP_p  = 0.26;

static const double A_POS = -12.07098855, K_POS = 73.72259622, B_POS = 14.20242467,
                     M_POS = 0.99474311,  V_POS = 6.83239913,  C_POS = 1.0;
static const double A_NEG = -70.9610860,  K_NEG = 7.47710923,  B_NEG = 2.69365001,
                     M_NEG = -3.79303820, V_NEG = 4.09908178e-04, C_NEG = 1.0;

const double T_MAX = 65.92;
const double T_MIN = -49.38;
const double DT_SIM = 1.0 / 30.0;

double thrust_from_cmd_richards(double cmd) {
    double T;
    if (cmd > 0.01)
        T = A_POS + (K_POS - A_POS) / pow(C_POS + exp(-B_POS * (cmd - M_POS)), 1.0 / V_POS);
    else if (cmd < -0.01)
        T = A_NEG + (K_NEG - A_NEG) / pow(C_NEG + exp(-B_NEG * (cmd - M_NEG)), 1.0 / V_NEG);
    else
        T = 0.0;
    if (T > T_MAX) T = T_MAX;
    if (T < T_MIN) T = T_MIN;
    return T;
}

double cmd_from_thrust_richards(double T_target) {
    double T_val = T_target;
    if (T_val > T_MAX) T_val = T_MAX;
    if (T_val < T_MIN) T_val = T_MIN;

    if (fabs(T_val) < 1e-3) return 0.0;

    if (T_val > 0) {
        double val = pow((K_POS - A_POS) / (T_val - A_POS), V_POS) - C_POS;
        if (val <= 0) return 1.0;
        double c = M_POS - (1.0 / B_POS) * log(val);
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        return c;
    } else {
        double val = pow((K_NEG - A_NEG) / (T_val - A_NEG), V_NEG) - C_NEG;
        if (val <= 0) return -1.0;
        double c = M_NEG - (1.0 / B_NEG) * log(val);
        if (c < -1.0) c = -1.0;
        if (c > 0.0) c = 0.0;
        return c;
    }
}
