#include "usv_params.h"
#include <math.h>

const double m11_9 = 27.6527951473284;
const double m22_9 = 30.76677961987949;
const double m33_9 = 6.422382544661223;
const double Xu_9  = 15.657929738187262;
const double Xuu_9 = 14.389382088099678;
const double Yv_9  = 42.20370985895385;
const double Yvv_9 = 0.7067569214632512;
const double Nr_9  = 4.940560878487763;
const double Nrr_9 = 3.1501144317267036;
const double dP_9  = 0.26;

const double SAMPLE_RATE_HZ = 30.0;
const double DT_SIM = 1.0 / 30.0;

const double A_POS = -12.07098855, K_POS = 73.72259622, B_POS = 14.20242467;
const double M_POS = 0.99474311,   V_POS = 6.83239913,  C_POS = 1.0;
const double A_NEG = -70.9610860,  K_NEG = 7.47710923,  B_NEG = 2.69365001;
const double M_NEG = -3.79303820,  V_NEG = 4.09908178e-04, C_NEG = 1.0;
const double T_MAX = 65.92;
const double T_MIN = -49.38;

double thrust_from_cmd_richards(double cmd)
{
    double T;

    if (cmd > 0.01) {
        T = A_POS + (K_POS - A_POS) / pow(C_POS + exp(-B_POS * (cmd - M_POS)), 1.0 / V_POS);
    } else if (cmd < -0.01) {
        T = A_NEG + (K_NEG - A_NEG) / pow(C_NEG + exp(-B_NEG * (cmd - M_NEG)), 1.0 / V_NEG);
    } else {
        T = 0.0;
    }

    if (T < T_MIN) T = T_MIN;
    if (T > T_MAX) T = T_MAX;
    return T;
}

double cmd_from_thrust_richards(double T_target)
{
    double T_val = T_target;
    double val, c;

    if (T_val < T_MIN) T_val = T_MIN;
    if (T_val > T_MAX) T_val = T_MAX;

    if (fabs(T_val) < 1e-3) {
        return 0.0;
    }

    if (T_val > 0.0) {
        val = pow((K_POS - A_POS) / (T_val - A_POS), V_POS) - C_POS;
        if (val <= 0.0) {
            return 1.0;
        }
        c = M_POS - (1.0 / B_POS) * log(val);
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        return c;
    } else {
        val = pow((K_NEG - A_NEG) / (T_val - A_NEG), V_NEG) - C_NEG;
        if (val <= 0.0) {
            return -1.0;
        }
        c = M_NEG - (1.0 / B_NEG) * log(val);
        if (c < -1.0) c = -1.0;
        if (c > 0.0) c = 0.0;
        return c;
    }
}

void thrust_from_cmd_array(const double *cmd, double *T_out, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        T_out[i] = thrust_from_cmd_richards(cmd[i]);
    }
}

void cmd_from_thrust_array(const double *T_array, double *cmd_out, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        cmd_out[i] = cmd_from_thrust_richards(T_array[i]);
    }
}
