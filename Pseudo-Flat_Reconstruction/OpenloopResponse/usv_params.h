#ifndef USV_PARAMS_H
#define USV_PARAMS_H

extern const double m11_9;
extern const double m22_9;
extern const double m33_9;
extern const double Xu_9;
extern const double Xuu_9;
extern const double Yv_9;
extern const double Yvv_9;
extern const double Nr_9;
extern const double Nrr_9;
extern const double dP_9;

extern const double SAMPLE_RATE_HZ;
extern const double DT_SIM;

extern const double A_POS, K_POS, B_POS, M_POS, V_POS, C_POS;
extern const double A_NEG, K_NEG, B_NEG, M_NEG, V_NEG, C_NEG;
extern const double T_MAX;
extern const double T_MIN;

double thrust_from_cmd_richards(double cmd);

double cmd_from_thrust_richards(double T_target);

void thrust_from_cmd_array(const double *cmd, double *T_out, int n);
void cmd_from_thrust_array(const double *T_array, double *cmd_out, int n);

#endif
