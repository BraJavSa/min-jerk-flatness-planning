#ifndef USV_PARAMS_H
#define USV_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

// Modelo USV de 6 parámetros (Fictitious-Input Full Actuation)
extern const double m11_6;
extern const double m22_6;
extern const double m33_6;
extern const double Xu_6;
extern const double Yv_6;
extern const double Nr_6;
extern const double dP_6;

// Alias de compatibilidad
#define m11_real m11_6
#define m22_real m22_6
#define m33_real m33_6
#define Xu_real  Xu_6
#define Yv_real  Yv_6
#define Nr_real  Nr_6
#define dP       dP_6

extern const double SURGE_GAIN;
extern const double YAW_ARM;

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

static inline double thrust_from_cmd(double cmd) {
    return thrust_from_cmd_richards(cmd);
}

static inline double cmd_from_thrust(double T_target) {
    return cmd_from_thrust_richards(T_target);
}

#ifdef __cplusplus
}
#endif

#endif
