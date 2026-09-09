/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 * Fully using the identified 9-parameter nonlinear model from
 * identified_models.json ("full-dynamics").
 */
#ifndef USV_PARAMS_H
#define USV_PARAMS_H

/* ---------------------------------------------------------------------
 * 9-Parameter full nonlinear model from identified_models.json
 * -> "full-dynamics"
 * --------------------------------------------------------------------- */
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

/* ---------------------------------------------------------------------
 * Sampling
 * --------------------------------------------------------------------- */
extern const double SAMPLE_RATE_HZ;
extern const double DT_SIM;

/* ---------------------------------------------------------------------
 * Thruster curve (Richards curve, positive/negative branches)
 * --------------------------------------------------------------------- */
extern const double A_POS, K_POS, B_POS, M_POS, V_POS, C_POS;
extern const double A_NEG, K_NEG, B_NEG, M_NEG, V_NEG, C_NEG;
extern const double T_MAX;
extern const double T_MIN;

/* Scalar thruster curve: command -> thrust */
double thrust_from_cmd_richards(double cmd);

/* Scalar inverse thruster curve: thrust -> command */
double cmd_from_thrust_richards(double T_target);

/* Vectorized versions (arrays of length n) */
void thrust_from_cmd_array(const double *cmd, double *T_out, int n);
void cmd_from_thrust_array(const double *T_array, double *cmd_out, int n);

#endif /* USV_PARAMS_H */
